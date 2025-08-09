\
# server.py - WebRTC + FastAPI server for live SER + Gemma LLM + TTS streaming + telemetry
import os, io, asyncio, time, json, base64
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import soundfile as sf
import torch

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# local loader for LLMs (expects models_loader.py present)
from models_loader import load_llm

# TTS (Coqui)
from TTS.api import TTS

app = FastAPI()

# ------------------ Configuration ------------------
SER_MODEL = os.getenv("SER_MODEL", "j-hartmann/emotion-english-speech-distilroberta-base")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gemma")
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
SER_LOOP_INTERVAL = float(os.getenv("SER_LOOP_INTERVAL", "1.2"))  # seconds between SER inference windows
SER_MIN_BYTES = int(os.getenv("SER_MIN_BYTES", str(16000*2)))  # min bytes in buffer to run SER
# ---------------------------------------------------

# Load SER model
print(f"[server] Loading SER model: {SER_MODEL}")
try:
    ser_processor = Wav2Vec2Processor.from_pretrained(SER_MODEL)
    ser_model = Wav2Vec2ForSequenceClassification.from_pretrained(SER_MODEL)
    ser_device = "cuda" if torch.cuda.is_available() else "cpu"
    ser_model.to(ser_device)
    print(f"[server] SER loaded on {ser_device}")
except Exception as e:
    print("[server] Failed to load SER model:", e)
    ser_processor = None
    ser_model = None
    ser_device = "cpu"

# Load LLM (Gemma or fallback). models_loader.load_llm will handle quantization/fallback.
print(f"[server] Loading LLM: {DEFAULT_LLM}")
try:
    tokenizer_llm, model_llm, llm_device = load_llm(DEFAULT_LLM)
    print(f"[server] LLM loaded on device: {llm_device}")
except Exception as e:
    print("[server] LLM load failed:", e)
    tokenizer_llm, model_llm, llm_device = (None, None, None)

# Load TTS
try:
    tts = TTS(model_name=TTS_MODEL, progress_bar=False, gpu=False)
    print("[server] TTS loaded")
except Exception as e:
    print("[server] TTS load failed:", e)
    tts = None

def infer_ser_from_wav_bytes(wav_bytes):
    if ser_model is None or ser_processor is None:
        return ("unknown", 0.0)
    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        target_sr = getattr(ser_processor.feature_extractor, "sampling_rate", 16000)
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        inputs = ser_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(ser_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(ser_device)
        with torch.no_grad():
            logits = ser_model(input_values, attention_mask=attention_mask).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(probs.argmax())
        label = ser_model.config.id2label[idx] if hasattr(ser_model.config, "id2label") else str(idx)
        conf = float(probs[idx])
        return (label, conf)
    except Exception as e:
        print("[server] SER inference error:", e)
        return ("error", 0.0)

def synthesize_tts_pcm_bytes(text, sample_rate=22050):
    if tts is None:
        return None, sample_rate
    wav = tts.tts(text)
    wav = np.array(wav, dtype=np.float32)
    pcm = (wav * 32767).astype(np.int16).tobytes()
    return pcm, sample_rate

def generate_with_local_llm(context_text, emotion_tag, llm_name=DEFAULT_LLM, max_tokens=200):
    tokenizer, model, device = load_llm(llm_name)
    if tokenizer is None or model is None:
        return f"I sense you might be feeling {emotion_tag}. I'm here to listen — tell me more.", 0
    prompt = (f"You are an empathetic assistant. The user appears to be '{emotion_tag}'. "
              f"Respond in a calm, validating, non-judgmental tone, concise. Context: {context_text}\n\nAssistant:")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    try:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception:
        pass
    import time
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    latency_ms = int((time.perf_counter() - start) * 1000)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text, latency_ms

# ---------------- WebRTC Offer Handler ----------------
pcs = set()

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    context_text = params.get("text", "")
    client_id = params.get("client_id", str(time.time()))
    pc = RTCPeerConnection()
    pcs.add(pc)

    data_channel = {"dc": None}
    audio_buffer = bytearray()
    buffer_lock = asyncio.Lock()
    convo_context = []  # short local conversation history

    @pc.on("datachannel")
    def on_datachannel(ch):
        print("[server] datachannel opened:", ch.label)
        data_channel["dc"] = ch

        @ch.on("message")
        def on_message(message):
            try:
                msg = json.loads(message)
            except Exception:
                return
            # client can request LLM reply via datachannel message
            if msg.get("type") == "request_reply":
                user_text = msg.get("text","")
                emotion = msg.get("emotion","unknown")
                reply_text, _ = generate_with_local_llm(user_text, emotion)
                # send LLM text back
                ch.send(json.dumps({"type":"llm_reply","text":reply_text}))
                # synthesize and enqueue to TTS track
                if tts is not None:
                    pcm, sr = synthesize_tts_pcm_bytes(reply_text)
                    if pcm is not None:
                        tts_track.enqueue(pcm, sr)

    @pc.on("track")
    def on_track(track):
        print("[server] Track received:", track.kind)
        if track.kind == "audio":
            async def consume_audio():
                nonlocal audio_buffer
                try:
                    while True:
                        frame = await track.recv()
                        arr = frame.to_ndarray()
                        if arr.ndim == 2:
                            samples = arr.mean(axis=0)
                        else:
                            samples = arr
                        if samples.dtype == np.float32 or samples.dtype == np.float64:
                            pcm = (samples * 32767).astype(np.int16).tobytes()
                        else:
                            pcm = samples.tobytes()
                        async with buffer_lock:
                            audio_buffer.extend(pcm)
                except Exception as e:
                    print("[server] audio consumer stopped:", e)
            asyncio.create_task(consume_audio())

        elif track.kind == "video":
            async def consume_video():
                try:
                    while True:
                        frame = await track.recv()
                        await asyncio.sleep(0.01)
                except Exception as e:
                    print("[server] video consumer stopped:", e)
            asyncio.create_task(consume_video())

    async def ser_loop():
        nonlocal audio_buffer, convo_context
        try:
            while True:
                await asyncio.sleep(SER_LOOP_INTERVAL)
                async with buffer_lock:
                    buf_len = len(audio_buffer)
                    if buf_len < SER_MIN_BYTES:
                        # send telemetry about buffer size
                        dc = data_channel.get("dc")
                        if dc and dc.readyState == "open":
                            try:
                                dc.send(json.dumps({"type":"telemetry","audio_buffer_bytes":buf_len}))
                            except Exception:
                                pass
                        continue
                    buf = bytes(audio_buffer)
                    audio_buffer = bytearray()
                # try common sample rates and pick the first that yields a valid result
                label, conf = ("error", 0.0)
                for sr in (48000,44100,16000,22050):
                    try:
                        wav_bytes = pcm16_to_wav_bytes(buf, sr)
                        label, conf = infer_ser_from_wav_bytes(wav_bytes)
                        break
                    except Exception:
                        label, conf = ("error", 0.0)
                print(f"[server] SER: {label} ({conf:.2f}) -- buf_bytes={len(buf)}")
                # send results & telemetry via datachannel
                dc = data_channel.get("dc")
                if dc and dc.readyState == "open":
                    try:
                        dc.send(json.dumps({"type":"ser_result","label":label,"confidence":conf,"audio_buffer_bytes":len(buf)}))
                    except Exception:
                        pass
                # add to conversation context (short memory)
                convo_context.append({"role":"user","text":f"[emotion:{label}] (voice)"})
                if len(convo_context) > 8:
                    convo_context = convo_context[-8:]
                # generate an LLM reply automatically if confidence is above threshold
                if conf >= 0.6 and tokenizer_llm is not None:
                    context_text = " ".join([c["text"] for c in convo_context])
                    reply_text, latency = generate_with_local_llm(context_text, label)
                    # send LLM reply via datachannel and enqueue TTS playback
                    if dc and dc.readyState == "open":
                        try:
                            dc.send(json.dumps({"type":"llm_reply","text":reply_text,"latency_ms":latency}))
                        except Exception:
                            pass
                    if tts is not None:
                        pcm, sr = synthesize_tts_pcm_bytes(reply_text)
                        if pcm is not None:
                            # chunk and enqueue to TTS track
                            chunk_size = int(0.02 * sr) * 2
                            for i in range(0, len(pcm), chunk_size):
                                tts_track.enqueue(pcm[i:i+chunk_size], sr)
                # also send telemetry about buffer size
                if dc and dc.readyState == "open":
                    try:
                        dc.send(json.dumps({"type":"telemetry","audio_buffer_bytes":0}))
                    except Exception:
                        pass
        except asyncio.CancelledError:
            pass

    def pcm16_to_wav_bytes(pcm_bytes, sample_rate):
        import io, soundfile as sf, numpy as _np
        arr = _np.frombuffer(pcm_bytes, dtype=_np.int16).astype(_np.int16)
        buf = io.BytesIO()
        sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    class TTSAudioTrack(MediaStreamTrack):
        kind = "audio"
        def __init__(self):
            super().__init__()
            self._q = asyncio.Queue()
        def enqueue(self, pcm_bytes, sr=22050):
            self._q.put_nowait((pcm_bytes, sr))
        async def recv(self):
            pcm_bytes, sr = await self._q.get()
            import numpy as _np
            arr = _np.frombuffer(pcm_bytes, dtype=_np.int16)
            frame = AudioFrame.from_ndarray(arr, layout="mono")
            frame.sample_rate = sr
            await asyncio.sleep(0.02)
            return frame

    tts_track = TTSAudioTrack()
    pc.addTrack(tts_track)

    ser_task = asyncio.create_task(ser_loop())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

@app.on_event("shutdown")
async def on_shutdown():
    for pc in list(pcs):
        await pc.close()
