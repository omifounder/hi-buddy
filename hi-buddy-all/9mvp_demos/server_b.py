\
# server_b.py - Audio SER + LLM reply + chunked TTS streaming via WebRTC (offer endpoint)
import os, io, asyncio, time, json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import torch

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, pipeline

# Optional: Coqui TTS (fast to synthesize). If not installed, server will still run but TTS disabled.
try:
    from TTS.api import TTS
except Exception:
    TTS = None

app = FastAPI()

# Configuration (env overrides allowed)
SER_MODEL = os.getenv("SER_MODEL", "j-hartmann/emotion-english-speech-distilroberta-base")
LLM_LOCAL = os.getenv("LLM_LOCAL", "distilgpt2")   # small demo model
SER_INTERVAL = float(os.getenv("SER_INTERVAL", "1.2"))
SER_MIN_BYTES = int(os.getenv("SER_MIN_BYTES", str(16000*2)))  # ~1s @16k int16
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

# Load SER model
print("[server_b] Loading SER model:", SER_MODEL)
ser_processor = None
ser_model = None
ser_device = "cpu"
try:
    ser_processor = Wav2Vec2Processor.from_pretrained(SER_MODEL)
    ser_model = Wav2Vec2ForSequenceClassification.from_pretrained(SER_MODEL)
    ser_device = "cuda" if torch.cuda.is_available() else "cpu"
    ser_model.to(ser_device)
    print(f"[server_b] SER loaded on {ser_device}")
except Exception as e:
    print("[server_b] SER load failed:", e)
    ser_processor = None
    ser_model = None

# Load small local LLM pipeline for demo replies
text_gen = None
try:
    print("[server_b] Loading local LLM:", LLM_LOCAL)
    text_gen = pipeline("text-generation", model=LLM_LOCAL, device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print("[server_b] Local LLM load failed (will use canned replies):", e)
    text_gen = None

# TTS loader (Coqui)
tts = None
if TTS is not None:
    try:
        print("[server_b] Loading TTS model:", TTS_MODEL)
        tts = TTS(model_name=TTS_MODEL, progress_bar=False, gpu=False)
    except Exception as e:
        print("[server_b] TTS load failed:", e)
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
        print("[server_b] SER inference error:", e)
        return ("error", 0.0)

def generate_reply_text(user_text, emotion_label):
    if text_gen is None:
        # canned empathic reply
        return f"I hear you. You sound {emotion_label}. Would you like to tell me more?"
    prompt = f"You are an empathetic assistant. The user appears {emotion_label}. Reply briefly and empathically. User: {user_text}\\nAssistant:"
    out = text_gen(prompt, max_length=80, do_sample=True, temperature=0.7, num_return_sequences=1)
    txt = out[0].get("generated_text", "")
    if txt.startswith(prompt):
        txt = txt[len(prompt):].strip()
    return txt

# TTS synth -> PCM int16 bytes
def synthesize_tts_pcm_bytes(text, sample_rate=22050):
    if tts is None:
        return None, sample_rate
    wav = tts.tts(text)
    wav = np.array(wav, dtype=np.float32)
    pcm = (wav * 32767).astype(np.int16).tobytes()
    return pcm, sample_rate

# WebRTC offer handler (audio only in Option B)
pcs = set()

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    data_channel = {"dc": None}
    audio_buffer = bytearray()
    buffer_lock = asyncio.Lock()

    @pc.on("datachannel")
    def on_datachannel(ch):
        print("[server_b] datachannel opened:", ch.label)
        data_channel["dc"] = ch

    @pc.on("track")
    def on_track(track):
        print("[server_b] Track received:", track.kind)
        if track.kind == "audio":
            async def consume_audio():
                nonlocal audio_buffer
                try:
                    while True:
                        frame = await track.recv()
                        arr = frame.to_ndarray()
                        # handle stereo/mono
                        if arr.ndim == 2:
                            samples = arr.mean(axis=0)
                        else:
                            samples = arr
                        # convert floats to int16 bytes if necessary
                        if samples.dtype == np.float32 or samples.dtype == np.float64:
                            pcm = (samples * 32767).astype(np.int16).tobytes()
                        else:
                            pcm = samples.tobytes()
                        async with buffer_lock:
                            audio_buffer.extend(pcm)
                except Exception as e:
                    print("[server_b] audio consumer ended:", e)
            asyncio.create_task(consume_audio())

    # TTS audio track for chunked streaming
    class TTSAudioTrack(MediaStreamTrack):
        kind = "audio"
        def __init__(self):
            super().__init__()
            self.queue = asyncio.Queue()

        def enqueue(self, pcm_bytes, sample_rate=22050):
            self.queue.put_nowait((pcm_bytes, sample_rate))

        async def recv(self):
            pcm_bytes, sr = await self.queue.get()
            import numpy as _np
            arr = _np.frombuffer(pcm_bytes, dtype=_np.int16)
            frame = AudioFrame.from_ndarray(arr, layout="mono")
            frame.sample_rate = sr
            await asyncio.sleep(0.02)
            return frame

    tts_track = TTSAudioTrack()
    pc.addTrack(tts_track)

    async def ser_task():
        nonlocal audio_buffer
        try:
            while True:
                await asyncio.sleep(SER_INTERVAL)
                async with buffer_lock:
                    buf_len = len(audio_buffer)
                    if buf_len < SER_MIN_BYTES:
                        # send telemetry
                        dc = data_channel.get("dc")
                        if dc and dc.readyState == "open":
                            try: dc.send(json.dumps({"type":"telemetry","audio_buffer_bytes":buf_len}))
                            except Exception: pass
                        continue
                    buf = bytes(audio_buffer)
                    audio_buffer = bytearray()
                # try a few sample rates
                label, conf = ("error", 0.0)
                for sr in (48000,44100,22050,16000):
                    try:
                        import soundfile as sf2, io as _io, numpy as _np
                        arr = _np.frombuffer(buf, dtype=_np.int16)
                        b = _io.BytesIO()
                        sf2.write(b, arr, sr, format="WAV", subtype="PCM_16")
                        wav_bytes = b.getvalue()
                        label, conf = infer_ser_from_wav_bytes(wav_bytes)
                        break
                    except Exception:
                        label, conf = ("error", 0.0)
                print(f"[server_b] SER -> {label} ({conf:.2f})")
                dc = data_channel.get("dc")
                if dc and dc.readyState == "open":
                    try:
                        dc.send(json.dumps({"type":"ser_result","label":label,"confidence":conf,"audio_buffer_bytes":len(buf)}))
                    except Exception:
                        pass
                # when confident, generate LLM reply and stream TTS in small chunks
                if conf >= 0.6:
                    reply = generate_reply_text("", label)
                    # send text reply via datachannel
                    if dc and dc.readyState == "open":
                        try: dc.send(json.dumps({"type":"llm_reply","text":reply}))
                        except Exception: pass
                    # synthesize and enqueue chunked PCM to TTS track
                    if tts is not None:
                        pcm, sr = synthesize_tts_pcm_bytes(reply)
                        if pcm is not None:
                            # chunk size for 20ms frames (int16 bytes)
                            frame_bytes = int(0.02 * sr) * 2
                            for i in range(0, len(pcm), frame_bytes):
                                chunk = pcm[i:i+frame_bytes]
                                tts_track.enqueue(chunk, sr)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(ser_task())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

@app.on_event("shutdown")
async def on_shutdown():
    for pc in list(pcs):
        await pc.close()
