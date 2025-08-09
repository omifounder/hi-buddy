# server.py - WebRTC + FastAPI server for live SER + TTS streaming
import os, io, asyncio, time, json, base64
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import soundfile as sf
import torch

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from TTS.api import TTS

app = FastAPI()

# Load SER
SER_MODEL = os.getenv("SER_MODEL", "j-hartmann/emotion-english-speech-distilroberta-base")
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

# Load TTS (Coqui)
try:
    tts = TTS(model_name=os.getenv("TTS_MODEL","tts_models/en/ljspeech/tacotron2-DDC"), progress_bar=False, gpu=False)
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
        print("[server] datachannel opened:", ch.label)
        data_channel["dc"] = ch

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
        nonlocal audio_buffer
        try:
            while True:
                await asyncio.sleep(1.5)
                async with buffer_lock:
                    if len(audio_buffer) < 16000*2:
                        continue
                    buf = bytes(audio_buffer)
                    audio_buffer = bytearray()
                for sr in (48000,44100,16000,22050):
                    try:
                        wav_bytes = pcm16_to_wav_bytes(buf, sr)
                        label, conf = infer_ser_from_wav_bytes(wav_bytes)
                        break
                    except Exception:
                        label, conf = ("error", 0.0)
                print(f"[server] SER: {label} ({conf:.2f})")
                dc = data_channel.get("dc")
                if dc and dc.readyState == "open":
                    try:
                        dc.send(json.dumps({"type":"ser_result","label":label,"confidence":conf}))
                    except Exception as e:
                        print("[server] DC send failed:", e)
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

pcs = set()

@app.on_event("shutdown")
async def on_shutdown():
    for pc in pcs:
        await pc.close()
