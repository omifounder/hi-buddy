\
# server_c.py - Audio + Video fusion -> LLM reply -> chunked TTS over WebRTC
import os, io, asyncio, time, json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import torch
import cv2

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, pipeline

# Optional FER lib
try:
    from fer import FER
except Exception:
    FER = None

# Optional Coqui TTS
try:
    from TTS.api import TTS
except Exception:
    TTS = None

app = FastAPI()

SER_MODEL = os.getenv("SER_MODEL", "j-hartmann/emotion-english-speech-distilroberta-base")
LLM_LOCAL = os.getenv("LLM_LOCAL", "distilgpt2")
SER_INTERVAL = float(os.getenv("SER_INTERVAL", "1.2"))
SER_MIN_BYTES = int(os.getenv("SER_MIN_BYTES", str(16000*2)))
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

# load SER
print("[server_c] loading SER:", SER_MODEL)
try:
    ser_processor = Wav2Vec2Processor.from_pretrained(SER_MODEL)
    ser_model = Wav2Vec2ForSequenceClassification.from_pretrained(SER_MODEL)
    ser_device = "cuda" if torch.cuda.is_available() else "cpu"
    ser_model.to(ser_device)
except Exception as e:
    print("[server_c] SER load failed:", e)
    ser_processor = None; ser_model = None; ser_device = "cpu"

# load FER if available
fer_detector = None
if FER is not None:
    try:
        fer_detector = FER(mtcnn=True)
        print("[server_c] FER loaded")
    except Exception as e:
        print("[server_c] FER init failed:", e); fer_detector = None

# local small LLM
text_gen = None
try:
    text_gen = pipeline("text-generation", model=LLM_LOCAL, device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print("[server_c] local LLM load failed:", e); text_gen = None

# TTS init
tts = None
if TTS is not None:
    try:
        tts = TTS(model_name=TTS_MODEL, progress_bar=False, gpu=False)
    except Exception as e:
        print("[server_c] TTS load failed:", e); tts = None

def infer_ser_from_wav_bytes(wav_bytes):
    if ser_model is None: return ("unknown",0.0)
    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        if audio.ndim>1: audio = audio.mean(axis=1)
        target_sr = getattr(ser_processor.feature_extractor, "sampling_rate", 16000)
        if sr!=target_sr:
            import librosa
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        inputs = ser_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(ser_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None: attention_mask = attention_mask.to(ser_device)
        with torch.no_grad():
            logits = ser_model(input_values, attention_mask=attention_mask).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(probs.argmax()); label = ser_model.config.id2label[idx] if hasattr(ser_model.config,"id2label") else str(idx)
        conf = float(probs[idx]); return (label, conf)
    except Exception as e:
        print("[server_c] SER inference error:", e); return ("error",0.0)

def infer_video_emotion(frame_bgr):
    # frame_bgr: numpy array BGR
    if fer_detector is not None:
        try:
            res = fer_detector.detect_emotions(frame_bgr)
            if res:
                top = res[0]
                emotions = top.get('emotions', {})
                if emotions:
                    label = max(emotions.items(), key=lambda x:x[1])[0]
                    conf = emotions[label]
                    return (label, conf)
        except Exception as e:
            print("[server_c] FER error:", e)
    # fallback: very simple heuristic using face ROI variance
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face.detectMultiScale(gray, 1.3, 5)
        if len(faces)>0:
            x,y,w,h = faces[0]
            roi = gray[y:y+h, x:x+w]
            var = float(roi.var())/255.0
            if var>0.02: return ("happy", 0.6)
            else: return ("neutral", 0.6)
    except Exception as e:
        print("[server_c] video heuristic failed:", e)
    return ("unknown", 0.0)

def generate_reply_text(context_str, emotion_label):
    if text_gen is None:
        return f"I think you're feeling {emotion_label}. Want to share more?"
    prompt = f"You are an empathetic assistant. The user seems {emotion_label}. Context: {context_str}. Reply empathetically."
    out = text_gen(prompt, max_length=120, do_sample=True, temperature=0.7, num_return_sequences=1)
    txt = out[0].get('generated_text','')
    if txt.startswith(prompt): txt = txt[len(prompt):].strip()
    return txt

def synthesize_tts_pcm_bytes(text, sample_rate=22050):
    if tts is None: return None, sample_rate
    wav = tts.tts(text)
    wav = np.array(wav, dtype=np.float32)
    pcm = (wav * 32767).astype(np.int16).tobytes()
    return pcm, sample_rate

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
    latest_frame = None
    frame_lock = asyncio.Lock()
    convo = []

    @pc.on("datachannel")
    def on_datachannel(ch):
        print("[server_c] datachannel opened:", ch.label)
        data_channel["dc"] = ch

    @pc.on("track")
    def on_track(track):
        print("[server_c] track:", track.kind)
        if track.kind == "audio":
            async def consume_audio():
                nonlocal audio_buffer
                try:
                    while True:
                        frame = await track.recv()
                        arr = frame.to_ndarray()
                        if arr.ndim==2: samples = arr.mean(axis=0)
                        else: samples = arr
                        if samples.dtype==np.float32 or samples.dtype==np.float64:
                            pcm = (samples * 32767).astype(np.int16).tobytes()
                        else:
                            pcm = samples.tobytes()
                        async with buffer_lock:
                            audio_buffer.extend(pcm)
                except Exception as e:
                    print("[server_c] audio consumer ended:", e)
            asyncio.create_task(consume_audio())

        elif track.kind == "video":
            async def consume_video():
                nonlocal latest_frame
                try:
                    while True:
                        frame = await track.recv()
                        img = frame.to_ndarray(format='bgr24')
                        async with frame_lock:
                            latest_frame = img.copy()
                except Exception as e:
                    print("[server_c] video consumer ended:", e)
            asyncio.create_task(consume_video())

    class TTSAudioTrack(MediaStreamTrack):
        kind="audio"
        def __init__(self):
            super().__init__(); self.q = asyncio.Queue()
        def enqueue(self, pcm_bytes, sr=22050):
            self.q.put_nowait((pcm_bytes, sr))
        async def recv(self):
            pcm_bytes, sr = await self.q.get()
            import numpy as _np
            arr = _np.frombuffer(pcm_bytes, dtype=_np.int16)
            frame = AudioFrame.from_ndarray(arr, layout="mono")
            frame.sample_rate = sr
            await asyncio.sleep(0.02)
            return frame

    tts_track = TTSAudioTrack()
    pc.addTrack(tts_track)

    async def fusion_loop():
        nonlocal audio_buffer, latest_frame, convo
        try:
            while True:
                await asyncio.sleep(SER_INTERVAL)
                async with buffer_lock:
                    b_len = len(audio_buffer)
                    if b_len < SER_MIN_BYTES:
                        dc = data_channel.get("dc")
                        if dc and dc.readyState=='open':
                            dc.send(json.dumps({"type":"telemetry","audio_buffer_bytes":b_len}))
                        continue
                    buf = bytes(audio_buffer); audio_buffer = bytearray()
                # infer speech emotion
                s_label, s_conf = ("error", 0.0)
                for sr in (48000,44100,22050,16000):
                    try:
                        import soundfile as sf2, io as _io, numpy as _np
                        arr = _np.frombuffer(buf, dtype=_np.int16)
                        b = _io.BytesIO(); sf2.write(b, arr, sr, format="WAV", subtype="PCM_16")
                        wav_bytes = b.getvalue()
                        s_label, s_conf = infer_ser_from_wav_bytes(wav_bytes)
                        break
                    except Exception:
                        s_label, s_conf = ("error", 0.0)
                v_label, v_conf = ("unknown", 0.0)
                async with frame_lock:
                    if latest_frame is not None:
                        v_label, v_conf = infer_video_emotion(latest_frame)
                fused = s_label; fused_conf = s_conf
                if v_conf > (s_conf + 0.05):
                    fused = v_label; fused_conf = v_conf
                print(f"[server_c] fused -> {fused} (s:{s_label}:{s_conf:.2f} v:{v_label}:{v_conf:.2f})")
                dc = data_channel.get("dc")
                if dc and dc.readyState=='open':
                    try:
                        dc.send(json.dumps({"type":"ser_result","speech":s_label,"speech_conf":s_conf,"video":v_label,"video_conf":v_conf,"fused":fused,"fused_conf":fused_conf,"audio_buffer_bytes":len(buf)}))
                    except Exception: pass
                if fused_conf >= 0.6:
                    convo.append({"role":"user","text":f"[emotion:{fused}]"})
                    if len(convo)>8: convo = convo[-8:]
                    ctx = " ".join([c["text"] for c in convo])
                    reply = generate_reply_text(ctx, fused)
                    if dc and dc.readyState=='open':
                        try: dc.send(json.dumps({"type":"llm_reply","text":reply}))
                        except Exception: pass
                    if tts is not None:
                        pcm, sr = synthesize_tts_pcm_bytes(reply)
                        if pcm is not None:
                            frame_bytes = int(0.02 * sr) * 2
                            for i in range(0, len(pcm), frame_bytes):
                                tts_track.enqueue(pcm[i:i+frame_bytes], sr)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(fusion_loop())
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

@app.on_event("shutdown")
async def on_shutdown():
    for pc in list(pcs):
        await pc.close()
