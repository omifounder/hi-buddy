import os, io, time, base64, asyncio, json
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import soundfile as sf
import cv2
import torch

# aiortc for TTS streaming
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole

# local modules
from models_loader import load_text_emotion_model, load_video_emotion_model, load_llm
from fusion import fuse_emotions
from live_logger import log_interaction

# TTS (Coqui TTS)
from TTS.api import TTS
from av import AudioFrame

app = FastAPI()

# load basic models (placeholders; heavy models may fail without manual download)
text_tokenizer, text_model = load_text_emotion_model("distilroberta")
video_model = load_video_emotion_model()

# load a simple TTS model (Coqui). This may download first-use models.
try:
    tts = TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC', progress_bar=False, gpu=False)
except Exception as e:
    print('TTS model load error (you can still run without TTS):', e)
    tts = None

# serve UI
@app.get('/')
async def index():
    with open('ui_advanced.html', 'r', encoding='utf-8') as f:
        return HTMLResponse(f.read())

# helper: decode base64 image
def decode_image_b64(b64str):
    if not b64str:
        return None
    data = base64.b64decode(b64str)
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

# helper: decode base64 audio bytes to numpy array
def decode_audio_b64(b64str):
    if not b64str:
        return None, None
    data = base64.b64decode(b64str)
    buf = io.BytesIO(data)
    try:
        wav, sr = sf.read(buf)
        return wav, sr
    except Exception as e:
        print('audio decode error', e)
        return None, None

# WebSocket endpoint for perception and conversation
@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    llm_choice = 'gemma'
    while True:
        try:
            msg = await ws.receive_text()
            data = json.loads(msg)
        except Exception:
            break

        text = data.get('text','')
        audio_b64 = data.get('audio')
        frame_b64 = data.get('frame')
        llm_choice = data.get('llm_model', llm_choice)
        video_enabled = data.get('video_enabled', True)

        # text emotion (simple)
        text_emotion = None
        if text:
            try:
                inputs = text_tokenizer(text, return_tensors='pt')
                outputs = text_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()
                top_idx = int(np.argmax(probs))
                text_emotion = str(top_idx)
            except Exception as e:
                print('text emotion error', e)

        # speech emotion (placeholder: will be replaced by SER model)
        speech_emotion = None
        if audio_b64:
            # minimal: mark as 'audio_received'
            speech_emotion = 'audio_received'

        # video emotion using FER model
        video_emotion = None
        if video_enabled and frame_b64:
            frame = decode_image_b64(frame_b64)
            if frame is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray,1.3,5)
                    if len(faces)>0:
                        x,y,w,h = faces[0]
                        roi = gray[y:y+h,x:x+w]
                        import numpy as _np
                        roi = cv2.resize(roi,(48,48)).astype('float')/255.0
                        roi = roi.reshape(1,48,48,1)
                        preds = video_model.predict(roi, verbose=0)[0]
                        video_emotion = str(int(np.argmax(preds)))
                except Exception as e:
                    print('video emotion error', e)

        fused = fuse_emotions(text_emotion, speech_emotion, video_emotion)

        # LLM reply (simplified)
        tokenizer_llm, model_llm = load_llm(llm_choice)
        if model_llm is None:
            response_text = f"I sense you're feeling {fused}. How can I help?"
        else:
            try:
                prompt = f"[User emotion: {fused}] Respond empathetically: {text}"
                tok = tokenizer_llm(prompt, return_tensors='pt').to(model_llm.device)
                with torch.no_grad():
                    out = model_llm.generate(**tok, max_new_tokens=200)
                response_text = tokenizer_llm.decode(out[0], skip_special_tokens=True)
            except Exception as e:
                print('LLM generate error', e)
                response_text = f"I sense you're {fused}."

        # Save media files for logs if any
        audio_file = None; video_file = None
        if audio_b64:
            wav, sr = decode_audio_b64(audio_b64)
            if wav is not None:
                fname = f'logs/media/audio_{int(time.time()*1000)}.wav'
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                sf.write(fname, wav, sr)
                audio_file = fname
        if frame_b64:
            frame = decode_image_b64(frame_b64)
            if frame is not None:
                fname = f'logs/media/frame_{int(time.time()*1000)}.jpg'
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                cv2.imwrite(fname, frame)
                video_file = fname

        # log interaction (non-blocking)
        log_interaction(text, text_emotion, speech_emotion, video_emotion, fused, llm_choice, response_text, audio_file, video_file, inference_latency_ms=None, response_sentiment=None)

        # send reply (text) back to UI
        await ws.send_text(json.dumps({'emotion': fused, 'reply': response_text}))
    await ws.close()

# ---- aiortc offer route for TTS streaming ----
pcs = set()

class TTSAudioTrack(MediaStreamTrack):
    kind = 'audio'
    def __init__(self, text, sample_rate=22050):
        super().__init__()
        self.text = text
        self.sample_rate = sample_rate
        self._generator = self._synthesize_generator()

    def _synthesize_generator(self):
        if tts is None:
            return iter([])
        # generate waveform as numpy float32
        wav = tts.tts(self.text)
        wav = np.array(wav, dtype=np.float32)
        # Coqui TTS usually returns at 22050; convert to int16 PCM
        pcm = (wav * 32767).astype(np.int16)
        chunk = int(0.02 * self.sample_rate)
        for i in range(0, len(pcm), chunk):
            yield pcm[i:i+chunk]
        return

    async def recv(self):
        frame_bytes = next(self._generator, None)
        if frame_bytes is None:
            await asyncio.sleep(0.02)
            raise asyncio.CancelledError()
        # build an AudioFrame
        frame = AudioFrame.from_ndarray(frame_bytes, layout='mono')
        frame.sample_rate = self.sample_rate
        await asyncio.sleep(0.02)
        return frame

@app.post('/offer')
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
    text = params.get('text', 'Hello from the assistant.')
    pc = RTCPeerConnection()
    pcs.add(pc)

    track = TTSAudioTrack(text)
    pc.addTrack(track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return JSONResponse({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})

# cleanup on shutdown
@app.on_event('shutdown')
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
