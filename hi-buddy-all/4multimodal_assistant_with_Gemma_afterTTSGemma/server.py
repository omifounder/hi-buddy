# server.py (updated with Gemma orchestration)
import os, io, time, base64, json
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import soundfile as sf
import cv2
import torch

from models_loader import load_llm, load_text_emotion_model, load_video_emotion_model
from fusion import fuse_emotions
from live_logger import log_interaction
from safety import check_crisis, simple_toxicity_filter

# TTS: Coqui TTS
from TTS.api import TTS
from av import AudioFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

app = FastAPI()

# load perception models (text/video) - placeholders
text_tokenizer, text_model = load_text_emotion_model('distilroberta')
video_model = load_video_emotion_model()

# preload Gemma LLM (attempt)
llm_name_default = 'gemma'
tokenizer_llm, model_llm, llm_device = load_llm(llm_name_default)

# TTS model (may be large)
try:
    tts = TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC', progress_bar=False, gpu=False)
except Exception as e:
    print('TTS load error:', e)
    tts = None

@app.get('/')
async def index():
    with open('ui_advanced.html','r',encoding='utf-8') as f:
        return HTMLResponse(f.read())

def build_empathy_prompt(fused_emotion, user_text):
    prompt = (f"You are an empathetic assistant. The user appears to be '{fused_emotion}'. " 
              f"Respond in a calm, validating, non-judgmental tone, concise. User message: {user_text}\n\nAssistant:")
    return prompt

def generate_with_llm(fused_emotion, user_text, llm_name=llm_name_default, max_tokens=256, temperature=0.7):
    tokenizer, model, device = load_llm(llm_name)
    if model is None or tokenizer is None:
        return (f"I sense you might be feeling {fused_emotion}. I'm here to listen — can you tell me more?", 0)
    prompt = build_empathy_prompt(fused_emotion, user_text)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    try:
        device_str = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = {k:v.to(model.device) for k,v in inputs.items()}
    except Exception:
        pass
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
    latency_ms = int((time.perf_counter() - start)*1000)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip prompt echo if present
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text, latency_ms

# helper decoders
def decode_image_b64(b64str):
    if not b64str:
        return None
    data = base64.b64decode(b64str)
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

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

@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    llm_choice = llm_name_default
    while True:
        try:
            msg = await ws.receive_text()
            data = json.loads(msg)
        except Exception:
            break
        user_text = data.get('text','')
        audio_b64 = data.get('audio')
        frame_b64 = data.get('frame')
        llm_choice = data.get('llm_model', llm_choice)
        video_enabled = data.get('video_enabled', True)

        # text emotion (placeholder)
        text_emotion = None
        if user_text:
            try:
                inputs = text_tokenizer(user_text, return_tensors='pt')
                outputs = text_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()
                top_idx = int(np.argmax(probs))
                text_emotion = str(top_idx)
            except Exception as e:
                print('text emotion error', e)

        # speech emotion placeholder or integrated SER if available
        speech_emotion = None
        if audio_b64:
            # server may have SER integrated; for now keep tag
            speech_emotion = 'audio_received'

        # video emotion
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
                        roi = cv2.resize(roi,(48,48)).astype('float')/255.0
                        roi = roi.reshape(1,48,48,1)
                        preds = video_model.predict(roi, verbose=0)[0]
                        video_emotion = str(int(np.argmax(preds)))
                except Exception as e:
                    print('video emotion error', e)

        fused = fuse_emotions(text_emotion, speech_emotion, video_emotion)

        # generate LLM reply
        reply_text, latency_ms = generate_with_llm(fused, user_text, llm_name=llm_choice)

        # safety checks
        if not simple_toxicity_filter(reply_text):
            reply_text = "I'm sorry — I can't assist with that request."
        if check_crisis(user_text):
            reply_text = "I'm really sorry you're feeling this way. If you're in immediate danger, please contact emergency services or a crisis hotline. Would you like resources in your area?"

        # persist media
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

        # log and send
        log_interaction(user_text, text_emotion, speech_emotion, video_emotion, fused, llm_choice, reply_text, audio_file, video_file, inference_latency_ms=latency_ms, response_sentiment=None)
        await ws.send_text(json.dumps({'emotion': fused, 'reply': reply_text, 'latency_ms': latency_ms}))
    await ws.close()

# aiortc /offer for TTS streaming (kept simple)
pcs = set()
class TTSAudioTrack(MediaStreamTrack):
    kind = 'audio'
    def __init__(self, text, sample_rate=22050):
        super().__init__()
        self.text = text
        self.sample_rate = sample_rate
        self._gen = self._synthesize()

    def _synthesize(self):
        if tts is None:
            return iter([])
        wav = tts.tts(self.text)
        wav = np.array(wav,dtype=np.float32)
        pcm = (wav * 32767).astype('int16')
        chunk = int(0.02 * self.sample_rate)
        for i in range(0,len(pcm),chunk):
            yield pcm[i:i+chunk]
        return

    async def recv(self):
        chunk = next(self._gen, None)
        if chunk is None:
            await asyncio.sleep(0.02)
            raise asyncio.CancelledError()
        frame = AudioFrame.from_ndarray(chunk, layout='mono')
        frame.sample_rate = self.sample_rate
        await asyncio.sleep(0.02)
        return frame

@app.post('/offer')
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
    text = params.get('text','Hello from assistant')
    pc = RTCPeerConnection()
    pcs.add(pc)
    track = TTSAudioTrack(text)
    pc.addTrack(track)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return JSONResponse({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})

@app.on_event('shutdown')
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
