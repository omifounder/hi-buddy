# Full server: WebRTC signaling stub, REST endpoints, and model hook usage.
import asyncio, tempfile, os, json
from fastapi import FastAPI, UploadFile, File, Request, WebSocket
from fastapi.responses import JSONResponse
from models_loader import init_all, stt_transcribe, llm_generate
app = FastAPI()
models = None
@app.on_event('startup')
async def startup():
    global models
    models = init_all()
@app.get('/ping')
def ping(): return {'ok': True}
@app.post('/infer_wav')
async def infer_wav(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(await file.read()); tmp.close()
    # simplistic: call stt_transcribe if available
    try:
        import soundfile as sf
        wav, sr = sf.read(tmp.name)
        text = stt_transcribe(models, wav, sampling_rate=sr)
    except Exception as e:
        text = ''
    os.unlink(tmp.name)
    return JSONResponse({'transcript': text})
@app.post('/generate_reply')
async def gen(req: Request):
    body = await req.json()
    text = body.get('text','')
    prompt = f'User: {text}'
    reply = llm_generate(models, prompt)
    return JSONResponse({'reply': reply})
