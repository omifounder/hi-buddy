from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
import tempfile, os, time, asyncio
from src.shared.models_loader import load_ser, load_fer, load_llm, load_tts
from src.shared.utils import ensure_wav_format
analyze_ser = load_ser()
analyze_fer = load_fer()
llm = load_llm()
tts = load_tts()
app = FastAPI()

@app.get('/ping')
def ping(): return {'ok': True}

@app.post('/infer_wav')
async def infer_wav(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(await file.read()); tmp.flush(); tmp.close()
    try:
        wav = ensure_wav_format(tmp.name)
        start = time.time()
        res = await asyncio.get_event_loop().run_in_executor(None, analyze_ser, wav)
        dur = (time.time()-start)*1000.0
        return JSONResponse({'result': res, 'dur_ms': dur})
    finally:
        try: os.remove(tmp.name)
        except: pass

@app.post('/generate_reply')
async def gen(req: Request):
    body = await req.json()
    text = body.get('text','')
    emotion = body.get('emotion',{})
    prompt = f"Emotion: {emotion}. User: {text}\nAssistant:"
    reply = llm.generate(prompt)
    return JSONResponse({'reply': reply})

@app.post('/synthesize_tts')
async def synth(req: Request):
    body = await req.json()
    text = body.get('text','')
    path = tts(text)
    return JSONResponse({'wav_path': path})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=True)
