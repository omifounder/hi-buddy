#!/usr/bin/env python3
import os, time, tempfile, asyncio, json
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from models_loader import load_ser, load_fer, load_llm, load_tts
from metrics_logger import log_metric

app = FastAPI()
analyze_ser = load_ser()
analyze_fer = load_fer()
llm = load_llm()
tts = load_tts()

@app.get('/ping')
def ping():
    log_metric('health_ping', 1)
    return {'ok': True}

@app.post('/infer_wav')
async def infer_wav(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(await file.read()); tmp.flush(); tmp.close()
    try:
        start = time.time()
        res = await asyncio.get_event_loop().run_in_executor(None, analyze_ser, tmp.name)
        dur = (time.time() - start) * 1000.0
        log_metric('ser_infer_time_ms', dur)
        log_metric('ser_confidence', float(res.get('confidence',0.0)), tags={'label': res.get('label','')})
        return JSONResponse({'result': res, 'dur_ms': dur})
    finally:
        try: os.remove(tmp.name)
        except: pass

@app.post('/generate_reply')
async def generate_reply(req: Request):
    body = await req.json()
    text = body.get('text','')
    emotion = body.get('emotion',{})
    prompt = f"Emotion:{emotion}. User: {text}\nAssistant:"
    reply = llm.generate(prompt) if llm else 'stub reply: ' + text[:120]
    log_metric('llm_reply_generated', 1)
    return JSONResponse({'reply': reply})

@app.post('/synthesize_tts')
async def synthesize_tts(req: Request):
    body = await req.json()
    text = body.get('text','')
    path = tts(text)
    return JSONResponse({'wav_path': path})

@app.get('/metrics')
def metrics():
    return JSONResponse({'note':'metrics endpoint - use Prometheus in production'})

if __name__ == '__main__':
    uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=True)
