#!/usr/bin/env python3
"""server.py - Dual-mode FastAPI server for Hello-World and Full mode.

- HELLO_WORLD_MODE=true -> runs lightweight mock models (no heavy downloads)
- HELLO_WORLD_MODE=false -> attempts to load real models via models_loader.py (if present)

Endpoints:
- POST /infer_wav : accept wav upload and run SER (mock or real)
- POST /offer : basic aiortc offer handler stub (if aiortc installed)
- GET /ping : health check
"""
import os, tempfile, asyncio
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
import uvicorn
import base64

HELLO_WORLD_MODE = os.environ.get('HELLO_WORLD_MODE', 'true').lower() in ('1','true','yes')

app = FastAPI()

if HELLO_WORLD_MODE:
    print('🚀 HELLO_WORLD_MODE enabled — using lightweight mocks.')
    # lightweight text generation using small HF model pipeline (if transformers installed)
    try:
        from transformers import pipeline
        _llm = pipeline('text-generation', model='distilgpt2')
        def generate_text(prompt):
            r = _llm(prompt, max_length=64, do_sample=False)
            return r[0]['generated_text']
    except Exception:
        def generate_text(prompt):
            return "Hello-world response (mock): " + prompt

    def analyze_ser_mock(wav_path):
        # simple energy-based mock SER
        import soundfile as sf, numpy as np
        try:
            data, sr = sf.read(wav_path, dtype='float32')
            if data.ndim > 1: data = data.mean(axis=1)
            energy = float(np.mean(np.abs(data)))
            if energy < 0.01:
                return {'label':'neutral','confidence':0.9}
            if energy < 0.05:
                return {'label':'sad','confidence':0.75}
            return {'label':'happy','confidence':0.8}
        except Exception as e:
            return {'label':'error','confidence':0.0,'error':str(e)}

    def analyze_fer_mock(image_bytes):
        # very small random-ish mock
        return {'label':'happy','confidence':0.9}

else:
    print('🧠 FULL MODE enabled — attempting to load real models from models_loader.py.')
    try:
        import models_loader
        llm = models_loader.load_llm()
        analyze_ser = models_loader.load_ser()
        analyze_fer = models_loader.load_fer()
        def generate_text(prompt):
            return llm.generate(prompt)
    except Exception as e:
        print('Failed to load models_loader:', e)
        # fallback to mocks
        HELLO_WORLD_MODE = True
        def generate_text(prompt):
            return "Fallback response: " + prompt
        def analyze_ser_mock(wav_path):
            return {'label':'neutral','confidence':0.6}
        def analyze_fer_mock(image_bytes):
            return {'label':'neutral','confidence':0.6}

@app.post('/infer_wav')
async def infer_wav(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(await file.read()); tmp.flush(); tmp.close()
    wav_path = tmp.name
    try:
        if HELLO_WORLD_MODE:
            res = analyze_ser_mock(wav_path)
            strategy = 'hello_mock'
        else:
            res = await asyncio.get_event_loop().run_in_executor(None, analyze_ser, wav_path)
            strategy = 'local_model'
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass
    return JSONResponse({'strategy': strategy, 'result': res})

@app.post('/offer')
async def offer(req: Request):
    # Minimal placeholder for WebRTC offer handling. In full deployment, replace with aiortc implementation.
    data = await req.json()
    # echo back a simple answer for demo purposes
    return JSONResponse({'sdp': data.get('sdp',''), 'type': 'answer', 'note': 'This is a demo echo answer; replace with aiortc server implementation.'})

@app.get('/ping')
def ping():
    return {'ok': True, 'hello_world': HELLO_WORLD_MODE}

if __name__ == '__main__':
    print('Starting server. HELLO_WORLD_MODE=', HELLO_WORLD_MODE)
    uvicorn.run('server:app', host='0.0.0.0', port=8000, log_level='info')
