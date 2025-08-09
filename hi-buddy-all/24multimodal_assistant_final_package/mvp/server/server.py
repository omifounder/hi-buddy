#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
import uvicorn, tempfile, os, time
from models_loader import analyze_ser, synthesize_tts_to_wav

app = FastAPI()

@app.get('/ping')
def ping():
    return {'ok': True}

@app.post('/infer_wav')
async def infer_wav(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(await file.read()); tmp.flush(); tmp.close()
    try:
        start = time.time()
        res = analyze_ser(tmp.name)
        dur = (time.time() - start) * 1000.0
        return JSONResponse({'result': res, 'dur_ms': dur})
    finally:
        try: os.remove(tmp.name)
        except: pass

@app.post('/generate_tts')
async def generate_tts(req: Request):
    body = await req.json()
    text = body.get('text','Hello from MVP')
    wav = synthesize_tts_to_wav(text)
    return JSONResponse({'wav_path': os.path.basename(wav)})

@app.get('/tts/{name}')
def get_tts(name: str):
    p = os.path.join('/tmp', name)
    if os.path.exists(p):
        return FileResponse(p, media_type='audio/wav')
    return JSONResponse({'error':'not found'}, status_code=404)

if __name__ == '__main__':
    uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=True)
