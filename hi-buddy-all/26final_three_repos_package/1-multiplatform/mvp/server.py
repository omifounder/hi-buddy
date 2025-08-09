from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
import tempfile, os, time
from src.shared.models_loader import load_ser, load_tts
analyze_ser = load_ser()
tts = load_tts()
app = FastAPI()

@app.get('/ping')
def ping(): return {'ok': True}

@app.post('/infer_wav')
async def infer_wav(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(await file.read()); tmp.flush(); tmp.close()
    try:
        start = time.time()
        res = analyze_ser(tmp.name)
        dur = (time.time()-start)*1000.0
        return JSONResponse({'result': res, 'dur_ms': dur})
    finally:
        try: os.remove(tmp.name)
        except: pass

@app.post('/synthesize_tts')
async def synth(req: Request):
    body = await req.json()
    text = body.get('text','Hello')
    path = tts(text)
    return JSONResponse({'wav_path': path})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=True)
