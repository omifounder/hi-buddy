from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile, os, time
app = FastAPI()

@app.get('/ping')
def ping(): return {'ok': True}

@app.post('/infer_wav')
async def infer_wav(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(await file.read()); tmp.flush(); tmp.close()
    try:
        # naive stub: return neutral
        return JSONResponse({'result':{'label':'neutral','confidence':0.8}})
    finally:
        try: os.remove(tmp.name)
        except: pass

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=8000)
