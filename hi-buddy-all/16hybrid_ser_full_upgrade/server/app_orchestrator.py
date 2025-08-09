# app_orchestrator.py
import os, tempfile, asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from strategies import DummyStrategy, LocalOnnxStrategy, ForwardToCloudStrategy

app = FastAPI()

MODE = os.environ.get('INFERENCE_ROUTER_MODE','hybrid')
ONNX_MODEL_PATH = os.environ.get('ONNX_MODEL_PATH','/models/ser/model.onnx')
CLOUD_ENDPOINT = os.environ.get('CLOUD_ENDPOINT', None)
CONFIDENCE_THRESHOLD = float(os.environ.get('CONF_THRESHOLD','0.6'))

dummy = DummyStrategy()
local_onnx = None
if os.path.exists(ONNX_MODEL_PATH):
    def preprocess_stub(wav_path):
        import soundfile as sf, numpy as np
        data, sr = sf.read(wav_path, dtype='float32')
        if data.ndim > 1: data = data.mean(axis=1)
        # simple pad/crop to 16000
        if len(data) < 16000:
            data = np.pad(data, (0,16000 - len(data)))
        else:
            data = data[:16000]
        # model may expect shape (1,16000)
        return data.reshape(1,-1).astype('float32')
    try:
        local_onnx = LocalOnnxStrategy(ONNX_MODEL_PATH, preprocess_stub)
    except Exception as e:
        print('Failed to init LocalOnnxStrategy:', e)
        local_onnx = None

def forward_to_cloud_sync(wav_path):
    import requests
    if not CLOUD_ENDPOINT:
        return {'label':'cloud_unavailable','confidence':0.0}
    files = {'file': open(wav_path,'rb')}
    r = requests.post(CLOUD_ENDPOINT, files=files, timeout=15)
    return r.json()

@app.post('/infer_wav')
async def infer_wav(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(await file.read())
    tmp.flush(); tmp.close()
    wav_path = tmp.name

    strategy_used = None; result = None

    if MODE == 'local' and local_onnx:
        strategy_used = 'local_onnx'
        result = await local_onnx.infer(wav_path)
        if result.get('confidence',0) < CONFIDENCE_THRESHOLD and CLOUD_ENDPOINT:
            cloud_res = await asyncio.get_event_loop().run_in_executor(None, forward_to_cloud_sync, wav_path)
            result['cloud_fallback'] = cloud_res; strategy_used='local_onnx->cloud'
    elif MODE == 'cloud':
        strategy_used='cloud'
        result = await asyncio.get_event_loop().run_in_executor(None, forward_to_cloud_sync, wav_path)
    else: # hybrid
        if local_onnx:
            strategy_used='local_onnx'
            result = await local_onnx.infer(wav_path)
            if result.get('confidence',0) < CONFIDENCE_THRESHOLD and CLOUD_ENDPOINT:
                result_cloud = await asyncio.get_event_loop().run_in_executor(None, forward_to_cloud_sync, wav_path)
                result['cloud_fallback'] = result_cloud; strategy_used='local_onnx->cloud'
        else:
            strategy_used='dummy'; result = await dummy.infer(wav_path)

    try:
        os.remove(wav_path)
    except Exception:
        pass

    return JSONResponse({'strategy': strategy_used, 'result': result})

@app.get('/ping')
def ping():
    return {'ok': True, 'mode': MODE, 'onnx': bool(local_onnx)}

if __name__ == '__main__':
    uvicorn.run('app_orchestrator:app', host='0.0.0.0', port=8000, log_level='info')
