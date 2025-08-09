# server/app_orchestrator.py
import os, tempfile, base64, uuid, asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from strategies import DummyStrategy, LocalONNXStrategy, ForwardToCloudStrategy

app = FastAPI()

# CONFIG (via env vars)
MODE = os.environ.get("INFERENCE_ROUTER_MODE", "hybrid")  # local, cloud, hybrid
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "/models/ser/model.onnx")
CLOUD_ENDPOINT = os.environ.get("CLOUD_ENDPOINT", None)  # required for forward-to-cloud
CONFIDENCE_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.6"))

dummy = DummyStrategy()
local_onnx = None
if os.path.exists(ONNX_MODEL_PATH):
    def preprocess_stub(wav_path):
        import soundfile as sf, numpy as np
        data, sr = sf.read(wav_path, dtype='float32')
        if data.ndim > 1: data = data.mean(axis=1)
        # pad/crop to 16000 samples
        if len(data) < 16000:
            pad = 16000 - len(data)
            data = np.pad(data, (0,pad))
        else:
            data = data[:16000]
        return data.reshape(1, -1).astype('float32')
    try:
        local_onnx = LocalONNXStrategy(ONNX_MODEL_PATH, preprocess_stub)
    except Exception as e:
        print("Local ONNX init failed:", e)
        local_onnx = None

def forward_to_cloud_sync(wav_path):
    import requests
    if not CLOUD_ENDPOINT:
        return {"label":"cloud_unavailable","confidence":0.0}
    files = {"file": open(wav_path, "rb")}
    headers = {}
    r = requests.post(CLOUD_ENDPOINT, files=files, headers=headers, timeout=15)
    return r.json()

@app.post("/infer_wav")
async def infer_wav(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(await file.read())
    tmp.flush()
    tmp.close()
    wav_path = tmp.name

    strategy_used = None
    result = None

    if MODE == "local" and local_onnx:
        strategy_used = "local_onnx"
        result = await local_onnx.infer(wav_path)
        if result.get("confidence", 0) < CONFIDENCE_THRESHOLD and CLOUD_ENDPOINT:
            cloud_res = await asyncio.get_event_loop().run_in_executor(None, forward_to_cloud_sync, wav_path)
            result["cloud_fallback"] = cloud_res
            strategy_used = "local_onnx->cloud"
    elif MODE == "cloud":
        strategy_used = "cloud"
        result = await asyncio.get_event_loop().run_in_executor(None, forward_to_cloud_sync, wav_path)
    else:  # hybrid
        if local_onnx:
            strategy_used = "local_onnx"
            result = await local_onnx.infer(wav_path)
            if result.get("confidence", 0) < CONFIDENCE_THRESHOLD and CLOUD_ENDPOINT:
                strategy_used = "local_onnx->cloud"
                result_cloud = await asyncio.get_event_loop().run_in_executor(None, forward_to_cloud_sync, wav_path)
                result["cloud_fallback"] = result_cloud
        else:
            strategy_used = "dummy"
            result = await dummy.infer(wav_path)

    try:
        os.remove(wav_path)
    except Exception:
        pass

    return JSONResponse({"strategy": strategy_used, "result": result})

@app.get("/ping")
def ping():
    return {"ok": True, "mode": MODE, "onnx": bool(local_onnx)}

if __name__ == "__main__":
    uvicorn.run("app_orchestrator:app", host="0.0.0.0", port=8000, log_level="info")
