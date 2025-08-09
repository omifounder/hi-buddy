#!/usr/bin/env python3
# server.py - example assistant server instrumented for Prometheus/Grafana (demo-ready)
import os, time, tempfile, asyncio, json
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

from metrics_logger import log_metric, prometheus_metrics

# Try to import models_loader if present; otherwise use stubs
try:
    import models_loader
    analyze_ser = models_loader.load_ser()
    analyze_fer = models_loader.load_fer()
    try:
        llm = models_loader.load_llm()
    except Exception:
        llm = None
except Exception:
    analyze_ser = lambda wav: {'label':'neutral','confidence':0.9}
    analyze_fer = lambda img: {'label':'neutral','confidence':0.7}
    llm = None

app = FastAPI()

@app.get('/ping')
def ping():
    log_metric('health_ping', 1)
    return {'ok': True}

@app.post('/infer_wav')
async def infer_wav(file: Request):
    start = time.time()
    data = await file.body()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(data); tmp.flush(); tmp.close()
    wav_path = tmp.name
    try:
        res = await asyncio.get_event_loop().run_in_executor(None, analyze_ser, wav_path)
        dur_ms = (time.time() - start) * 1000.0
        log_metric('ser_infer_time_ms', dur_ms)
        try:
            log_metric('ser_confidence', float(res.get('confidence',0.0)), tags={'label': res.get('label','')})
        except Exception:
            pass
        return JSONResponse({'result': res})
    finally:
        try: os.remove(wav_path)
        except: pass

# Minimal /offer placeholder for WebRTC (for demo)
@app.post('/offer')
async def offer(req: Request):
    data = await req.json()
    log_metric('webrtc_offer_received', 1)
    # echo-back for demo; real aiortc server should respond with SDP answer
    return JSONResponse({'sdp': data.get('sdp'), 'type': 'answer'})

# Metrics endpoint for Prometheus scraping
@app.get('/metrics')
def metrics_endpoint():
    return Response(prometheus_metrics(), media_type='text/plain; version=0.0.4; charset=utf-8')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT',8000)), log_level='info')
