#!/usr/bin/env python3
# assistant_server.py - lightweight assistant endpoint for demo metrics
import os, time, tempfile, asyncio, json
from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from metrics_logger import log_metric, prometheus_metrics

# minimal models_loader stub (simulate confidences)
def analyze_ser_stub(wav_path):
    import random, time
    time.sleep(0.05 + random.random()*0.1)
    lab = random.choice(['happy','sad','neutral','angry'])
    conf = round(0.5 + random.random()*0.5, 3)
    return {'label': lab, 'confidence': conf}

app = FastAPI()

@app.get('/ping')
def ping():
    log_metric('health_ping', 1)
    return {'ok': True}

@app.post('/infer_wav')
async def infer_wav(file: UploadFile = File(...)):
    start = time.time()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(await file.read()); tmp.flush(); tmp.close()
    wav_path = tmp.name
    try:
        res = await asyncio.get_event_loop().run_in_executor(None, analyze_ser_stub, wav_path)
        dur_ms = (time.time() - start) * 1000.0
        log_metric('ser_infer_time_ms', dur_ms)
        log_metric('ser_confidence', float(res.get('confidence',0.0)), tags={'label':res.get('label','')})
        return JSONResponse({'result': res})
    finally:
        try: os.remove(wav_path)
        except: pass

@app.get('/metrics')
def metrics_endpoint():
    return Response(prometheus_metrics(), media_type='text/plain; version=0.0.4; charset=utf-8')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT',8000)), log_level='info')
