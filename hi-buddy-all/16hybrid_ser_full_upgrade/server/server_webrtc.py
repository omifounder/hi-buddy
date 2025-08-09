# server_webrtc.py - aiortc DataChannel server that accepts binary audio blobs (raw WAV bytes) over DC
import os, tempfile, base64, uuid, asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import BYE
from aiortc.contrib.media import MediaBlackhole
import json, soundfile as sf
from strategies import DummyStrategy, LocalOnnxStrategy

app = FastAPI()
pcs = set()

# Instantiate strategies as in app_orchestrator
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
        if len(data) < 16000:
            data = np.pad(data, (0,16000 - len(data)))
        else:
            data = data[:16000]
        return data.reshape(1,-1).astype('float32')
    try:
        local_onnx = LocalOnnxStrategy(ONNX_MODEL_PATH, preprocess_stub)
    except Exception as e:
        print('Local ONNX init failed:', e)
        local_onnx = None

@app.post('/offer')
async def offer(request: Request):
    params = await request.json()
    offer_sdp = params['sdp']
    offer_type = params.get('type','offer')
    pc = RTCPeerConnection()
    pcs.add(pc)
    dc = None

    @pc.on('datachannel')
    def on_datachannel(channel):
        nonlocal dc
        dc = channel
        print('DC created', channel.label)
        @channel.on('message')
        def on_message(message):
            # message may be binary or JSON
            try:
                if isinstance(message, (bytes, bytearray)):
                    # expect raw WAV bytes - write to temp file and run inference
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    tmp.write(message); tmp.flush(); tmp.close()
                    wav_path = tmp.name
                    # choose strategy
                    async def run_and_respond():
                        if MODE in ('local','hybrid') and local_onnx:
                            res = await local_onnx.infer(wav_path)
                            # fallback to cloud logic omitted for brevity
                        else:
                            res = await dummy.infer(wav_path)
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        channel.send(json.dumps({'type':'ser_result','result':res}))
                    asyncio.ensure_future(run_and_respond())
                else:
                    # assume JSON control
                    obj = json.loads(message)
                    print('DC JSON msg', obj)
            except Exception as e:
                print('DC message handling error', e)

    @pc.on('track')
    def on_track(track):
        print('Track received', track.kind)
        # optionally record RTP track

    offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return JSONResponse({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})

@app.on_event('shutdown')
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
