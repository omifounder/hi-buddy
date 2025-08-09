\
#!/usr/bin/env python3
# server_webrtc.py - FastAPI + aiortc minimal server for MVP demo
# Receives WebRTC offer at /offer, creates a PeerConnection, listens on a DataChannel for binary WAV chunks,
# runs mock SER/FER (or real models if wired), and sends JSON results back over the DataChannel.
import os, asyncio, tempfile, json, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaPlayer, MediaStreamTrack
    from aiortc.contrib.media import MediaRelay
except Exception:
    raise RuntimeError("aiortc is required. Install with: pip install aiortc av")

app = FastAPI()
pcs = set()

# Simple mock analyzers (replace with models_loader functions for real models)
def analyze_ser_mock(wav_path):
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

def analyze_fer_mock(image_bytes=None):
    return {'label':'happy','confidence':0.9}

relay = MediaRelay()

@app.post('/offer')
async def offer(req: Request):
    params = await req.json()
    offer_sdp = params.get('sdp')
    offer_type = params.get('type', 'offer')
    if not offer_sdp:
        return JSONResponse({'error': 'sdp missing'}, status_code=400)

    pc = RTCPeerConnection()
    pcs.add(pc)
    data_channel = None

    @pc.on('datachannel')
    def on_datachannel(channel):
        nonlocal data_channel
        data_channel = channel
        print('[server] DataChannel created:', channel.label)

        @channel.on('message')
        def on_message(message):
            # If we get bytes, expect WAV bytes
            try:
                if isinstance(message, (bytes, bytearray)):
                    # write wav bytes to temp file and run SER
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    tmp.write(message); tmp.flush(); tmp.close()
                    wav_path = tmp.name
                    # run mock or real SER
                    res = analyze_ser_mock(wav_path)
                    # send result back as JSON
                    try:
                        channel.send(json.dumps({'type':'SER','result':res}))
                    except Exception as e:
                        print('failed to send dc reply', e)
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
                else:
                    # JSON control messages
                    try:
                        obj = json.loads(message)
                        print('[server] DC json msg:', obj)
                        # respond with mock FER for demo
                        if obj.get('cmd') == 'fer_request':
                            channel.send(json.dumps({'type':'FER','result': analyze_fer_mock()}))
                    except Exception as e:
                        print('dc message parse error', e)
            except Exception as e:
                print('dc handling error', e)

    @pc.on('track')
    def on_track(track):
        print('[server] Track received kind=', track.kind)
        # Optionally process tracks

    offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    print('[server] Returning answer')
    return JSONResponse({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})

@app.on_event('shutdown')
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)

if __name__ == '__main__':
    uvicorn.run('server_webrtc:app', host='0.0.0.0', port=8000, log_level='info')
