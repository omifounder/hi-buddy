#!/usr/bin/env python3
# server_webrtc_real.py
import os, tempfile, asyncio, json, uuid, subprocess
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaPlayer, MediaStreamTrack
    from aiortc.contrib.media import MediaRelay
except Exception as e:
    raise RuntimeError('aiortc and av are required: pip install aiortc av') from e

import models_loader

app = FastAPI()
pcs = set()
relay = MediaRelay()

# load model callables
analyze_ser = models_loader.load_ser()
analyze_fer = models_loader.load_fer()

def webm_to_wav_bytes(webm_bytes):
    # write webm bytes to temp file and call ffmpeg to convert to wav (16k mono)
    in_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    with open(in_path, 'wb') as f: f.write(webm_bytes)
    cmd = ['ffmpeg', '-y', '-i', in_path, '-ar', '16000', '-ac', '1', out_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out_path, 'rb') as f: wav = f.read()
    finally:
        try: os.remove(in_path)
        except: pass
        try: os.remove(out_path)
        except: pass
    return wav

def synthesize_tts_to_wav(text):
    # try pyttsx3, else gTTS + ffmpeg
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return out_path
    except Exception as e:
        try:
            from gtts import gTTS
            tmpmp3 = out_path + '.mp3'
            gTTS(text).save(tmpmp3)
            subprocess.run(['ffmpeg','-y','-i', tmpmp3, '-ar','16000','-ac','1', out_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(tmpmp3)
            return out_path
        except Exception as e2:
            print('TTS failed:', e, e2)
            return None

@app.post('/offer')
async def offer(req: Request):
    params = await req.json()
    offer_sdp = params.get('sdp')
    offer_type = params.get('type', 'offer')
    if not offer_sdp:
        return JSONResponse({'error':'sdp missing'}, status_code=400)

    pc = RTCPeerConnection()
    pcs.add(pc)
    dc = None

    @pc.on('datachannel')
    def on_datachannel(channel):
        nonlocal dc
        dc = channel
        print('[server] DataChannel created:', channel.label)

        @channel.on('message')
        def on_message(message):
            # if bytes => webm/opus blob, decode -> wav -> analyze
            try:
                if isinstance(message, (bytes, bytearray)):
                    # convert webm/opus to wav bytes using ffmpeg then run SER
                    try:
                        wav_bytes = webm_to_wav_bytes(message)
                        # write wav to temp file for analyze_ser which expects a file path
                        tmpwav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                        with open(tmpwav, 'wb') as f: f.write(wav_bytes)
                        res = analyze_ser(tmpwav)
                        # send back result
                        channel.send(json.dumps({'type':'SER','result':res}))
                        # optionally synthesize TTS reply and add audio track
                        reply_text = f"I hear {res.get('label','unknown')} (conf {res.get('confidence',0):.2f})"
                        tts_path = synthesize_tts_to_wav(reply_text)
                        if tts_path:
                            player = MediaPlayer(tts_path)
                            if player.audio:
                                pc.addTrack(relay.subscribe(player.audio))
                                channel.send(json.dumps({'type':'TTS','status':'added'}))
                    except Exception as e:
                        print('processing error', e)
                else:
                    # assume JSON control
                    obj = json.loads(message)
                    if obj.get('cmd') == 'fer_request':
                        # client asked for FER - respond with mock or real FER
                        res = analyze_fer(None)
                        channel.send(json.dumps({'type':'FER','result':res}))
            except Exception as e:
                print('dc message handler exception', e)

    @pc.on('track')
    def on_track(track):
        print('[server] Track received kind=', track.kind)
        # not required for this flow

    offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    print('[server] returning answer')
    return JSONResponse({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})

@app.on_event('shutdown')
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)

if __name__ == '__main__':
    print('Starting server_webrtc_real on :8000')
    uvicorn.run('server_webrtc_real:app', host='0.0.0.0', port=8000, log_level='info')
