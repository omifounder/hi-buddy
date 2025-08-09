#!/usr/bin/env python3
# server_webrtc_opt.py - optimized server: PyAV decoding for opus/webm, ffmpeg worker pool fallback, models_loader usage, TTS track reuse
import os, asyncio, tempfile, json, uuid, subprocess, shutil
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn, logging
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaPlayer, MediaStreamTrack
    from aiortc.contrib.media import MediaRelay
except Exception as e:
    raise RuntimeError('aiortc & av required') from e

# local modules
import models_loader

app = FastAPI()
pcs = set()
relay = MediaRelay()
logger = logging.getLogger('server_webrtc_opt')
logger.setLevel(logging.INFO)

# load models
analyze_ser = models_loader.load_ser()
analyze_fer = models_loader.load_fer()

# Persistent TTS outgoing track management (reuse single MediaPlayer by replacing file)
class TTSPlayer:
    def __init__(self):
        self.current = None  # MediaPlayer
        self.track = None

    def add_tts(self, pc, wav_path):
        try:
            player = MediaPlayer(wav_path)
            if player.audio:
                pc.addTrack(relay.subscribe(player.audio))
                logger.info('TTS track added for %s', wav_path)
                self.current = player
                return True
        except Exception as e:
            logger.exception('TTS add failed: %s', e)
        return False

tts_player = TTSPlayer()

# ffmpeg worker pool (simple threadpool that runs ffmpeg per task to decode webm->wav)
from concurrent.futures import ThreadPoolExecutor
FFMPEG_WORKERS = int(os.environ.get('FFMPEG_WORKERS','2'))
executor = ThreadPoolExecutor(max_workers=FFMPEG_WORKERS)

def ffmpeg_decode_webm_to_wav_bytes(webm_bytes):
    # write input file then call ffmpeg to convert
    in_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    try:
        with open(in_path, 'wb') as f: f.write(webm_bytes)
        cmd = ['ffmpeg','-y','-i', in_path, '-ar','16000','-ac','1', out_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out_path, 'rb') as f: wav = f.read()
        return wav
    finally:
        for p in (in_path, out_path):
            try: os.remove(p)
            except: pass

# Try PyAV decode (preferred) - returns wav bytes or raise
def pyav_decode_webm_to_wav_bytes(webm_bytes):
    try:
        import av, io, numpy as np, soundfile as sf, resampy
        container = av.open(io.BytesIO(webm_bytes))
        # collect all audio frames, convert to numpy
        samples = []
        sample_rate = None
        for frame in container.decode(audio=0):
            frame_samples = frame.to_ndarray().astype('float32')
            # frame_samples shape: (channels, samples) or (samples,) depending
            if frame_samples.ndim > 1:
                # average channels
                frame_mono = frame_samples.mean(axis=0)
            else:
                frame_mono = frame_samples
            if sample_rate is None:
                sample_rate = frame.sample_rate
            samples.append(frame_mono)
        if not samples:
            raise RuntimeError('no audio decoded by PyAV')
        audio = np.concatenate(samples)
        # resample to 16k if needed
        if sample_rate != 16000:
            try:
                audio = resampy.resample(audio, sample_rate, 16000)
            except Exception:
                import numpy as _np
                idx = _np.round(_np.linspace(0, len(audio)-1, int(len(audio) * 16000 / sample_rate))).astype(int)
                audio = audio[idx]
        # write to wav bytes
        out = io.BytesIO()
        sf.write(out, audio, 16000, format='WAV', subtype='PCM_16')
        return out.getvalue()
    except Exception as e:
        raise

@app.post('/offer')
async def offer(req: Request):
    params = await req.json()
    offer_sdp = params.get('sdp')
    offer_type = params.get('type','offer')
    if not offer_sdp:
        return JSONResponse({'error':'sdp missing'}, status_code=400)

    pc = RTCPeerConnection()
    pcs.add(pc)
    dc = None

    @pc.on('datachannel')
    def on_datachannel(channel):
        nonlocal dc
        dc = channel
        logger.info('DataChannel created: %s', channel.label)

        @channel.on('message')
        def on_message(message):
            try:
                if isinstance(message, (bytes, bytearray)):
                    # try PyAV decode first
                    try:
                        wav_bytes = pyav_decode_webm_to_wav_bytes(message)
                        # write temp wav file
                        tmpwav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                        with open(tmpwav, 'wb') as f: f.write(wav_bytes)
                        res = analyze_ser(tmpwav)
                        channel.send(json.dumps({'type':'SER','result':res}))
                        # TTS reply
                        reply_text = f"I detected {res.get('label','unknown')} (conf {res.get('confidence',0):.2f})"
                        tts_path = synthesize_tts_to_wav(reply_text)
                        if tts_path:
                            tts_player.add_tts(pc, tts_path)
                            channel.send(json.dumps({'type':'TTS','status':'added'}))
                        try: os.remove(tmpwav)
                        except: pass
                    except Exception as e:
                        # fallback to ffmpeg worker pool decoding
                        future = executor.submit(ffmpeg_decode_webm_to_wav_bytes, message)
                        def _cb(fut):
                            try:
                                wavb = fut.result()
                                tmpwav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                                with open(tmpwav,'wb') as f: f.write(wavb)
                                res = analyze_ser(tmpwav)
                                channel.send(json.dumps({'type':'SER','result':res}))
                                reply_text = f"I detected {res.get('label','unknown')} (conf {res.get('confidence',0):.2f})"
                                tts_path = synthesize_tts_to_wav(reply_text)
                                if tts_path:
                                    tts_player.add_tts(pc, tts_path)
                                    channel.send(json.dumps({'type':'TTS','status':'added'}))
                                try: os.remove(tmpwav)
                                except: pass
                            except Exception as ex:
                                logger.exception('ffmpeg worker failed: %s', ex)
                                try: channel.send(json.dumps({'type':'error','error':str(ex)}))
                                except: pass
                        future.add_done_callback(_cb)
                else:
                    try:
                        obj = json.loads(message)
                        if obj.get('cmd') == 'fer_request':
                            # If client sends a frame (base64), run FER
                            img_b64 = obj.get('img_b64')
                            if img_b64:
                                import base64, io
                                img_bytes = base64.b64decode(img_b64)
                                res = analyze_fer(img_bytes)
                                channel.send(json.dumps({'type':'FER','result':res}))
                            else:
                                res = analyze_fer(None)
                                channel.send(json.dumps({'type':'FER','result':res}))
                    except Exception as e:
                        logger.exception('DC JSON handler failed: %s', e)
            except Exception as e:
                logger.exception('DC message handler exception: %s', e)

    @pc.on('track')
    def on_track(track):
        logger.info('Track received kind=%s', track.kind)

    offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info('Returning answer')
    return JSONResponse({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})

def synthesize_tts_to_wav(text):
    # try pyttsx3 then gTTS as fallback
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
            logger.exception('TTS synthesis failed: %s %s', e, e2)
            return None

@app.on_event('shutdown')
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    executor.shutdown(wait=True)

if __name__ == '__main__':
    print('Starting server_webrtc_opt on :8000')
    uvicorn.run('server_webrtc_opt:app', host='0.0.0.0', port=8000, log_level='info')
