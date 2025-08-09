# mvp/server/models_loader.py - stubs for SER and TTS used in MVP
import tempfile, wave, struct, time, math, os

def analyze_ser(wav_path):
    # Quick stub: pretend to analyze and return neutral
    time.sleep(0.05)
    return {'label':'neutral','confidence':0.80}

def synthesize_tts_to_wav(text):
    # Create a tiny silent WAV and return the path
    out = os.path.join('/tmp', 'mvp_tts_' + str(int(time.time())) + '.wav')
    framerate = 16000
    duration = 0.6
    nframes = int(framerate * duration)
    with wave.open(out, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        for i in range(nframes):
            wf.writeframes(struct.pack('<h', 0))
    return out
