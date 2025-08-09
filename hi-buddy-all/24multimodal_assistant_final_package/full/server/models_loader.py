# full/server/models_loader.py - implement real model loading here
import logging, os, time

def load_ser():
    def analyze(wav_path):
        # insert real model inference here
        time.sleep(0.05)
        return {'label':'neutral','confidence':0.85}
    return analyze

def load_fer():
    def analyze(image_bytes):
        return {'label':'neutral','confidence':0.8}
    return analyze

def load_llm():
    class LLMStub:
        def generate(self, prompt, stream=False):
            return 'LLM stub reply for prompt: ' + prompt[:200]
    return LLMStub()

def load_tts():
    def synth(text):
        out = '/tmp/full_tts_' + str(int(time.time())) + '.wav'
        # create tiny silent file
        import wave, struct
        framerate = 16000; duration = 0.6
        with wave.open(out,'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(framerate)
            for i in range(int(framerate*duration)):
                wf.writeframes(struct.pack('<h', 0))
        return out
    return synth
