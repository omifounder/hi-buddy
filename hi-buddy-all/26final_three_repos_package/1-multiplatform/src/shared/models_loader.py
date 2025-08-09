# Shared model loader hooks - replace with concrete implementations
def load_ser():
    def analyze(wav_path):
        return {'label':'neutral','confidence':0.8}
    return analyze

def load_fer():
    def analyze(image_bytes):
        return {'label':'neutral','confidence':0.75}
    return analyze

def load_llm():
    class LLMStub:
        def generate(self, prompt, stream=False):
            return "Stub LLM reply for: " + prompt[:200]
    return LLMStub()

def load_tts():
    def synth(text):
        # produce small silent wav for demos (path)
        import wave, struct, time, os
        out = os.path.join('/tmp', 'tts_' + str(int(time.time())) + '.wav')
        framerate = 16000; duration = 0.5
        with wave.open(out,'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(framerate)
            for i in range(int(framerate*duration)):
                wf.writeframes(struct.pack('<h', 0))
        return out
    return synth
