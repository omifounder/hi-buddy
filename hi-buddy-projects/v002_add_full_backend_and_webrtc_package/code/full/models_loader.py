# Initial model hooks (stub) - to be replaced with real loaders in v003
def init_all():
    return {'llm': None, 'stt': None, 'fer': None}
def stt_transcribe(models, wav, sampling_rate=16000):
    return 'demo transcription'
def llm_generate(models, prompt):
    return 'demo reply to: ' + prompt[:120]
