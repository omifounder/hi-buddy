# models_loader.py - stub for loading heavy models in Full mode.
# Implement load_llm(), load_ser(), load_fer() for your models (Gemma, Wav2Vec, FER models).
def load_llm():
    raise NotImplementedError('Implement load_llm to return an object with .generate(prompt)')

def load_ser():
    # Return a callable: analyze_ser(wav_path) -> {'label':..., 'confidence':...}
    raise NotImplementedError('Implement load_ser to return a callable that accepts a wav path')

def load_fer():
    # Return a callable: analyze_fer(image_bytes) -> {'label':..., 'confidence':...}
    raise NotImplementedError('Implement load_fer to return a callable that accepts image bytes')
