# download_models.py - preloads key models (may require HF access for some models)
from transformers import pipeline, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
from TTS.api import TTS
import gdown, os

os.makedirs('models', exist_ok=True)

print('Preloading text emotion model...')
pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

print('Preloading speech emotion model...')
Wav2Vec2FeatureExtractor.from_pretrained('j-hartmann/emotion-english-speech-distilroberta-base')
Wav2Vec2ForSequenceClassification.from_pretrained('j-hartmann/emotion-english-speech-distilroberta-base')

print('Preloading TTS model...')
try:
    TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC')
except Exception as e:
    print('TTS preload failed:', e)

print('Attempting to preload Gemma (will choose quantized fallback if no GPU)...')
model_id = 'google/gemma-2b-it'
try:
    AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')
    else:
        print('No GPU detected - try quantized download (requires bitsandbytes).')
        # try quantized load (may require bitsandbytes installed)
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_8bit=True)
            AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map='auto', trust_remote_code=True)
        except Exception as e:
            print('Quantized Gemma preload failed:', e)
except Exception as e:
    print('Gemma preload failed:', e)

print('Done.')
