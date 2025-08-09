
import torch
from transformers import pipeline, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

def preload_models():
    print("Downloading text emotion model...")
    pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    print("Downloading speech emotion model...")
    Wav2Vec2FeatureExtractor.from_pretrained("j-hartmann/emotion-english-speech-distilroberta-base")
    Wav2Vec2ForSequenceClassification.from_pretrained("j-hartmann/emotion-english-speech-distilroberta-base")

    print("Downloading video emotion model (FER2013)...")
    try:
        import fer
        fer.FER(mtcnn=True)
    except Exception as e:
        print("FER download skipped or failed:", e)

    print("Downloading TTS model...")
    try:
        from TTS.api import TTS
        TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    except Exception as e:
        print("TTS download skipped or failed:", e)

    print("Downloading Gemma LLM...")
    try:
        model_id = "google/gemma-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if torch.cuda.is_available():
            print("GPU detected: loading full precision Gemma")
            AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        else:
            print("No GPU detected: loading quantized Gemma (8-bit)")
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")
    except Exception as e:
        print("Gemma download skipped or failed:", e)

if __name__ == "__main__":
    preload_models()
