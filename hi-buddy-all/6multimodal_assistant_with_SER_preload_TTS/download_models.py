
from transformers import pipeline, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch

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
        # This downloads a default fast model (can be replaced with enterprise one)
        TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    except Exception as e:
        print("TTS download skipped or failed:", e)

if __name__ == "__main__":
    preload_models()
