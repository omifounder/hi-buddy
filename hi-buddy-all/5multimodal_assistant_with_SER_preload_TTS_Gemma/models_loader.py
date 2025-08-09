# models_loader.py (simplified loaders and placeholders)
import os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import tensorflow as tf

MODEL_CACHE = {}

def load_text_emotion_model(choice='distilroberta'):
    # map to HF model id or local path
    mapping = {'distilroberta': 'j-hartmann/emotion-english-distilroberta-base'}
    model_id = mapping.get(choice, choice)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    MODEL_CACHE['text'] = (tokenizer, model)
    return tokenizer, model

def load_video_emotion_model():
    path = 'models/fer_model.h5'
    if not os.path.exists(path):
        raise FileNotFoundError('FER model missing. Run download_models.py')
    model = tf.keras.models.load_model(path)
    MODEL_CACHE['video'] = model
    return model

def load_llm(name='gemma'):
    if name == 'cloud':
        return None, None
    repo = {'gemma':'google/gemma-2b-it', 'mistral':'mistralai/Mistral-7B-v0.1', 'vicuna':'lmsys/vicuna-7b-v1.5'}
    repo_id = repo.get(name, list(repo.values())[0])
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, device_map='auto' if torch.cuda.is_available() else None)
    MODEL_CACHE['llm'] = (tokenizer, model)
    return tokenizer, model
