import os
import torch
from fastapi import FastAPI
from models_loader import load_llm

app = FastAPI()

DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gemma")
tokenizer_llm, model_llm, llm_device = load_llm(DEFAULT_LLM)

def generate_with_llm(user_text, emotion_label, max_new_tokens=150, temperature=0.7):
    if tokenizer_llm is None or model_llm is None:
        return f"I sense you might be feeling {emotion_label}. Tell me more."
    prompt = f"You are an empathetic assistant. The user appears to be '{emotion_label}'. Context: {user_text}\n\nAssistant:"
    inputs = tokenizer_llm(prompt, return_tensors="pt", truncation=True, max_length=2048)
    try:
        device = next(model_llm.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass
    with torch.no_grad():
        out = model_llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.9)
    text = tokenizer_llm.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

@app.get("/")
def root():
    return {"message": "MVP Gemma Server B Running"}
