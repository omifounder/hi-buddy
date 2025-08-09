import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_llm(name="gemma"):
    model_map = {
        "gemma": "google/gemma-2b-it"
    }
    model_id = model_map.get(name, model_map["gemma"])

    print(f"[Model Loader] Loading LLM: {model_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"[Error] Tokenizer load failed: {e}")
        return None, None, None

    try:
        if torch.cuda.is_available():
            try:
                print("[Model Loader] Trying GPU FP16")
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
                return tokenizer, model, "cuda"
            except:
                print("[Model Loader] Trying 8-bit quantized mode")
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
                return tokenizer, model, "cuda"
        print("[Model Loader] Using CPU fallback")
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
        return tokenizer, model, "cpu"
    except Exception as e:
        print(f"[Error] Model load failed: {e}")
        return tokenizer, None, "cpu"
