# models_loader.py (updated - Gemma integration with quantized fallback)
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_CACHE = {}

def load_llm(name="gemma"):
    """Load an LLM for local inference.
    - If GPU available: loads FP16 model (better perf).
    - If no GPU: attempts 8-bit quantized (bitsandbytes) load.
    Returns (tokenizer, model, device).
    """

    if name in MODEL_CACHE:
        return MODEL_CACHE[name]

    if name == "cloud":
        MODEL_CACHE[name] = (None, None, None)
        return None, None, None

    repo_map = {
        "gemma": "google/gemma-2b-it",
        "vicuna": "lmsys/vicuna-7b-v1.5",
        "mistral": "mistralai/Mistral-7B-v0.1"
    }
    model_id = repo_map.get(name, name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"[LLM] Tokenizer load failed for {model_id}: {e}")
        MODEL_CACHE[name] = (None, None, None)
        return None, None, None

    # GPU FP16 load
    if device == "cuda":
        try:
            print(f"[LLM] Loading {model_id} on GPU (fp16)...")
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
            MODEL_CACHE[name] = (tokenizer, model, "cuda")
            return tokenizer, model, "cuda"
        except Exception as e:
            print(f"[LLM] GPU load failed: {e} — trying quantized/CPU fallback.")

    # Try quantized 8-bit via bitsandbytes
    try:
        print(f"[LLM] Trying 8-bit quantized load for {model_id}...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config, trust_remote_code=True)
        MODEL_CACHE[name] = (tokenizer, model, "cpu")
        return tokenizer, model, "cpu"
    except Exception as e:
        print(f"[LLM] Quantized load failed: {e}")

    # Last resort CPU FP32
    try:
        print(f"[LLM] Trying CPU FP32 load for {model_id}... (may OOM)")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        MODEL_CACHE[name] = (tokenizer, model, "cpu")
        return tokenizer, model, "cpu"
    except Exception as e:
        print(f"[LLM] CPU load failed: {e}")

    MODEL_CACHE[name] = (None, None, None)
    return None, None, None
