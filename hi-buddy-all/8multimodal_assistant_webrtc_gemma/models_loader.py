\
# models_loader.py - simple loader for Gemma with quantized fallback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

_model_cache = {}

def load_llm(name='gemma'):
    if name in _model_cache:
        return _model_cache[name]
    repo_map = {'gemma':'google/gemma-2b-it'}
    model_id = repo_map.get(name, name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print('[models_loader] tokenizer load failed:', e)
        _model_cache[name] = (None,None,None)
        return None, None, None
    # GPU load
    if device == 'cuda':
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto', low_cpu_mem_usage=True)
            _model_cache[name] = (tokenizer, model, 'cuda')
            return tokenizer, model, 'cuda'
        except Exception as e:
            print('[models_loader] gpu load failed:', e)
    # try 8-bit quantized
    try:
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map='auto', trust_remote_code=True)
        _model_cache[name] = (tokenizer, model, 'cpu')
        return tokenizer, model, 'cpu'
    except Exception as e:
        print('[models_loader] quantized load failed:', e)
    # fallback CPU FP32 (may OOM)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        _model_cache[name] = (tokenizer, model, 'cpu')
        return tokenizer, model, 'cpu'
    except Exception as e:
        print('[models_loader] cpu load failed:', e)
    _model_cache[name] = (None,None,None)
    return None, None, None
