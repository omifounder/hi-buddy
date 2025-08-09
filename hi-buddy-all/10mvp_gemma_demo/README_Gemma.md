# MVP Gemma Demo

## Setup
1. Install dependencies:
```bash
pip install fastapi uvicorn torch transformers bitsandbytes
```

2. Set Hugging Face token (needed for Gemma):
```bash
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

3. Run:
```bash
uvicorn server_b:app --reload --host 0.0.0.0 --port 8000
```
