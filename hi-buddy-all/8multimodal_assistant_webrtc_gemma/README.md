Multimodal Assistant — WebRTC Live SER + Gemma + TTS + Dashboard

Quick start:
1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt  (install torch appropriate for your system first)
4. (optional) set HUGGINGFACE_HUB_TOKEN if needed for gated models
5. uvicorn server:app --reload
6. Open http://localhost:8000/ui_webrtc.html in Chrome
Notes:
- The server will try to load Gemma (google/gemma-2b-it) and a SER model.
- If Gemma is large for your hardware, the models_loader attempts quantized loading; ensure bitsandbytes is installed.
