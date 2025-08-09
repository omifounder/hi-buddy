Multimodal Upgrade — Opus DataChannel + Models + TTS
==================================================

This package upgrades the MVP to:
- Use real model hooks via models_loader.py (loads SER ONNX if provided)
- Send Opus (audio/webm;codecs=opus) blobs from client via MediaRecorder over DataChannel
- Server decodes incoming webm/opus blobs via ffmpeg -> WAV -> analyze_ser
- Server synthesizes TTS and streams the WAV as an outgoing audio track (MediaPlayer)

Files:
- server_webrtc_real.py : FastAPI + aiortc server that decodes blobs and runs models_loader.analyze_ser
- client_opus.html      : Browser client that captures audio/video, sends opus blobs over DataChannel, displays SER/FER
- models_loader.py      : loader that tries to load SER ONNX model if SER_ONNX_PATH provided else stub
- requirements.txt      : python deps (install in venv)
- README.md

Quick start (local):
1. Install system packages (Ubuntu):
   sudo apt update
   sudo apt install -y ffmpeg libsndfile1 build-essential python3-dev

2. Create virtualenv, install Python deps:
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

3. (Optional) Place ONNX SER model and set env:
   export SER_ONNX_PATH=/abs/path/to/your/model.onnx

4. Run server:
   python server_webrtc_real.py

5. Serve client files and open browser:
   python3 -m http.server 8080
   Open http://localhost:8080/client_opus.html

Notes & Troubleshooting:
- ffmpeg must be installed and in PATH for server to decode webm blobs.
- aiortc and av may require system libraries; if 'av' fails to install, install ffmpeg dev packages or use Docker.
- Adjust SER_LABELS env if your model has different labels.
- TTS uses pyttsx3 (offline) or gTTS (internet) as fallback. Ensure network if using gTTS.

