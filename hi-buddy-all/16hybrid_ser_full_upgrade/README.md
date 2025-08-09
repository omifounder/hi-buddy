Hybrid SER Full Upgrade Package
===============================

This package contains:
- server/: FastAPI REST orchestrator and aiortc DataChannel server to accept binary audio, with pluggable strategies.
- client/web/: client pages for local TFJS, REST chunk upload, and WebRTC DC raw WAV sending.
- tools/: e2e_client.py for latency/accuracy testing.
- convert_tools/: scripts to export a HuggingFace audio classification model to ONNX and convert to TFJS.

Quick run (no Docker):
1) Install system deps:
   sudo apt update
   sudo apt install -y ffmpeg libsndfile1 build-essential pkg-config

2) Create venv and install Python deps:
   cd server
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3) Start REST server:
   export INFERENCE_ROUTER_MODE=hybrid
   uvicorn app_orchestrator:app --host 0.0.0.0 --port 8000 --reload

4) Optionally start WebRTC DC server (aiortc):
   python server_webrtc.py

5) Serve client files:
   cd ../client/web
   python3 -m http.server 8080
   Open http://localhost:8080/client_hybrid.html

E2E test:
   python tools/e2e_client.py http://localhost:8000 path/to/test.wav

Model conversion:
   See convert_tools/*.py and README in this package. Converting the HF model requires internet and may be heavy.

Notes:
- Replace LocalOnnxStrategy preprocess function to match your ONNX model input.
- The DataChannel DC server expects raw WAV bytes sent as ArrayBuffer from client (client_hybrid.html sends them).
