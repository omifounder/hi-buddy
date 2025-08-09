Hybrid SER Demo Package
=======================

Files created under `/mnt/data/hybrid_ser_demo`:
- server/: FastAPI server with orchestration (app_orchestrator.py, strategies.py, requirements.txt, Dockerfile)
- client/web/: simple web clients (client_ser_webrtc_mode.html, ser_live_panel.html)
- docker-compose.yml

Quick start (no Docker)
-----------------------
1. Install system deps (Ubuntu):
   sudo apt update
   sudo apt install -y ffmpeg libsndfile1

2. Create venv and install Python deps:
   cd /mnt/data/hybrid_ser_demo/server
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. Run server:
   export INFERENCE_ROUTER_MODE=hybrid
   # optionally set ONNX_MODEL_PATH and CLOUD_ENDPOINT env vars
   uvicorn app_orchestrator:app --host 0.0.0.0 --port 8000 --reload

4. Serve client (in new terminal):
   cd /mnt/data/hybrid_ser_demo/client/web
   python3 -m http.server 8080
   Open http://localhost:8080/client_ser_webrtc_mode.html

Quick start (Docker)
--------------------
1. Build & start:
   docker-compose up --build

2. Open client as above (point to server host).

Notes
-----
- The server uses a DummyStrategy by default if no ONNX model is present.
- Replace preprocess and model with your actual ONNX/TFJS models for production.
- Adjust CONFIDENCE_THRESHOLD and routing logic as needed.
