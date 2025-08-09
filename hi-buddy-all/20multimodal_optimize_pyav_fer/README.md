Multimodal Optimized Package - PyAV decoding, FFmpeg worker pool, FER integration
===============================================================================

What's new:
- PyAV-based decoding of Opus/WebM blobs (preferred for lower overhead and streaming decoding)
- FFmpeg worker pool fallback (runs ffmpeg per blob in a worker thread; reduces CPU spikes by pooling)
- FER model integration: set FER_MODEL_PATH to a huggingface-compatible image classifier or a TorchScript .pt
- TTS reuse via a persistent outgoing track (tts_player) to avoid too many tracks

Files:
- server_webrtc_opt.py : optimized FastAPI + aiortc server (PyAV primary, ffmpeg pool fallback)
- client_opus.html     : client capturing Opus blobs + FER snapshot support
- models_loader.py     : loads SER (ONNX) and FER (Torch/Transformers) if paths provided
- requirements.txt, README.md

Setup & Run:
1. Install system deps (Ubuntu):
   sudo apt update
   sudo apt install -y ffmpeg libsndfile1 build-essential python3-dev

2. Create venv and install Python deps:
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

3. Provide models (optional):
   export SER_ONNX_PATH=/abs/path/to/ser_model.onnx
   export FER_MODEL_PATH=/abs/path/to/fer_model_folder_or_pt

4. Run server:
   python server_webrtc_opt.py

5. Serve client and open:
   python3 -m http.server 8080
   Open http://localhost:8080/client_opus.html

Notes & tuning:
- PyAV decoding is efficient and avoids frequent ffmpeg subprocess spawns. If PyAV not available or decoding fails, the server uses an ffmpeg worker pool to transcode webm->wav in worker threads.
- ffmpeg worker pool size controlled via env FFMPEG_WORKERS (default 2).
- FER integration expects either a transformers-style model repo (with feature extractor & model) or a TorchScript .pt file. Adjust FER_LABELS env var if needed.
- TTS uses pyttsx3 or gTTS fallback. The TTS player adds outgoing audio tracks; reuse and proper audio handling may require custom MediaStreamTrack implementation for production.

Troubleshooting:
- aiortc/av installation issues: ensure ffmpeg dev libs installed; consider using Docker with prebuilt wheels.
- PyAV import errors: install `av` and system libav packages.
- If decoding fails for certain browser blobs, check browser's MediaRecorder encoding options; some browsers produce different container params.
