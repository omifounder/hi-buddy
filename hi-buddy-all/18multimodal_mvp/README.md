Multimodal MVP - WebRTC + SER/FER demo
=====================================

This package provides a small MVP to demo real-time multimodal emotion detection using WebRTC DataChannel.

Contents:
- server_webrtc.py : FastAPI server that handles /offer and runs mock SER/FER on incoming WAV bytes over DataChannel.
- client_index.html : Client page that captures mic+camera, sends WAV chunks over a DataChannel every ~1s, and displays SER/FER results.
- requirements.txt : Python dependencies.

Quick start (local):
1. Create virtualenv and install deps:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Run the server:
   python server_webrtc.py

3. Serve the client (required for getUserMedia over file:// in some browsers):
   python3 -m http.server 8080
   Open http://localhost:8080/client_index.html

Notes:
- The server uses mock analyzers for SER and FER to keep the MVP lightweight. Replace analyze_ser_mock/analyze_fer_mock with calls into models_loader.load_ser()/load_fer() to use real models.
- aiortc requires libav/ffmpeg tooling and the 'av' Python package. If 'av' fails to install, ensure system FFmpeg dev packages are installed.
- This demo sends WAV bytes across DataChannel; in production you may prefer Opus streaming to save bandwidth.
