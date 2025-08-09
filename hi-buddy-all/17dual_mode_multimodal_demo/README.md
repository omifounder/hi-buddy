Dual-mode Multimodal Demo (Hello-World + Full)
=============================================

This package provides a minimal demo that runs in two modes:
- HELLO_WORLD_MODE (default): lightweight mocks so you can test end-to-end without heavy downloads.
- Full mode: try to load real models via models_loader.py (you must implement those loaders).

Files:
- server.py               : FastAPI server with HELLO_WORLD_MODE toggle.
- models_loader.py        : stub to implement real model loading for Full mode.
- client/mobile_webrtc.html : lightweight mobile-friendly WebRTC wrapper.
- client/hello_world.html : simple client to test Hello-World mode (capture + POST).
- README.md               : this file.

Run Hello-World mode (local, no Docker):
----------------------------------------
1. Create venv & install (optional for Hello-World, but recommended):
   python3 -m venv venv
   source venv/bin/activate
   pip install fastapi uvicorn

2. Start server in Hello-World mode (default):
   HELLO_WORLD_MODE=true python server.py

3. Serve client files (in another terminal):
   cd client
   python3 -m http.server 8080
   Open http://localhost:8080/hello_world.html or /mobile_webrtc.html

Run Full mode (attempts to load heavy models):
----------------------------------------------
1. Implement models_loader.load_llm(), load_ser(), load_fer() to return real model callables.
2. Set HELLO_WORLD_MODE=false and run server:
   HELLO_WORLD_MODE=false python server.py

Docker usage:
-------------
Build a Docker image using the supplied Dockerfile and run with HELLO_WORLD_MODE=true for quick tests.

Notes:
- Hello-World mode is safe to run on low-end machines and in containers.
- Full mode requires you to implement actual model loaders and ensure dependencies (onnxruntime, transformers, torch, etc.) are installed.
