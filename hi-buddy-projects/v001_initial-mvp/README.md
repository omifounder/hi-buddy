
# MVP Assistant - v001 Initial

This is the minimal viable product backend for the assistant project.

## Features
- Simple Flask API
- Local run only
- No Docker, GPU, or real model integration

## How to Run (Local)
1. Install dependencies:
    pip install flask

2. Start the server:
    python server.py

3. Test:
    curl http://127.0.0.1:5000/api/ping

Expected Response:
    {"status": "ok", "message": "MVP server running locally"}
