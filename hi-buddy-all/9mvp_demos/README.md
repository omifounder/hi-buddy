
MVP Demos - Option B and Option C (chunked TTS)

Structure:
- server_b.py + client_b.html -> Option B (audio-only SER -> LLM reply -> chunked TTS)
- server_c.py + client_c.html -> Option C (audio+video fusion -> LLM reply -> chunked TTS)

How to run (example):
1. Save files to a folder.
2. Create venv and activate:
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\\Scripts\\activate
3. Install torch appropriate for your system (see pytorch.org), then:
   pip install -r requirements.txt
4. (Optional) Hugging Face auth:
   pip install huggingface_hub
   huggingface-cli login
5. Start a server (choose B or C):
   uvicorn server_b:app --host 0.0.0.0 --port 8000 --reload
   or
   uvicorn server_c:app --host 0.0.0.0 --port 8000 --reload
6. Open in Chrome:
   http://localhost:8000/client_b.html  (Option B)
   or
   http://localhost:8000/client_c.html  (Option C)

Notes:
- TTS uses Coqui TTS if installed; if unavailable, the client still receives text replies (which the browser can speak).
- FER uses the 'fer' package if installed; otherwise a simple heuristic is used.
- Replace demo LLM with Gemma using the models_loader approach we discussed earlier for production.
