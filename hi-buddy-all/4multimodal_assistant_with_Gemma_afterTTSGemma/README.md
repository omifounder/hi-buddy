# Multimodal Assistant (local prototype)
This repository is a **local prototype** of a multimodal assistant with:
- Text emotion detection
- Video (FER) emotion detection
- Speech Emotion Recognition (Wav2Vec2, hook placeholder)
- Fusion of emotions
- Local LLM selection (Gemma/Mistral/Vicuna placeholders)
- **TTS streaming to the browser via aiortc** (Coqui TTS example)
- Live logging and a WebSocket broadcaster for dashboards
- Simple HTML UI and dashboard

**WARNING**: This is a prototype for local testing. Some large LLM model downloads may require manual steps (Git LFS / Hugging Face authentication). See `download_models.py` and `models_loader.py` for details.

## Quick start (local, cpu)
1. Create and activate venv
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. (Optional) download models:
```bash
python download_models.py
```
3. Run the server:
```bash
python server.py
```
4. Open UI at `http://localhost:8000/` and open `dashboard.html` in a browser for live logs (`dashboard.html` connects to the logger WS at ws://localhost:8765).

If you want a Docker image, build with `docker build -t multimodal_assistant .` (may take long and require model downloads).

