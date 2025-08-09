import pytest
import requests
import os

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

@pytest.mark.parametrize("audio_file", ["tests/samples/happy.wav", "tests/samples/angry.wav"])
def test_ser_endpoint(audio_file):
    files = {"file": open(audio_file, "rb")}
    response = requests.post(f"{BASE_URL}/api/ser", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "emotion" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0

def test_llm_endpoint():
    payload = {"text": "Hello, how are you?"}
    r = requests.post(f"{BASE_URL}/api/llm", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "reply" in data

def test_tts_endpoint(tmp_path):
    payload = {"text": "This is a TTS test."}
    r = requests.post(f"{BASE_URL}/api/tts", json=payload)
    assert r.status_code == 200
    out_file = tmp_path / "tts.wav"
    out_file.write_bytes(r.content)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
