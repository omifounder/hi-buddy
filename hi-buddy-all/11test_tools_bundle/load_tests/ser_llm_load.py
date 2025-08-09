# load_tests/ser_llm_load.py
from locust import HttpUser, task, between
import random
import json

class SERLLMLoadUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def ser_inference(self):
        files = {"file": ("sample.wav", b"\\x00" * 1000, "audio/wav")}
        with self.client.post("/api/ser", files=files, catch_response=True) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if "label" in data:
                        resp.success()
                    else:
                        resp.failure("No label in SER response")
                except Exception as e:
                    resp.failure(f"JSON parse error: {e}")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def llm_orchestration(self):
        payload = {
            "text": random.choice(["Hello!", "I feel happy today.", "It's been a tough day."]),
            "emotion": random.choice(["happy", "sad", "neutral"]),
        }
        headers = {"Content-Type": "application/json"}
        with self.client.post("/api/llm", data=json.dumps(payload), headers=headers, catch_response=True) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if "reply" in data:
                        resp.success()
                    else:
                        resp.failure("No reply in LLM response")
                except Exception as e:
                    resp.failure(f"JSON parse error: {e}")
            else:
                resp.failure(f"HTTP {resp.status_code}")
