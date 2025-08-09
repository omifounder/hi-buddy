from locust import HttpUser, task, between
import random

class AssistantUser(HttpUser):
    wait_time = between(1, 3)

    @task(2)
    def ser_emotion(self):
        with open("tests/samples/happy.wav", "rb") as f:
            self.client.post("/api/ser", files={"file": f})

    @task(3)
    def llm_generate(self):
        prompts = ["Hello", "Tell me a joke", "What's the weather like?"]
        self.client.post("/api/llm", json={"text": random.choice(prompts)})

    @task(1)
    def tts_voice(self):
        self.client.post("/api/tts", json={"text": "Testing load"})
