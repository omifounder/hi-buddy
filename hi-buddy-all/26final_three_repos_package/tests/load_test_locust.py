from locust import HttpUser, task, between
import random

class AssistantUser(HttpUser):
    wait_time = between(1,3)

    @task(2)
    def ser(self):
        with open('tests/samples/happy.wav','rb') as ff:
            self.client.post('/infer_wav', files={'file': ff})

    @task(1)
    def llm(self):
        self.client.post('/generate_reply', json={'text':'hello'})
