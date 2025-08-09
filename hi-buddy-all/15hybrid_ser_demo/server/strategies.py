# server/strategies.py
import os, tempfile, numpy as np, soundfile as sf

class StrategyBase:
    async def infer(self, wav_path):
        raise NotImplementedError()

class DummyStrategy(StrategyBase):
    async def infer(self, wav_path):
        # Simple energy heuristic - demo only
        try:
            data, sr = sf.read(wav_path, dtype='float32')
            if data.ndim > 1: data = data.mean(axis=1)
            energy = float(np.mean(np.abs(data)))
            if energy < 0.01:
                return {"label":"neutral","confidence":0.6}
            if energy < 0.05:
                return {"label":"sad","confidence":0.7}
            return {"label":"happy","confidence":0.65}
        except Exception as e:
            return {"label":"error","confidence":0.0, "error": str(e)}

class LocalONNXStrategy(StrategyBase):
    def __init__(self, model_path, preprocess_fn):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.preprocess = preprocess_fn

    async def infer(self, wav_path):
        try:
            inp = self.preprocess(wav_path)
            input_name = self.sess.get_inputs()[0].name
            outs = self.sess.run(None, {input_name: inp})
            logits = np.array(outs[0]).squeeze()
            # stable softmax
            exps = np.exp(logits - np.max(logits))
            probs = exps / np.sum(exps)
            idx = int(np.argmax(probs))
            return {"label_index": idx, "confidence": float(probs[idx]), "probs": probs.tolist()}
        except Exception as e:
            return {"label":"error","confidence":0.0, "error": str(e)}

class ForwardToCloudStrategy(StrategyBase):
    def __init__(self, endpoint_url, api_key=None, timeout=15.0):
        import requests
        self.requests = requests
        self.endpoint = endpoint_url
        self.api_key = api_key
        self.timeout = timeout

    async def infer(self, wav_path):
        try:
            files = {"file": open(wav_path, "rb")}
            headers = {}
            if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"
            r = self.requests.post(self.endpoint, files=files, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"label":"error","confidence":0.0, "error": str(e)}
