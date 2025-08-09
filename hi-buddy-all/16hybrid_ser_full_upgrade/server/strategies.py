# strategies.py
import os, tempfile, numpy as np, soundfile as sf

class StrategyBase:
    async def infer(self, wav_path):
        raise NotImplementedError()

class DummyStrategy(StrategyBase):
    async def infer(self, wav_path):
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

class LocalOnnxStrategy(StrategyBase):
    """Skeleton Local ONNX runtime strategy. Provide a preprocess_fn that maps wav_path -> numpy input
    matching the ONNX model's expected input shape."""
    def __init__(self, model_path, preprocess_fn=None, label_map=None):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError('onnxruntime not available: ' + str(e))
        self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.preprocess = preprocess_fn
        self.label_map = label_map or ["neutral","happy","sad","angry","fear","disgust","surprise"]

    async def infer(self, wav_path):
        try:
            if self.preprocess is None:
                raise RuntimeError('preprocess function not provided')
            inp = self.preprocess(wav_path)  # should return numpy array shaped as model expects
            input_name = self.sess.get_inputs()[0].name
            outs = self.sess.run(None, {input_name: inp})
            logits = np.array(outs[0]).squeeze()
            exps = np.exp(logits - np.max(logits))
            probs = exps / np.sum(exps)
            idx = int(np.argmax(probs))
            return {"label": self.label_map[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}
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
