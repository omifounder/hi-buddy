# models_loader.py
import os

def load_ser():
    onnx = os.environ.get('SER_ONNX_PATH','').strip()
    if onnx and os.path.exists(onnx):
        try:
            import onnxruntime as ort, numpy as np, soundfile as sf, resampy
            sess = ort.InferenceSession(onnx, providers=['CPUExecutionProvider'])
            input_name = sess.get_inputs()[0].name
            labels = os.environ.get('SER_LABELS', 'angry,happy,neutral,sad').split(',')
            def preprocess(wav_path, target_sr=16000, max_len=16000):
                data, sr = sf.read(wav_path, dtype='float32')
                if data.ndim>1: data=data.mean(axis=1)
                if sr != target_sr:
                    data = resampy.resample(data, sr, target_sr)
                if len(data) < max_len:
                    data = np.pad(data, (0, max_len - len(data)))
                else:
                    data = data[:max_len]
                return data.reshape(1,-1).astype('float32')
            def analyze_ser(wav_path):
                try:
                    x = preprocess(wav_path)
                    outs = sess.run(None, {input_name: x})
                    logits = np.array(outs[0]).squeeze()
                    exps = np.exp(logits - logits.max())
                    probs = exps / exps.sum()
                    idx = int(probs.argmax())
                    return {'label': labels[idx] if idx < len(labels) else str(idx), 'confidence': float(probs[idx]), 'probs': probs.tolist()}
                except Exception as e:
                    return {'label':'error','confidence':0.0,'error':str(e)}
            print('[models_loader] Loaded SER ONNX at', onnx)
            return analyze_ser
        except Exception as e:
            print('[models_loader] Failed to load ONNX SER:', e)
    # fallback
    def analyze_ser_stub(wav_path):
        import soundfile as sf, numpy as np
        try:
            data, sr = sf.read(wav_path, dtype='float32')
            if data.ndim>1: data=data.mean(axis=1)
            energy = float((abs(data)).mean())
            if energy < 0.01: return {'label':'neutral','confidence':0.9}
            if energy < 0.05: return {'label':'sad','confidence':0.75}
            return {'label':'happy','confidence':0.8}
        except Exception as e:
            return {'label':'error','confidence':0.0,'error':str(e)}
    print('[models_loader] Using SER stub (no model)')
    return analyze_ser_stub

def load_fer():
    fer_path = os.environ.get('FER_MODEL_PATH','').strip()
    if fer_path and os.path.exists(fer_path):
        try:
            import torch, torchvision.transforms as T
            from PIL import Image
            # try loading a scripted or state_dict model
            model = torch.jit.load(fer_path, map_location='cpu') if fer_path.endswith('.pt') else None
            if model is None:
                # try AutoModelForImageClassification if repo available
                from transformers import AutoFeatureExtractor, AutoModelForImageClassification
                extractor = AutoFeatureExtractor.from_pretrained(fer_path)
                model = AutoModelForImageClassification.from_pretrained(fer_path)
                labels = os.environ.get('FER_LABELS', 'neutral,happy,sad,angry,surprise,disgust,fear').split(',')
                def analyze_fer_bytes(image_bytes):
                    try:
                        import io, numpy as np
                        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        tfm = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
                        x = tfm(img).unsqueeze(0)
                        with torch.no_grad():
                            out = model(x)
                        probs = torch.softmax(out.logits[0], dim=0).cpu().numpy().tolist()
                        idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                        return {'label': labels[idx], 'confidence': float(probs[idx])}
                    except Exception as e:
                        return {'label':'error','confidence':0.0,'error':str(e)}
                print('[models_loader] Loaded FER transformer-style model at', fer_path)
                return analyze_fer_bytes
            else:
                # generic torch-scripted model expecting image tensor
                def analyze_fer_bytes(image_bytes):
                    try:
                        import io, numpy as np
                        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        tfm = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
                        x = tfm(img).unsqueeze(0)
                        with torch.no_grad():
                            out = model(x)
                        probs = torch.softmax(out[0], dim=0).cpu().numpy().tolist()
                        idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                        return {'label': str(idx), 'confidence': float(probs[idx])}
                    except Exception as e:
                        return {'label':'error','confidence':0.0,'error':str(e)}
                print('[models_loader] Loaded scripted FER model at', fer_path)
                return analyze_fer_bytes
        except Exception as e:
            print('[models_loader] Failed to load FER model:', e)
    # fallback stub
    def analyze_fer_stub(image_bytes):
        return {'label':'neutral','confidence':0.7}
    print('[models_loader] Using FER stub (no model)')
    return analyze_fer_stub
