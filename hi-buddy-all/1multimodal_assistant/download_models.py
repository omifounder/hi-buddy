# Lightweight downloader: fetch FER model (Keras h5) and text emotion model via HF cache.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gdown, os
os.makedirs('models', exist_ok=True)
print('Downloading FER model...')
FER_URL='https://drive.google.com/uc?id=1pE91oymFKkV-UumAqQG_GbmDUo2cV7Qu'
if not os.path.exists('models/fer_model.h5'):
    gdown.download(FER_URL, 'models/fer_model.h5', quiet=False)
print('Done.')
print('Downloading text emotion model to HF cache...')
AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
AutoModelForSequenceClassification.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
print('Text model cached.')
