Extra Package for Mobile On-Device Workflows
============================================

Files included:
- train_ser_small.py  : Train a tiny SER demo model (synthetic data)
- convert_to_tfjs.sh  : Convert SavedModel to TFJS layers model
- mel_extractor.js    : Example using Meyda to extract audio features in browser
- react_pwa/          : Minimal React PWA scaffold (install deps to run)
- capacitor_extra/    : Capacitor build helper notes

Notes:
- The train_ser_small.py creates a toy model on synthetic data for demo purposes. Replace with real trained models for production.
- After training, convert model to TFJS using tensorflowjs_converter (pip install tensorflowjs).
- Use Meyda (or a WASM STFT implementation) to compute mel spectrograms in the browser to feed TFJS SER model.
