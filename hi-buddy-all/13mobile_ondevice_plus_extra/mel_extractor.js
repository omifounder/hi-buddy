/* mel_extractor.js
Using Meyda to compute mel-spectrogram frames from an AudioContext source.
Include Meyda via CDN: <script src="https://unpkg.com/meyda/dist/web/meyda.min.js"></script>
*/
async function createMeydaAnalyzer(stream, onFeature) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(stream);

  // Meyda analyzer options
  const bufferSize = 1024;
  const melBands = 64;

  const analyzer = Meyda.createMeydaAnalyzer({
    audioContext: audioCtx,
    source: src,
    bufferSize: bufferSize,
    featureExtractors: ['mfcc', 'rms', 'spectralFlatness'],
    callback: (features) => {
      if (onFeature) onFeature(features);
    }
  });
  analyzer.start();
  return analyzer;
}

// Example usage:
// createMeydaAnalyzer(stream, (features)=>console.log(features));
