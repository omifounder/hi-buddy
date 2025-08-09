/* mel_spectrogram.js
Pure JS STFT -> mel spectrogram extractor for browser use.
Note: For best performance, prefer a WASM implementation. This is a reference implementation.
Usage:
  const mel = computeMelSpectrogram(float32Samples, sampleRate, {frameSize:1024, hopSize:512, melBins:64});
Returns Float32Array of shape [frames, melBins] flattened.
*/
function hannWindow(length){
  const w = new Float32Array(length);
  for(let i=0;i<length;i++) w[i] = 0.5 * (1 - Math.cos(2*Math.PI*i/(length-1)));
  return w;
}

function nextPow2(v){ return 1<<Math.ceil(Math.log2(v)); }

function fftMagnitudes(frame){ 
  // Simple FFT using built-in AnalyserNode not available here; include simple FFT implementation (Cooley-Tukey)
  // For brevity, we'll use a lightweight implementation (real FFT) - but note performance limits.
  const n = frame.length;
  // zero-pad to power of two
  const N = nextPow2(n);
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  for(let i=0;i<n;i++) re[i] = frame[i];
  // Cooley-Tukey (iterative)
  const logN = Math.log2(N);
  for(let s=1;s<=logN;s++){
    const m = 1<<s;
    const m2 = m>>1;
    const theta = -2*Math.PI/m;
    const wReal = 1.0;
    for(let k=0;k<m2;k++){
      const wr = Math.cos(theta*k);
      const wi = Math.sin(theta*k);
      for(let j=k;j<N;j+=m){
        const tRe = wr*re[j+m2] - wi*im[j+m2];
        const tIm = wr*im[j+m2] + wi*re[j+m2];
        re[j+m2] = re[j] - tRe;
        im[j+m2] = im[j] - tIm;
        re[j] += tRe;
        im[j] += tIm;
      }
    }
  }
  const mags = new Float32Array(N/2+1);
  for(let k=0;k<mags.length;k++) mags[k] = Math.hypot(re[k], im[k]);
  return mags;
}

function melFilterBank(numFilters, fftSize, sampleRate, fmin=0, fmax=null){
  if(!fmax) fmax = sampleRate/2;
  const mel = x => 1125*Math.log(1 + x/700);
  const invMel = m => 700*(Math.exp(m/1125)-1);
  const melMin = mel(fmin), melMax = mel(fmax);
  const melPoints = new Float32Array(numFilters+2);
  for(let i=0;i<melPoints.length;i++) melPoints[i] = melMin + (melMax-melMin)*i/(numFilters+1);
  const hzPoints = new Float32Array(melPoints.length);
  for(let i=0;i<hzPoints.length;i++) hzPoints[i] = invMel(melPoints[i]);
  const bin = new Int32Array(hzPoints.length);
  for(let i=0;i<hzPoints.length;i++) bin[i] = Math.floor((fftSize+1)*hzPoints[i]/sampleRate);
  const fb = new Array(numFilters);
  for(let m=1;m<=numFilters;m++){
    const f_m_minus = bin[m-1], f_m = bin[m], f_m_plus = bin[m+1];
    const filt = new Float32Array(fftSize/2+1);
    for(let k=f_m_minus;k<=f_m;k++) filt[k] = (k - f_m_minus) / (f_m - f_m_minus);
    for(let k=f_m;k<=f_m_plus;k++) filt[k] = (f_m_plus - k) / (f_m_plus - f_m);
    fb[m-1] = filt;
  }
  return fb;
}

function computeMelSpectrogram(samples, sampleRate, opts={frameSize:1024, hopSize:512, melBins:64}){
  const frameSize = opts.frameSize || 1024;
  const hopSize = opts.hopSize || 512;
  const melBins = opts.melBins || 64;
  const window = hannWindow(frameSize);
  const frames = Math.max(0, Math.floor((samples.length - frameSize)/hopSize) + 1);
  const fftSize = nextPow2(frameSize);
  const melFB = melFilterBank(melBins, fftSize, sampleRate);
  const melSpectrogram = new Float32Array(frames * melBins);
  let outIdx = 0;
  for(let i=0;i<frames;i++){
    const offset = i*hopSize;
    const frame = new Float32Array(frameSize);
    for(let j=0;j<frameSize;j++) frame[j] = (samples[offset+j] || 0) * window[j];
    const mags = fftMagnitudes(frame);
    for(let m=0;m<melBins;m++){
      let sum = 0;
      const filt = melFB[m];
      for(let k=0;k<filt.length;k++) sum += filt[k] * (mags[k] || 0);
      melSpectrogram[outIdx++] = Math.log10(1e-6 + sum);
    }
  }
  return {melSpectrogram, frames, melBins};
}

if(typeof module !== 'undefined') module.exports = { computeMelSpectrogram };
