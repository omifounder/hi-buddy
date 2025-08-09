import React, { useEffect } from 'react';
import './App.css';

function App() {
  useEffect(()=>{
    const s = document.createElement('script');
    s.src = '/ondevice_model_loader.js';
    document.body.appendChild(s);
  }, []);
  return (
    <div className="App">
      <header className="App-header">
        <h2>Mobile Multimodal Assistant - PWA</h2>
        <p>Installable PWA with on-device inference.</p>
      </header>
    </div>
  );
}

export default App;
