# tools/e2e_client.py
import argparse
import asyncio
import json
import sys
import wave
import aiohttp

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRecorder
from av import AudioFrame

class AudioFileTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self, path):
        super().__init__()
        self.wav = wave.open(path, "rb")
        self.sample_rate = self.wav.getframerate()
        self.channels = self.wav.getnchannels()
    async def recv(self):
        frames = self.wav.readframes(960)
        if not frames:
            await asyncio.sleep(0.5)
            raise asyncio.CancelledError
        aframe = AudioFrame(format="s16", layout="stereo" if self.channels == 2 else "mono", samples=960)
        aframe.pts = None
        aframe.sample_rate = self.sample_rate
        aframe.planes[0].update(frames)
        return aframe

async def run(server_url, audio_file):
    pc = RTCPeerConnection()
    dc = pc.createDataChannel("client_channel")
    results = {"ser": False, "llm": False}
    @dc.on("message")
    def on_message(message):
        try:
            data = json.loads(message)
        except Exception:
            print(f"[DC] Non-JSON message: {message}")
            return
        if "ser_result" in data:
            print(f"[DC] SER: {data['ser_result']}")
            results["ser"] = True
        if "llm_reply" in data:
            print(f"[DC] LLM: {data['llm_reply']}")
            results["llm"] = True
    audio_track = AudioFileTrack(audio_file)
    pc.addTrack(audio_track)
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{server_url}/offer", json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }) as resp:
            answer = await resp.json()
    await pc.setRemoteDescription(RTCSessionDescription(
        sdp=answer["sdp"], type=answer["type"]
    ))
    try:
        await asyncio.sleep(15)
    finally:
        await pc.close()
    if results["ser"] and results["llm"]:
        print("[E2E] PASS: Both SER and LLM reply received")
        sys.exit(0)
    else:
        print("[E2E] FAIL: Missing results:", results)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.server, args.audio))
