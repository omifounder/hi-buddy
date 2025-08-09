import asyncio
import json
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaPlayer

SIGNALING_URL = "ws://localhost:8000/ws"

async def run_test():
    async with websockets.connect(SIGNALING_URL) as ws:
        pc = RTCPeerConnection()
        player = MediaPlayer("tests/samples/happy.wav")
        pc.addTrack(player.audio)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        await ws.send(json.dumps({"type": "offer", "sdp": pc.localDescription.sdp}))
        msg = await ws.recv()
        answer = json.loads(msg)

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type="answer")
        )
        print("✅ WebRTC connection established")

asyncio.run(run_test())
