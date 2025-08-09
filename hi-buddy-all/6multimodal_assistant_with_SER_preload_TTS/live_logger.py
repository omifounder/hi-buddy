import os, csv, json, threading, asyncio, websockets
from datetime import datetime
LOG_FILE = 'logs/conversation_logs.csv'
os.makedirs('logs/media', exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE,'w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp','user_text','emotion_text','emotion_audio','emotion_video','fused_emotion','llm_model','llm_response','audio_file','video_frame_file','inference_latency_ms','response_sentiment'])

broadcast_queue = asyncio.Queue()
clients = set()

def log_interaction(user_text, emotion_text, emotion_audio, emotion_video, fused_emotion, llm_model, llm_response, audio_file=None, video_frame_file=None, inference_latency_ms=None, response_sentiment=None):
    timestamp = datetime.utcnow().isoformat()
    row = [timestamp,user_text,emotion_text,emotion_audio,emotion_video,fused_emotion,llm_model,llm_response,audio_file or '',video_frame_file or '',inference_latency_ms or '',response_sentiment or '']
    with open(LOG_FILE,'a',newline='',encoding='utf-8') as f:
        csv.writer(f).writerow(row)
    payload = {'timestamp':timestamp,'user_text':user_text,'emotion_text':emotion_text,'emotion_audio':emotion_audio,'emotion_video':emotion_video,'fused_emotion':fused_emotion,'llm_model':llm_model,'llm_response':llm_response,'audio_file':audio_file,'video_frame_file':video_frame_file,'inference_latency_ms':inference_latency_ms,'response_sentiment':response_sentiment}
    loop = asyncio.get_event_loop()
    asyncio.run_coroutine_threadsafe(broadcast_queue.put(payload), loop)

async def ws_handler(websocket, path):
    clients.add(websocket)
    try:
        while True:
            payload = await broadcast_queue.get()
            await asyncio.wait([c.send(json.dumps(payload)) for c in clients], timeout=1)
    except:
        pass
    finally:
        clients.remove(websocket)

def start_ws_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websockets.serve(ws_handler,'localhost',8765))
    loop.run_forever()

threading.Thread(target=start_ws_server,daemon=True).start()
