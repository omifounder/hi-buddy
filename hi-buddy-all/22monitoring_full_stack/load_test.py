import requests, time, random, threading
URL = 'http://localhost:8000/infer_wav'

def worker():
    files = {'file': ('sample.wav', open('sample.wav','rb'), 'audio/wav')}
    while True:
        try:
            r = requests.post(URL, files=files, timeout=10)
            print('resp', r.status_code, r.text)
        except Exception as e:
            print('err', e)
        time.sleep(random.uniform(0.2, 1.0))

if __name__ == '__main__':
    # spawn a few threads
    threads = []
    for i in range(4):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)
    while True:
        time.sleep(1)
