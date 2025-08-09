# tools/e2e_client.py
# Simple E2E test: POST a WAV file to /infer_wav and measure latency + print result.
import sys, time, requests

def run_test(server_url, wav_path):
    with open(wav_path, 'rb') as f:
        data = f.read()
    start = time.time()
    r = requests.post(server_url.rstrip('/') + '/infer_wav', files={'file': ('test.wav', data)})
    elapsed = time.time() - start
    print('latency:', elapsed)
    print('status:', r.status_code)
    try:
        print('json:', r.json())
    except Exception as e:
        print('response text:', r.text)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python e2e_client.py http://localhost:8000 path/to/test.wav')
        sys.exit(1)
    run_test(sys.argv[1], sys.argv[2])
