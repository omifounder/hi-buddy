import requests, os
BASE = os.getenv('MP_BASE','http://localhost:8000')
def test_ping():
    r = requests.get(BASE + '/ping')
    assert r.status_code == 200
