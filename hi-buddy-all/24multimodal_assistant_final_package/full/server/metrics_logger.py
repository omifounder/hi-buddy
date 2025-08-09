# full/server/metrics_logger.py - simple JSONL logger
import os, time, json
DEV_LOG = os.environ.get('METRICS_DEV_LOG', '/tmp/metrics_dev.jsonl')
def log_metric(name, value, tags=None):
    entry = {'ts': int(time.time()*1000), 'name': name, 'value': value, 'tags': tags or {}}
    try:
        with open(DEV_LOG, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass
