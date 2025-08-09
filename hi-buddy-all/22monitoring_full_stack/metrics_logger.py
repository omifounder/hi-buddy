# metrics_logger.py - lightweight metrics logger for demo assistant
import json, time, threading, os
from prometheus_client import CollectorRegistry, Gauge, Histogram, generate_latest

registry = CollectorRegistry()
_gauges = {}
_histograms = {}
_lock = threading.Lock()

DEV_LOG_PATH = os.environ.get('METRICS_DEV_LOG', 'metrics_dev.jsonl')

def _gauge_key(name, tags):
    return name + '|' + (','.join([f"{k}={v}" for k,v in (tags or {}).items()]))

def log_metric(name, value, tags=None):
    entry = {'name': name, 'value': value, 'tags': tags or {}, 'ts': int(time.time()*1000)}
    try:
        with open(DEV_LOG_PATH, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass
    try:
        key = _gauge_key(name, tags)
        with _lock:
            if name.endswith('_ms') or name.endswith('_seconds') or name.endswith('_latency'):
                if name not in _histograms:
                    _histograms[name] = Histogram(name, name, registry=registry)
                _histograms[name].observe(float(value)/1000.0 if name.endswith('_ms') else float(value))
            else:
                if key not in _gauges:
                    if tags:
                        g = Gauge(name, name, list(tags.keys()), registry=registry)
                        _gauges[key] = (g, list(tags.keys()))
                    else:
                        g = Gauge(name, name, registry=registry)
                        _gauges[key] = (g, [])
                g, label_names = _gauges[key]
                if label_names:
                    g.labels(*[tags[k] for k in label_names]).set(value)
                else:
                    g.set(value)
    except Exception:
        pass

def prometheus_metrics():
    try:
        return generate_latest(registry)
    except Exception:
        return b''
