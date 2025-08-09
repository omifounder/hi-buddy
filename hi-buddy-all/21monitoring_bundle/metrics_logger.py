# metrics_logger.py
import json, time, threading, os
from prometheus_client import CollectorRegistry, Gauge, Histogram, push_to_gateway, generate_latest

# Registry used for the process; expose generate_latest via metrics endpoint
registry = CollectorRegistry()
_gauges = {}
_histograms = {}
_lock = threading.Lock()

DEV_LOG_PATH = os.environ.get('METRICS_DEV_LOG', 'metrics_dev.jsonl')
PUSHGATEWAY = os.environ.get('PROMETHEUS_PUSHGATEWAY')  # e.g., http://pushgateway:9091

def _gauge_key(name, tags):
    return name + '|' + (','.join([f"{k}={v}" for k,v in (tags or {}).items()]))

def log_metric(name, value, tags=None):
    entry = {
        'name': name,
        'value': value,
        'tags': tags or {},
        'ts': int(time.time()*1000)
    }
    # write local dev log (append)
    try:
        with open(DEV_LOG_PATH, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass

    # update/prometheus gauge or histogram based on metric naming convention
    try:
        key = _gauge_key(name, tags)
        with _lock:
            if name.endswith('_seconds') or name.endswith('_latency_ms') or name.endswith('_infer_time_ms'):
                # use histogram for latency-like metrics; convert ms -> seconds if suffix _ms
                if name not in _histograms:
                    _histograms[name] = Histogram(name, name, registry=registry)
                hist = _histograms[name]
                # convert ms to seconds when applicable
                val = value
                if name.endswith('_ms'):
                    val = float(value) / 1000.0
                hist.observe(val)
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

    # optional push to pushgateway for production
    try:
        if PUSHGATEWAY:
            push_to_gateway(PUSHGATEWAY, job=os.environ.get('PROMETHEUS_JOB','ai_assistant'), registry=registry)
    except Exception:
        pass

def prometheus_metrics():
    """Return Prometheus metrics exposition bytes."""
    try:
        return generate_latest(registry)
    except Exception:
        return b""
