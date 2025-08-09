Monitoring Bundle - Prometheus + Grafana + Assistant (demo)
===========================================================

Files included:
- Dockerfile
- docker-compose.yml (assistant, prometheus, grafana)
- prometheus.yml (scrape config)
- alert_rules.yml (Prometheus alert rules)
- grafana-dashboard.json (starter dashboard)
- metrics_logger.py (emits Prometheus metrics + JSONL dev log)
- server.py (instrumented assistant entrypoint)
- requirements.txt

Quick start (Docker):
1. Build and start everything:
   docker compose up --build

2. Access services:
   - Assistant API: http://localhost:8000/ping
   - Prometheus UI: http://localhost:9090
   - Grafana UI: http://localhost:3000 (anonymous access enabled)

3. Test:
   - Call assistant ping: curl http://localhost:8000/ping
   - Prometheus should scrape assistant at assistant:8000/metrics

Notes:
- The assistant service is built from the local Dockerfile and runs server.py.
- Grafana provisioning via mounting a single JSON file is simplistic; for production use use proper provisioning config.
- Alert routing (Slack/PagerDuty) requires configuring Alertmanager (not included in this bundle).
