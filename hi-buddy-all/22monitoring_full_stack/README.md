Full Monitoring Bundle (Prometheus + Grafana + Alertmanager + Assistant)
======================================================================

Contents:
- assistant_server.py : demo assistant server (builds into assistant image)
- metrics_logger.py   : lightweight Prometheus logger used by assistant
- Dockerfile          : builds assistant image
- docker-compose.yml  : assistant + prometheus + grafana + alertmanager
- prometheus.yml      : prometheus config (scrapes assistant)
- alert_rules.yml     : prometheus alert rules
- alertmanager.yml    : alertmanager config (placeholders for Slack/PagerDuty/Email)
- provisioning/        : Grafana provisioning (datasource + dashboard provider)
- dashboards/assistant.json : dashboard JSON
- load_test.py        : synthetic load generator
- sample.wav          : short audio file used by load_test.py

Quick start:
1. Build and run everything:
   docker compose up --build

2. Access:
   - Assistant: http://localhost:8000/ping
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (anonymous enabled)
   - Alertmanager: http://localhost:9093

3. Run the load test (locally, after stack is up):
   python load_test.py

Notes:
- Replace placeholders in alertmanager.yml (Slack webhook, PagerDuty key, email/smtp settings) before enabling alerts.
- For production, secure Grafana and configure SMTP and alert routing appropriately.
