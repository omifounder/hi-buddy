# Detailed Test Plan - Multimodal Assistant

## Overview
This test plan includes unit, integration, E2E, performance, UX, and failure recovery tests for both MVP and Full.

## Environments
- Local dev: single machine (no TURN)
- Staging: k8s cluster with GPU nodes for model inference
- Mobile test devices: iPhone (A14+), Android mid-range (Snapdragon 7xx+)

## Unit Tests
- U01: models_loader.analyze_ser returns valid dict with confidence in [0,1]
- U02: synthesize_tts_to_wav creates a readable WAV file
- U03: metrics_logger.log_metric writes JSONL

## Integration Tests
- I01: POST /infer_wav returns 200 and result
- I02: POST /generate_reply returns reply text
- I03: POST /synthesize_tts then GET /tts/<name> returns file

## E2E Tests
- E01: 10-turn conversation: simulate a client sending audio to server, server runs SER, LLM, TTS sequence
- E02: WebRTC offer/answer handshake (requires aiortc) - client receives answer

## Performance & Load
- P01: Run 50 concurrent /infer_wav requests using locust or custom script; measure p95 latency, CPU, mem
- P02: WebRTC stress with 20 concurrent clients streaming audio + video

## Failure & Recovery
- FR01: Disable local model (simulate file missing) -> expect cloud fallback or stub reply and error logged
- FR02: Network interruption mid-call -> client should reconnect and resume stream

## Acceptance Criteria
- All unit tests passing in CI
- Integration tests pass in staging
- p95 inference latency < thresholds: MVP ser 2s, Full ser 1s
- System auto-recovers or degrades gracefully on faults
