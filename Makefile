# One-command entry points. Anything that calls Meta needs META_API_KEY in server/.env.
UV := uv --directory server
PY := $(UV) run python

LIVE := wss://earshot-yuansshi.fly.dev/ws

.PHONY: setup test serve probe probe-asr smoke deploy ios-build

setup:           ## install the server's Python deps
	$(UV) sync

test:            ## server unit tests (fakes only: no network, no keys)
	$(UV) run pytest -q

serve:           ## agent server + web client on http://localhost:8000 (0.0.0.0 so the iPhone can reach it)
	$(UV) run uvicorn earshot.app:app --reload --host 0.0.0.0 --port 8000

probe:           ## spike S1: time to first token for Muse Spark (and Muse Spark Open if FIREWORKS_API_KEY is set)
	$(PY) scripts/probe.py --n 20

probe-asr:       ## spike S1b: stream synthesized speech through Meta realtime ASR
	$(PY) scripts/probe.py --only asr

smoke:           ## speak "Hey Earshot, will it rain…" to a running server (URL=ws://localhost:8000/ws for local)
	$(PY) scripts/smoke.py --url $(or $(URL),$(LIVE))

deploy:          ## build with Docker Desktop and deploy to Fly.io as one always-on machine
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock fly deploy --local-only --ha=false

ios-build:       ## compile the iPhone app for the simulator (no signing); run on a phone from Xcode
	xcodebuild -project ios/Earshot.xcodeproj -target Earshot -sdk iphonesimulator CODE_SIGNING_ALLOWED=NO build -quiet
