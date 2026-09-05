#!/usr/bin/env bash
# Starts the full local dev stack: iThink backend (8123), voice-agent
# (its own backend on 8000 + Next frontend on 3000), and the demo
# dashboard (8080). Pass --tunnels to also stand up cloudflared quick
# tunnels for the backend and frontend and wire the resulting URLs into
# every place that needs them (backend/.env, voice-agent/server/.env,
# demo/dashboard.html, voice-agent/web/next.config.ts).
#
# Pass --prod to run the frontend as a production build (`next build` +
# `next start`) instead of `next dev`. Dev mode's HMR/Fast Refresh/file
# watching is a real, constant CPU cost -- confirmed live as a likely
# contributor to SEND_AUDIO_BITRATE_TOO_LOW when the call is joined from
# the same machine running the dev servers (the browser's real-time audio
# encoder loses CPU time to it). --prod has none of that, and is closer
# to what a judge would actually see. Trade-off: no hot reload, so a code
# change needs `scripts/stop.sh` + a fresh `--prod` start to take effect.
#
# Usage:
#   scripts/start.sh                # local only, dev mode
#   scripts/start.sh --tunnels      # also expose backend+frontend publicly
#   scripts/start.sh --prod         # frontend as a production build
#   scripts/start.sh --tunnels --prod
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$ROOT_DIR/.dev"
mkdir -p "$STATE_DIR"

USE_TUNNELS=false
USE_PROD=false
for arg in "$@"; do
  case "$arg" in
    --tunnels) USE_TUNNELS=true ;;
    --prod) USE_PROD=true ;;
  esac
done

start_bg() {
  # $1 = service name (used for .pid/.log files), rest = command to run.
  # setsid gives it its own process group so stop.sh can kill the whole
  # tree (e.g. `bun run dev`'s concurrently -> sh -> node/python children)
  # in one shot instead of leaving orphaned children behind.
  local name="$1"; shift
  local pidfile="$STATE_DIR/$name.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "  $name already running (pid $(cat "$pidfile")), skipping"
    return 0
  fi
  setsid "$@" >"$STATE_DIR/$name.log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "  started $name (pid $!, log: $STATE_DIR/$name.log)"
}

wait_for_tunnel_url() {
  local logfile="$1" timeout="${2:-25}" elapsed=0 url=""
  while [ "$elapsed" -lt "$timeout" ]; do
    url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$logfile" 2>/dev/null | head -1)
    if [ -n "$url" ]; then printf '%s' "$url"; return 0; fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  return 1
}

if [ "$USE_TUNNELS" = true ]; then
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "cloudflared not found on PATH -- install it or drop --tunnels." >&2
    exit 1
  fi

  echo "Starting cloudflared tunnels..."
  start_bg tunnel-backend cloudflared tunnel --url http://localhost:8123
  start_bg tunnel-frontend cloudflared tunnel --url http://localhost:3000

  BACKEND_URL=$(wait_for_tunnel_url "$STATE_DIR/tunnel-backend.log") \
    || { echo "Timed out waiting for the backend tunnel URL -- see $STATE_DIR/tunnel-backend.log" >&2; exit 1; }
  FRONTEND_URL=$(wait_for_tunnel_url "$STATE_DIR/tunnel-frontend.log") \
    || { echo "Timed out waiting for the frontend tunnel URL -- see $STATE_DIR/tunnel-frontend.log" >&2; exit 1; }

  echo "  backend tunnel:  $BACKEND_URL"
  echo "  frontend tunnel: $FRONTEND_URL"

  FRONTEND_HOST="${FRONTEND_URL#https://}"
  LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  [ -z "$LAN_IP" ] && LAN_IP="192.168.1.8"

  echo "Wiring the new URLs into config files..."
  sed -i "s#^VOICE_AGENT_WEB_BASE_URL=.*#VOICE_AGENT_WEB_BASE_URL=$FRONTEND_URL#" "$ROOT_DIR/backend/.env"
  sed -i "s#^ITHINK_BACKEND_BASE_URL=.*#ITHINK_BACKEND_BASE_URL=$BACKEND_URL/api/v1#" "$ROOT_DIR/voice-agent/server/.env"
  sed -i "s#const WEB_CLIENT_BASE = \".*\";#const WEB_CLIENT_BASE = \"$FRONTEND_URL\";#" "$ROOT_DIR/demo/dashboard.html"
  sed -i "s#allowedDevOrigins: \[.*\],#allowedDevOrigins: ['$LAN_IP', '$FRONTEND_HOST'],#" "$ROOT_DIR/voice-agent/web/next.config.ts"
  echo "  done (backend/.env, voice-agent/server/.env, demo/dashboard.html, voice-agent/web/next.config.ts)"
fi

echo "Starting iThink backend (port 8123)..."
(cd "$ROOT_DIR/backend" && start_bg backend .venv/bin/python -m uvicorn app.main:app --reload --port 8123)

if [ "$USE_PROD" = true ]; then
  echo "Building voice-agent frontend for production..."
  # next.config.ts's rewrites() is evaluated at BUILD time and baked into
  # .next/routes-manifest.json -- next start serves from that manifest as-is,
  # it does not re-run rewrites() itself. Confirmed live: building without
  # these set produces an empty rewrites array, silently 404ing every
  # /api/* route (get_config, startAgent, recordUtterance, chatNotes, ...)
  # even though next start's own env looked correct.
  if ! (cd "$ROOT_DIR/voice-agent/web" && AGENT_BACKEND_URL=http://localhost:8000 ITHINK_BACKEND_URL=http://127.0.0.1:8123/api/v1 bun run build); then
    echo "Production build failed -- see output above. Not starting the frontend." >&2
    exit 1
  fi

  echo "Starting voice-agent (backend :8000 + frontend :3000, production build)..."
  (cd "$ROOT_DIR/voice-agent/server" && start_bg voiceagent-backend bash -c \
    "(venv/bin/python -m pip --version >/dev/null 2>&1 || (rm -rf venv && python3 -m venv venv)) && source venv/bin/activate && python -m pip install -q -r requirements.txt && python src/server.py")
  (cd "$ROOT_DIR/voice-agent/web" && AGENT_BACKEND_URL=http://localhost:8000 ITHINK_BACKEND_URL=http://127.0.0.1:8123/api/v1 \
    start_bg voiceagent-frontend bun run start)
else
  echo "Starting voice-agent (backend :8000 + frontend :3000)..."
  (cd "$ROOT_DIR/voice-agent" && start_bg voiceagent bun run dev)
fi

echo "Starting demo dashboard (port 8080)..."
(cd "$ROOT_DIR/demo" && start_bg dashboard python3 -m http.server 8080)

echo ""
check() {
  # Retries because Next's first compile can take well past a fixed wait --
  # curl's -w already prints "000" on connection failure by itself, so no
  # extra `|| echo "000"` fallback here (that doubled up to "000000").
  local name="$1" url="$2" tries=15 code
  for _ in $(seq 1 "$tries"); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$url" 2>/dev/null)
    [ "$code" != "000" ] && break
    sleep 2
  done
  printf "  %-10s %-40s %s\n" "$name" "$url" "$code"
}

echo "Status (retries up to ~30s for a slow first compile):"
check "backend"   "http://localhost:8123/openapi.json"
check "voice-be"  "http://localhost:8000/get_config"
check "frontend"  "http://localhost:3000/"
check "dashboard" "http://localhost:8080/dashboard.html"

echo ""
if [ "$USE_TUNNELS" = true ]; then
  echo "Join link:  $FRONTEND_URL/?channel=<your-channel>"
  echo "Dashboard:  http://localhost:8080/dashboard.html (still local-only)"
else
  echo "App:        http://localhost:3000/?channel=<your-channel>"
  echo "Dashboard:  http://localhost:8080/dashboard.html"
fi
echo "Logs:       $STATE_DIR/*.log"
echo "Stop with:  scripts/stop.sh"
