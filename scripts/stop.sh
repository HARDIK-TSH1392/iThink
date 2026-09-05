#!/usr/bin/env bash
# Stops everything scripts/start.sh started: iThink backend, voice-agent
# (backend + frontend), the demo dashboard, and both cloudflared tunnels
# if they were running. Safe to run even if nothing (or only some
# services) is up -- falls back to pattern-matching in case a PID file
# is missing or stale.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$ROOT_DIR/.dev"

stop_one() {
  local name="$1"
  local pidfile="$STATE_DIR/$name.pid"
  if [ ! -f "$pidfile" ]; then
    echo "  $name: no pid file, skipping"
    return 0
  fi
  local pid
  pid=$(cat "$pidfile")
  if kill -0 "$pid" 2>/dev/null; then
    echo "  stopping $name (pid $pid, whole process group)"
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  else
    echo "  $name: not running (stale pid $pid)"
  fi
  rm -f "$pidfile"
}

echo "Stopping tracked services..."
for svc in tunnel-backend tunnel-frontend backend voiceagent voiceagent-backend voiceagent-frontend dashboard; do
  stop_one "$svc"
done

sleep 2

# Fallback in case a PID file was missing/stale, or a child process
# escaped its process group -- matches the exact command lines these
# services run as, so this is a no-op if nothing matches.
echo "Sweeping for anything left over..."
pkill -f "uvicorn app.main:app.*--port 8123" 2>/dev/null || true
pkill -f "concurrently -n backend,frontend" 2>/dev/null || true
pkill -f "python src/server.py" 2>/dev/null || true
pkill -f "next dev --webpack" 2>/dev/null || true
pkill -f "next start" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
pkill -f "http.server 8080" 2>/dev/null || true
pkill -f "cloudflared tunnel --url http://localhost:8123" 2>/dev/null || true
pkill -f "cloudflared tunnel --url http://localhost:3000" 2>/dev/null || true

sleep 1

echo ""
echo "Port check:"
all_down=true
for p in 3000 8000 8080 8123; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://localhost:$p/" 2>/dev/null)
  if [ "$code" = "000" ]; then
    echo "  $p: down"
  else
    echo "  $p: STILL UP (http $code) -- check manually, e.g. lsof -i :$p"
    all_down=false
  fi
done

rm -rf "$STATE_DIR"

echo ""
if [ "$all_down" = true ]; then
  echo "All services stopped."
else
  echo "Some ports are still responding -- see above."
fi
