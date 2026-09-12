#!/bin/bash
# Fires the same synthetic log dashboard.html's "Simulate Incident" button
# sends (see simulateIncident() there), from the terminal instead of a
# browser click -- for a pre-demo dry run or as the actual live trigger.
#
# service MUST stay "auth-api": that's the only entry in
# SERVICE_TO_GITHUB_REPO (backend/app/iCall/iCall_utils.py) and it's what
# points Watcher's GitHub MCP check_recent_deploys lookup at
# HARDIK-TSH1392/stark-payment-global -- the repo seeded with the DB
# connection-pool rollout commit. The message below is written to match
# that root cause (payment charge timeouts from an exhausted DB pool) so
# the story is consistent end to end, even though the service name itself
# is a leftover from how the demo mapping was originally set up.
#
# Usage: ./trigger_incident.sh [API_BASE]
#   API_BASE defaults to http://127.0.0.1:8123/api/v1 (same default as
#   demo/dashboard.html) -- pass a different one if the backend is behind
#   a tunnel.

set -euo pipefail

API_BASE="${1:-http://127.0.0.1:8123/api/v1}"
SOURCE_ID="demo-source-$(date +%s)-$$"
TIMESTAMP="$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat())' 2>/dev/null \
  || date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "POST ${API_BASE}/ilogs/ingest"
echo "source_id: ${SOURCE_ID}"
echo

RESPONSE="$(curl -sS -X POST "${API_BASE}/ilogs/ingest" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "source_id": "${SOURCE_ID}",
  "source_type": "k8s_deployment",
  "region": "us-east",
  "service": "auth-api",
  "environment": "production",
  "severity": "prod_down",
  "timestamp": "${TIMESTAMP}",
  "message": "auth-api payment charge calls timing out -- DB connection pool exhausted, 5xx spike across all pods"
}
JSON
)"

echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
echo

echo "Waiting for iTriage's background task to create the incident..."
sleep 3

echo
echo "Matching incident (by source_id):"
curl -sS "${API_BASE}/incidents/" \
  | jq --arg sid "$SOURCE_ID" --arg base "$API_BASE" \
    '[.[] | select(.source_id == $sid)][0] // ("not found yet -- check again in a few seconds, or GET " + $base + "/incidents/ yourself")'
