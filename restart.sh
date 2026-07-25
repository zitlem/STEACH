#!/bin/bash

# STEACH Training Server — Restart Script (Linux & macOS)
# Stops any running training_server.py, frees the port, then starts fresh.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
OS=$(uname -s)

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
if [ -f "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

PORT=$("$PYTHON_BIN" -c "import json; print(json.load(open('config.json')).get('server',{}).get('port',5001))" 2>/dev/null || echo 5001)

echo "Stopping STEACH..."
pkill -TERM -f "training_server\.py" 2>/dev/null
sleep 1
pkill -9 -f "training_server\.py" 2>/dev/null

# Free the port in case a process is still holding it
if [ "$OS" = "Linux" ]; then
    command -v fuser > /dev/null 2>&1 && fuser -k "$PORT/tcp" 2>/dev/null
else
    lsof -ti :"$PORT" 2>/dev/null | xargs kill -9 2>/dev/null
fi

# Wait for a clean shutdown
RETRIES=0
while pgrep -f "training_server\.py" > /dev/null 2>&1; do
    echo "Waiting for STEACH to stop..."
    pkill -9 -f "training_server\.py" 2>/dev/null
    sleep 1
    RETRIES=$((RETRIES + 1))
    if [ "$RETRIES" -ge 10 ]; then
        echo -e "${RED}[ERROR]${NC} Could not stop STEACH after 10 attempts"
        break
    fi
done
echo "STEACH stopped"
sleep 1

# Start fresh (start.sh does the launch + verification)
exec "$SCRIPT_DIR/start.sh"
