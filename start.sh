#!/bin/bash

# STEACH Training Server — Start Script (Linux & macOS)

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Prefer a project venv if present, else system python3
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
if [ -f "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

# Read port from config.json (server.port), default 5001
PORT=$("$PYTHON_BIN" -c "import json; print(json.load(open('config.json')).get('server',{}).get('port',5001))" 2>/dev/null || echo 5001)

LOG="$SCRIPT_DIR/server.log"
PIDFILE="$SCRIPT_DIR/server.pid"

# Already running?
if pgrep -f "training_server\.py" > /dev/null 2>&1; then
    echo -e "${YELLOW}[WARNING]${NC} STEACH is already running"
    echo "Use ./restart.sh to restart"
    exit 1
fi

echo "Starting STEACH on port $PORT..."
nohup "$PYTHON_BIN" "$SCRIPT_DIR/training_server.py" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

sleep 2
if pgrep -f "training_server\.py" > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} STEACH started on http://localhost:$PORT"
    echo "View logs: tail -f $LOG"
else
    echo -e "${RED}[ERROR]${NC} STEACH failed to start. Check $LOG"
    exit 1
fi
