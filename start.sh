#!/bin/bash
# ─────────────────────────────────────────────
#  Anthropic Proxy v4.5 — Start Script
#  Run: bash start.sh
# ─────────────────────────────────────────────

cd "$(dirname "$0")"

# Activate virtual environment
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate 2>/dev/null

if [ $? -ne 0 ]; then
  echo ""
  echo "  ✗ venv not found. Creating one..."
  python -m venv venv
  source venv/Scripts/activate 2>/dev/null || source venv/bin/activate
  pip install fastapi uvicorn httpx --quiet
  echo "  ✓ Dependencies installed"
fi

python anthropic_proxy.py


