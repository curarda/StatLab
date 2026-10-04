#!/usr/bin/env bash
# StatLab başlatıcı — venv'i kullanır, tarayıcıda açar.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Sanal ortam kuruluyor..."
  python3.13 -m venv .venv 2>/dev/null || python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
  ./.venv/bin/pip install -r requirements.txt
fi

exec ./.venv/bin/streamlit run app.py
