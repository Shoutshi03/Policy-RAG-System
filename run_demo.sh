#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
streamlit run app.py \
  --server.address="${BIND_ADDRESS:-0.0.0.0}" \
  --server.port="${STREAMLIT_SERVER_PORT:-8501}"
