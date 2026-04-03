#!/usr/bin/env bash
# run.sh — Start both backend (FastAPI) and frontend (Streamlit) with one command
set -e

VENV=".venv"
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"

# Activate venv
source "$VENV/bin/activate"

echo "🚀 Starting FastAPI backend..."
cd "$BACKEND_DIR"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Give the backend a moment to start
sleep 2

echo "🎨 Starting Streamlit frontend..."
cd "$FRONTEND_DIR"
streamlit run app.py --server.port 8501 &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Services running:"
echo "   Backend  → http://localhost:8000"
echo "   Frontend → http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services."

# Wait and clean up on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
