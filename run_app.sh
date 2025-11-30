#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Stopping BugFlow..."
    kill $(jobs -p) 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

echo "=================================================="
echo "🚀 Starting BugFlow System"
echo "=================================================="

# Kill any existing processes on ports 8000 and 8501
echo "🧹 Cleaning up existing processes..."
fuser -k 8000/tcp 2>/dev/null
fuser -k 8501/tcp 2>/dev/null
pkill -f "uvicorn.*app.main:app" 2>/dev/null
pkill -f "streamlit run frontend/app.py" 2>/dev/null
sleep 2

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "[1/2] Starting Backend API..."
uvicorn app.main:app --reload --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!

echo "Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check backend.log"
    cat backend.log | tail -20
    exit 1
fi

echo "[2/2] Starting Frontend UI..."
streamlit run frontend/app.py > frontend.log 2>&1 &
FRONTEND_PID=$!

echo "=================================================="
echo "✅ BugFlow is running!"
echo "   - Frontend: http://localhost:8501"
echo "   - Backend:  http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "=================================================="
echo "Press Ctrl+C to stop"

wait
