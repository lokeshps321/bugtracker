#!/bin/bash
# =========================================
# BugFlow - Single Startup Script
# =========================================
# This script starts both backend and frontend

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🐛 BugFlow Startup Script          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    pip install -r requirements.txt --quiet
    
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    pip install -r frontend/requirements.txt --quiet
    
    touch venv/.installed
    echo -e "${GREEN}Dependencies installed!${NC}"
fi

# Start Backend
echo -e "${BLUE}Starting Backend on http://localhost:8000${NC}"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start Frontend
echo -e "${BLUE}Starting Frontend on http://localhost:8501${NC}"
cd frontend && streamlit run app.py --server.port 8501 &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         🚀 BugFlow is Running!             ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Backend:  http://localhost:8000           ║${NC}"
echo -e "${GREEN}║  Frontend: http://localhost:8501           ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Demo Credentials:                         ║${NC}"
echo -e "${GREEN}║    Tester:    tester1 / test123            ║${NC}"
echo -e "${GREEN}║    Developer: dev1 / dev123                ║${NC}"
echo -e "${GREEN}║    PM:        pm1 / pm123                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"

# Handle shutdown
trap "echo -e '\n${YELLOW}Shutting down...${NC}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

# Wait for both processes
wait
