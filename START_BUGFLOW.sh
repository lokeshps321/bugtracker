#!/bin/bash

# BugFlow - Complete Startup Script
# This script starts both the backend and frontend servers

echo "🚀 Starting BugFlow System..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to start backend
start_backend() {
    echo -e "${BLUE}📦 Starting Backend Server...${NC}"
    cd "$SCRIPT_DIR"
    
    # Check if venv exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate venv and start server
    source venv/bin/activate
    echo -e "${GREEN}✓ Backend starting on http://localhost:8000${NC}"
    echo -e "${YELLOW}  API Docs: http://localhost:8000/docs${NC}"
    echo ""
    
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    echo "Backend PID: $BACKEND_PID"
}

# Function to start frontend
start_frontend() {
    echo -e "${BLUE}⚛️  Starting Frontend Server...${NC}"
    cd "$SCRIPT_DIR/frontend"
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "Installing dependencies..."
        npm install
    fi
    
    echo -e "${GREEN}✓ Frontend starting on http://localhost:3000${NC}"
    echo ""
    
    npm start &
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
}

# Main execution
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        BugFlow - Bug Management         ║${NC}"
echo -e "${GREEN}║     AI-Powered Bug Tracking System      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""

# Start both servers
start_backend
sleep 2
start_frontend

echo ""
echo -e "${GREEN}✓ Both servers started successfully!${NC}"
echo ""
echo -e "${BLUE}📍 Access Points:${NC}"
echo -e "   Frontend:     ${YELLOW}http://localhost:3000${NC}"
echo -e "   Backend API:  ${YELLOW}http://localhost:8000${NC}"
echo -e "   API Docs:     ${YELLOW}http://localhost:8000/docs${NC}"
echo ""
echo -e "${BLUE}🔐 Demo Credentials:${NC}"
echo -e "   PM:     ${YELLOW}pm1@example.com / password${NC}"
echo -e "   Tester: ${YELLOW}tester1@example.com / password${NC}"
echo -e "   Dev:    ${YELLOW}dev1@example.com / password${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
