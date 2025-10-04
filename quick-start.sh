#!/bin/bash
# Quick Docker Path Fix and Start Script for ViDove

echo "===================================="
echo "ViDove Docker Quick Fix & Start"
echo "===================================="
echo ""

# Add Docker to PATH for this session
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

# Verify Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker still not found after adding to PATH"
    echo ""
    echo "Please ensure:"
    echo "1. Docker Desktop is installed"
    echo "2. Docker Desktop is running (check menu bar)"
    echo ""
    echo "Then run:"
    echo "  export PATH=\"/Applications/Docker.app/Contents/Resources/bin:\$PATH\""
    exit 1
fi

echo "✅ Docker found: $(docker --version)"
echo "✅ Docker Compose: $(docker compose version)"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file"
        echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
        echo ""
        read -p "Press Enter after you've added your API key to .env..."
    else
        echo "❌ Error: .env.example not found"
        exit 1
    fi
fi

# Check if API key is set
if ! grep -q "OPENAI_API_KEY=.*[a-zA-Z0-9]" .env; then
    echo "⚠️  OPENAI_API_KEY not set in .env"
    echo "Please edit .env and add your OpenAI API key"
    exit 1
fi

echo "✅ Environment configured"
echo ""

# Stop any existing containers
echo "Stopping existing containers..."
docker compose down 2>/dev/null || true
echo ""

# Build and start
echo "Building and starting ViDove..."
echo "This may take 5-10 minutes on first build..."
echo ""

docker compose up -d --build

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================="
    echo "✅ ViDove Started Successfully!"
    echo "===================================="
    echo ""
    echo "Access Points:"
    echo "  Frontend:  http://localhost:3000"
    echo "  Backend:   http://localhost:8000"
    echo "  API Docs:  http://localhost:8000/docs"
    echo ""
    echo "Useful Commands:"
    echo "  View logs:        docker compose logs -f"
    echo "  View status:      docker compose ps"
    echo "  Stop services:    docker compose down"
    echo "  Restart:          docker compose restart"
    echo ""
    echo "Checking container status..."
    sleep 5
    docker compose ps
    echo ""
    echo "To view logs, run: docker compose logs -f"
else
    echo ""
    echo "❌ Error starting ViDove"
    echo ""
    echo "Check logs with: docker compose logs"
    exit 1
fi
