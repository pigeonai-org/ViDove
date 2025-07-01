#!/bin/bash

# ViDove Unified Docker Build and Deploy Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Building ViDove Unified Container${NC}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Copying from env.example...${NC}"
    if [ -f "env.example" ]; then
        cp env.example .env
        echo -e "${RED}❗ Please edit .env file with your actual API keys before running the container${NC}"
    else
        echo -e "${RED}❌ No env.example file found. Please create .env file manually${NC}"
        exit 1
    fi
fi

# Build the Docker image
echo -e "${GREEN}🔨 Building Docker image with isolated environments...${NC}"
echo -e "${YELLOW}📦 Using UV for fast Python package management${NC}"
echo -e "${YELLOW}🎯 Pipeline env: pipeline-venv (pyproject.toml + uv.lock)${NC}"
echo -e "${YELLOW}🌐 Backend env: backend-venv (backend/requirements.txt)${NC}"
echo -e "${YELLOW}⚛️  Frontend env: npm start (live React dev server)${NC}"
docker build -f Dockerfile.unified -t vidove:latest .

echo -e "${GREEN}✅ Build completed successfully!${NC}"
echo -e "${GREEN}🎉 To run the container:${NC}"
echo -e "${YELLOW}docker run -d --name vidove-app --env-file .env -p 3000:3000 -p 8000:8000 -v \$(pwd)/local_dump:/app/local_dump vidove:latest${NC}"
echo ""
echo -e "${GREEN}📝 Access points:${NC}"
echo -e "🌐 Web Interface: http://localhost:3000"
echo -e "🔧 API Backend: http://localhost:8000"
echo -e "📖 API Docs: http://localhost:8000/docs"
echo ""
echo -e "${GREEN}📂 Persistent data will be stored in ./local_dump${NC}"
