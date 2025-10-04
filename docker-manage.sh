#!/bin/bash
# ViDove Docker Management Script
# This script provides convenient commands for managing ViDove Docker deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_usage() {
    cat << EOF
ViDove Docker Management Script

Usage: ./docker-manage.sh [command]

Commands:
    start           Start all services (docker-compose up -d)
    stop            Stop all services (docker-compose down)
    restart         Restart all services
    logs            View logs from all services
    logs-backend    View backend logs only
    logs-frontend   View frontend logs only
    build           Rebuild all images
    rebuild         Rebuild and restart all services
    status          Show status of all containers
    clean           Stop services and remove volumes (WARNING: deletes data)
    shell-backend   Open shell in backend container
    shell-frontend  Open shell in frontend container
    dev             Start in development mode
    prod            Start in production mode
    health          Check health status of all services
    stats           Show resource usage statistics

Examples:
    ./docker-manage.sh start
    ./docker-manage.sh logs-backend
    ./docker-manage.sh rebuild

EOF
}

check_env() {
    if [ ! -f .env ]; then
        echo -e "${YELLOW}Warning: .env file not found${NC}"
        echo "Creating .env from .env.example..."
        if [ -f .env.example ]; then
            cp .env.example .env
            echo -e "${GREEN}✓ Created .env file${NC}"
            echo -e "${YELLOW}Please edit .env and add your API keys before starting${NC}"
            exit 1
        else
            echo -e "${RED}Error: .env.example not found${NC}"
            exit 1
        fi
    fi
    
    if ! grep -q "OPENAI_API_KEY=.*[a-zA-Z0-9]" .env; then
        echo -e "${YELLOW}Warning: OPENAI_API_KEY not set in .env file${NC}"
        echo "Please add your OpenAI API key to .env before starting"
    fi
}

start_services() {
    echo -e "${GREEN}Starting ViDove services...${NC}"
    check_env
    docker compose up -d
    echo -e "${GREEN}✓ Services started${NC}"
    echo -e "Frontend: http://localhost:3000"
    echo -e "Backend API: http://localhost:8000"
    echo -e "API Docs: http://localhost:8000/docs"
}

stop_services() {
    echo -e "${YELLOW}Stopping ViDove services...${NC}"
    docker compose down
    echo -e "${GREEN}✓ Services stopped${NC}"
}

restart_services() {
    echo -e "${YELLOW}Restarting ViDove services...${NC}"
    docker compose restart
    echo -e "${GREEN}✓ Services restarted${NC}"
}

view_logs() {
    docker compose logs -f "$@"
}

build_images() {
    echo -e "${GREEN}Building Docker images...${NC}"
    docker compose build --no-cache
    echo -e "${GREEN}✓ Images built${NC}"
}

rebuild_services() {
    echo -e "${GREEN}Rebuilding and restarting services...${NC}"
    check_env
    docker compose up -d --build
    echo -e "${GREEN}✓ Services rebuilt and started${NC}"
}

show_status() {
    echo -e "${GREEN}Container Status:${NC}"
    docker compose ps
}

clean_all() {
    echo -e "${RED}WARNING: This will stop services and delete all volumes (uploads, results, etc.)${NC}"
    read -p "Are you sure? (yes/no): " -r
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo -e "${YELLOW}Cleaning up...${NC}"
        docker compose down -v
        echo -e "${GREEN}✓ Cleaned up${NC}"
    else
        echo "Cancelled"
    fi
}

shell_backend() {
    echo -e "${GREEN}Opening shell in backend container...${NC}"
    docker compose exec backend /bin/bash
}

shell_frontend() {
    echo -e "${GREEN}Opening shell in frontend container...${NC}"
    docker compose exec frontend /bin/sh
}

start_dev() {
    echo -e "${GREEN}Starting in development mode...${NC}"
    check_env
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up
}

start_prod() {
    echo -e "${GREEN}Starting in production mode...${NC}"
    check_env
    docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    echo -e "${GREEN}✓ Production services started${NC}"
}

check_health() {
    echo -e "${GREEN}Checking health status...${NC}"
    echo ""
    echo "Backend health:"
    curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health || echo "Backend not responding"
    echo ""
    echo "Frontend health:"
    curl -s http://localhost:3000/health || echo "Frontend not responding"
    echo ""
    echo -e "${GREEN}Container status:${NC}"
    docker compose ps
}

show_stats() {
    echo -e "${GREEN}Resource usage statistics:${NC}"
    docker stats --no-stream $(docker compose ps -q)
}

# Main command dispatcher
case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        view_logs
        ;;
    logs-backend)
        view_logs backend
        ;;
    logs-frontend)
        view_logs frontend
        ;;
    build)
        build_images
        ;;
    rebuild)
        rebuild_services
        ;;
    status)
        show_status
        ;;
    clean)
        clean_all
        ;;
    shell-backend)
        shell_backend
        ;;
    shell-frontend)
        shell_frontend
        ;;
    dev)
        start_dev
        ;;
    prod)
        start_prod
        ;;
    health)
        check_health
        ;;
    stats)
        show_stats
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '${1:-}'${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
