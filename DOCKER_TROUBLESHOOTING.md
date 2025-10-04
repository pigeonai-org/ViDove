# Docker Troubleshooting Guide for ViDove

## Issue: Docker command not found

### Problem
Your terminal can't find the `docker` command even though Docker Desktop is installed at `/Applications/Docker.app`.

### Solution

#### Option 1: Add Docker to PATH (Recommended)

Add this line to your `~/.zshrc` file:

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

Then reload your shell:
```bash
source ~/.zshrc
```

#### Option 2: Use Full Path Temporarily

Instead of using `docker`, use the full path:
```bash
/Applications/Docker.app/Contents/Resources/bin/docker compose up -d
```

#### Option 3: Create an alias

Add to your `~/.zshrc`:
```bash
alias docker="/Applications/Docker.app/Contents/Resources/bin/docker"
```

### Quick Fix Command

Run this now to fix for current terminal session:
```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

Then verify:
```bash
docker --version
docker compose version
```

## Starting ViDove After PATH Fix

Once Docker is in your PATH:

### For Production (Recommended)
```bash
# Stop any existing containers
docker compose down

# Rebuild and start
docker compose up -d --build

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### For Development
```bash
# Stop any existing containers
docker compose down

# Start in dev mode with hot reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## Common Errors and Fixes

### Error: "no such file or directory: /app/backend-venv/bin/python"
**Cause**: Dev mode command override wasn't using correct working directory

**Fix**: Already fixed in docker-compose.dev.yml - now uses `bash -c "cd ..."` to change directory first

### Error: Frontend 404 errors (favicon.ico, manifest.json)
**Cause**: 
1. React app not built
2. Frontend can't connect to backend

**Fixes Applied**:
1. Dockerfile builds React app correctly
2. API URL now includes port 8000: `http://localhost:8000`
3. Build args pass API URL to React build

### Error: "ERR_CONNECTION_REFUSED" for /api/chat/start
**Cause**: Frontend trying to connect to backend without port number

**Fix**: Updated `api.ts` to use `:8000` by default

## Verification Steps

After starting containers, verify everything works:

### 1. Check containers are running
```bash
docker compose ps
```

Expected output:
```
NAME              STATUS        PORTS
vidove-backend    Up (healthy)  0.0.0.0:8000->8000/tcp
vidove-frontend   Up (healthy)  0.0.0.0:3000->80/tcp
```

### 2. Test backend health
```bash
curl http://localhost:8000/health
```

Expected: `{"status":"healthy","service":"vidove-backend"}`

### 3. Test frontend
Open browser: http://localhost:3000

Should see ViDove interface without 404 errors

### 4. Test backend API docs
Open browser: http://localhost:8000/docs

Should see FastAPI Swagger UI

## Clean Slate Rebuild

If things are really broken, start fresh:

```bash
# Stop everything
docker compose down -v

# Remove images
docker rmi vidove-backend vidove-frontend

# Rebuild from scratch
docker compose build --no-cache

# Start
docker compose up -d

# Watch logs
docker compose logs -f
```

## Development Workflow

### Working on Backend Code
```bash
# Changes to Python files in src/ or web_frontend/backend/ 
# will auto-reload with --reload flag in dev mode
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Working on Frontend Code
Option 1 - Run frontend locally:
```bash
cd web_frontend/frontend
npm install
npm start  # Runs on port 3000 with hot reload
```

Option 2 - Rebuild frontend container:
```bash
docker compose build frontend
docker compose up -d frontend
```

## Logs and Debugging

### View all logs
```bash
docker compose logs -f
```

### Backend logs only
```bash
docker compose logs -f backend
```

### Frontend logs only
```bash
docker compose logs -f frontend
```

### Access backend shell
```bash
docker compose exec backend /bin/bash

# Inside container
ls -la /app/
ls -la /app/backend-venv/bin/
cd /app/web_frontend/backend
/app/backend-venv/bin/python --version
```

### Access frontend shell
```bash
docker compose exec frontend /bin/sh

# Inside container
ls -la /usr/share/nginx/html/
cat /etc/nginx/conf.d/default.conf
```

## Environment Setup Checklist

- [ ] Docker Desktop is installed
- [ ] Docker is running (check menu bar icon)
- [ ] Docker is in PATH
- [ ] `.env` file exists with `OPENAI_API_KEY`
- [ ] No other services using ports 3000 or 8000

## Port Conflicts

If ports 3000 or 8000 are in use:

### Find what's using the port
```bash
lsof -i :3000
lsof -i :8000
```

### Change ports in docker-compose.yml
```yaml
services:
  backend:
    ports:
      - "8001:8000"  # External:Internal
  frontend:
    ports:
      - "3001:80"
```

Don't forget to update frontend environment:
```yaml
  frontend:
    environment:
      - REACT_APP_API_URL=http://localhost:8001
```

## Performance Tips

### Speed up builds
- Don't use `--no-cache` unless necessary
- Keep .dockerignore up to date
- Use layer caching effectively

### Monitor resource usage
```bash
docker stats
```

### Limit resources if needed
Edit docker-compose.yml:
```yaml
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
```

## Getting Help

1. Check logs first: `docker compose logs -f`
2. Check container status: `docker compose ps`
3. Check health: `curl http://localhost:8000/health`
4. Check if backend is accessible: `docker compose exec backend /app/backend-venv/bin/python --version`
5. Review this guide: `DOCKER_TROUBLESHOOTING.md`
6. Full guide: `DOCKER_DEPLOYMENT.md`
