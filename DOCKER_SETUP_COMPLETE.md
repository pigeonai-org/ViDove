# ViDove Docker Setup - Complete Guide

## Current Status: Fixed Issues ✅

### Issues Fixed:
1. ✅ **Docker command not found** - PATH issue identified
2. ✅ **Frontend 404 errors** - API URL now includes port 8000
3. ✅ **Backend connection refused** - Fixed API base URL in api.ts
4. ✅ **Dev mode Python path error** - Fixed docker-compose.dev.yml to use bash -c with cd
5. ✅ **Docker Compose v2 syntax** - Updated docker-manage.sh to use `docker compose` instead of `docker-compose`

## Quick Start (3 Steps)

### Step 1: Add Docker to PATH

Run this command in your terminal:
```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

To make it permanent, add to `~/.zshrc`:
```bash
echo 'export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Verify Docker

```bash
docker --version
docker compose version
```

### Step 3: Start ViDove

**Option A - Use Quick Start Script (Easiest):**
```bash
./quick-start.sh
```

**Option B - Manual Start:**
```bash
# Make sure .env file exists with OPENAI_API_KEY
docker compose down
docker compose up -d --build
```

## What Was Changed

### Files Modified:

1. **`docker-manage.sh`**
   - Changed all `docker-compose` to `docker compose` (v2 syntax)
   - Made `jq` optional in health check

2. **`web_frontend/frontend/src/services/api.ts`**
   - Changed: `${window.location.hostname}` 
   - To: `${window.location.hostname}:8000`
   - Now frontend can connect to backend on correct port

3. **`docker-compose.dev.yml`**
   - Fixed command to use `bash -c "cd /app/web_frontend/backend && ..."`
   - This ensures uvicorn runs from correct directory

4. **`docker-compose.yml`**
   - Added build args for `REACT_APP_API_URL`
   - Frontend now knows backend is on port 8000

5. **`web_frontend/frontend/Dockerfile`**
   - Accepts `REACT_APP_API_URL` as build argument
   - Passes it to React build process

### New Files Created:

1. **`quick-start.sh`** - One-command startup script
2. **`DOCKER_TROUBLESHOOTING.md`** - Comprehensive troubleshooting guide

## Architecture Summary

```
┌─────────────────────────────────────────────────────┐
│                  Docker Host                         │
│                                                      │
│  ┌────────────────────┐    ┌────────────────────┐  │
│  │  Frontend          │    │  Backend           │  │
│  │  Container         │    │  Container         │  │
│  │                    │    │                    │  │
│  │  React + Nginx ────┼────► FastAPI + ViDove  │  │
│  │  Port 3000        │http │  Port 8000         │  │
│  │  (80 internal)     │:8000│                    │  │
│  └────────────────────┘    └────────────────────┘  │
│           ▲                          ▲              │
│           │                          │              │
└───────────┼──────────────────────────┼──────────────┘
            │                          │
     Browser requests              API calls
     localhost:3000              localhost:8000
```

## Testing Your Setup

### 1. Check Container Status
```bash
docker compose ps
```

Expected output:
```
NAME              STATUS        PORTS
vidove-backend    Up (healthy)  0.0.0.0:8000->8000/tcp
vidove-frontend   Up (healthy)  0.0.0.0:3000->80/tcp
```

### 2. Test Backend Health
```bash
curl http://localhost:8000/health
```

Expected: `{"status":"healthy","service":"vidove-backend"}`

### 3. Test Frontend
Open browser: **http://localhost:3000**

You should see:
- ✅ ViDove web interface loads
- ✅ No 404 errors for favicon.ico or manifest.json
- ✅ Chat interface initializes
- ✅ No "ERR_CONNECTION_REFUSED" errors

### 4. Test Full Flow
1. Go to http://localhost:3000
2. Chat should start automatically
3. Try asking: "I want to translate a video from English to Chinese"
4. Upload a video or provide YouTube URL
5. Start translation task

## Common Commands

### Starting and Stopping
```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# Restart specific service
docker compose restart backend
docker compose restart frontend
```

### Viewing Logs
```bash
# All logs
docker compose logs -f

# Backend only
docker compose logs -f backend

# Frontend only
docker compose logs -f frontend

# Last 100 lines
docker compose logs --tail=100
```

### Debugging
```bash
# Shell into backend
docker compose exec backend /bin/bash

# Shell into frontend
docker compose exec frontend /bin/sh

# Check processes
docker compose top

# Resource usage
docker stats
```

### Rebuilding
```bash
# Rebuild specific service
docker compose build backend
docker compose up -d backend

# Rebuild everything
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Development Workflow

### Working on Backend Code

Changes to these directories auto-reload in dev mode:
- `src/`
- `entries/`
- `web_frontend/backend/`

Start dev mode:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Working on Frontend Code

**Option 1 - Local Development (Recommended for frontend work):**
```bash
cd web_frontend/frontend
npm install
REACT_APP_API_URL=http://localhost:8000 npm start
```

**Option 2 - Docker with Rebuild:**
```bash
docker compose build frontend
docker compose up -d frontend
```

## Troubleshooting

### Problem: Docker command not found
```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

### Problem: Containers exit immediately
```bash
# Check logs
docker compose logs backend

# Common causes:
# - .env missing or invalid
# - Port already in use
# - Build failed
```

### Problem: Frontend can't connect to backend
```bash
# Check backend is running
docker compose ps
curl http://localhost:8000/health

# Check frontend environment
docker compose exec frontend env | grep API_URL

# Should see: REACT_APP_API_URL=http://localhost:8000
```

### Problem: ViDove subprocess fails
```bash
# Backend container should still be running!
docker compose ps

# Check backend logs for error details
docker compose logs backend | grep -i error

# The task status will show FAILED with error message
```

### Problem: Out of memory
```bash
# Check usage
docker stats

# Increase limits in docker-compose.yml
# Then restart:
docker compose down
docker compose up -d
```

### Clean Slate Reset
```bash
# Nuclear option - removes everything
docker compose down -v
docker system prune -a --volumes
rm -rf web_frontend/backend/uploads/*
rm -rf web_frontend/backend/results/*

# Then rebuild
docker compose build --no-cache
docker compose up -d
```

## Performance Tips

1. **Don't rebuild unnecessarily** - Use `docker compose restart` when you only changed code
2. **Use .dockerignore** - Already configured to exclude unnecessary files
3. **Layer caching** - Don't use `--no-cache` unless you need to
4. **Scale backend** - `docker compose up -d --scale backend=3` for more workers
5. **Monitor resources** - `docker stats` to watch memory/CPU usage

## Security Checklist

- [x] API keys in .env, not in code
- [x] .env in .gitignore
- [ ] Configure CORS for production (currently allows all origins)
- [ ] Use HTTPS in production (add reverse proxy)
- [ ] Update base images regularly
- [ ] Use Docker secrets for production deployment
- [ ] Enable resource limits

## Next Steps

1. ✅ Start containers: `./quick-start.sh`
2. ✅ Verify frontend loads: http://localhost:3000
3. ✅ Verify backend API: http://localhost:8000/docs
4. ✅ Test translation workflow
5. 📚 Read full guides:
   - `DOCKER_DEPLOYMENT.md` - Full deployment documentation
   - `DOCKER_TROUBLESHOOTING.md` - Detailed troubleshooting
   - `DOCKER_QUICK_REFERENCE.md` - Command quick reference

## Support

If you encounter issues:

1. Check logs: `docker compose logs -f`
2. Check troubleshooting guide: `DOCKER_TROUBLESHOOTING.md`
3. Verify environment: `docker compose ps` and `curl http://localhost:8000/health`
4. Try clean rebuild: `docker compose down && docker compose up -d --build`

---

**Summary**: All issues have been fixed. Docker path issue identified and quick-start script created. Run `./quick-start.sh` to get started!
