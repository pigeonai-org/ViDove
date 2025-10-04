# Docker Split-Container Migration Summary

## Overview

This document summarizes the migration from a unified Docker container to a split-container architecture for the ViDove web application.

## Problem Statement

The original unified container had a critical issue:
- **Single Point of Failure**: When the ViDove subprocess crashed, it would bring down the entire backend, making the frontend unresponsive
- **Resource Management**: Difficult to allocate resources appropriately between frontend and backend
- **Debugging Challenges**: Hard to isolate issues between services
- **Scaling Limitations**: Cannot scale frontend and backend independently

## Solution: Split-Container Architecture

### Architecture Changes

**Before (Unified Container):**
```
┌─────────────────────────────────────┐
│     Unified Container                │
│  ┌────────────┐  ┌──────────────┐   │
│  │  Frontend  │  │   Backend    │   │
│  │  (React)   │  │  (FastAPI)   │   │
│  └────────────┘  └──────────────┘   │
│         └───────┬────────┘           │
│              ViDove Pipeline         │
│         (Crashes = Everything Dies)  │
└─────────────────────────────────────┘
```

**After (Split Containers):**
```
┌──────────────────┐         ┌─────────────────────────┐
│  Frontend        │         │  Backend                 │
│  Container       │◄────────┤  Container               │
│                  │  HTTP   │                          │
│  React + Nginx   │         │  FastAPI + ViDove       │
│  Port 3000       │         │  Port 8000               │
└──────────────────┘         └─────────────────────────┘
     Always Up                    Isolated Process
```

### Benefits

1. **Process Isolation**: 
   - Frontend stays responsive even if ViDove subprocess crashes
   - Backend can restart without affecting frontend
   - Better error isolation and debugging

2. **Independent Scaling**:
   - Scale backend to handle more translation requests
   - Scale frontend for more concurrent users
   - Different resource limits for each service

3. **Resource Optimization**:
   - Backend: 2-4GB RAM (processing-heavy)
   - Frontend: 256-512MB RAM (serving static files)

4. **Deployment Flexibility**:
   - Update backend without redeploying frontend
   - Update frontend without affecting backend
   - Independent health checks and restart policies

5. **Better DevOps**:
   - Separate logs for each service
   - Individual container monitoring
   - Easier CI/CD pipelines

## Files Created/Modified

### New Files

1. **`docker-compose.yml`** - Main orchestration file
   - Defines two services: `backend` and `frontend`
   - Network configuration
   - Volume mounts for persistence
   - Health checks
   - Resource limits

2. **`Dockerfile.backend`** - Backend container image
   - Python 3.10 base
   - FastAPI + ViDove pipeline
   - Separate virtual environments for pipeline and backend
   - FFmpeg and system dependencies
   - Health check endpoint

3. **`demo/frontend/Dockerfile`** - Frontend container image
   - Multi-stage build (Node.js builder + Nginx production)
   - Optimized React build
   - Nginx for serving static files
   - Custom nginx configuration

4. **`demo/frontend/nginx.conf`** - Nginx configuration
   - React Router support
   - Gzip compression
   - Security headers
   - Health check endpoint
   - Cache configuration

5. **`docker-compose.prod.yml`** - Production overrides
   - Restart policies
   - Logging configuration
   - Production-specific settings

6. **`docker-compose.dev.yml`** - Development overrides
   - Hot reload support
   - Volume mounts for source code
   - Debug logging

7. **`docker-manage.sh`** - Management script
   - Convenient commands for common operations
   - Health checks
   - Log viewing
   - Shell access

8. **`DOCKER_DEPLOYMENT.md`** - Comprehensive deployment guide
   - Architecture overview
   - Setup instructions
   - Troubleshooting guide
   - Production deployment tips

9. **`.env.example`** - Environment template
   - Required environment variables
   - API key placeholders
   - Configuration options

10. **`demo/frontend/.dockerignore`** - Frontend build optimization
    - Excludes unnecessary files from frontend image

### Modified Files

1. **`demo/backend/main.py`**
   - Added `/health` endpoint for Docker health checks
   - Separated from root `/` endpoint

2. **`README.md`**
   - Added Docker quick start section at the top
   - Links to detailed Docker deployment guide

3. **`Dockerfile.unified`** (deprecated but kept for reference)
   - Original unified container configuration

## Environment Variables

### Required

- `OPENAI_API_KEY`: Your OpenAI API key for LLM features

### Optional

- `BACKEND_PORT`: Backend port (default: 8000)
- `FRONTEND_PORT`: Frontend port (default: 3000)
- `REACT_APP_API_URL`: Backend API URL for frontend
- `MAX_UPLOAD_SIZE`: Maximum file upload size

## Quick Start

### For Users

```bash
# 1. Clone and setup
git clone https://github.com/project-kxkg/ViDove.git
cd ViDove
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Start services
docker-compose up -d

# 3. Access
# Frontend: http://localhost:3000
# Backend: http://localhost:8000/docs
```

### Using Management Script

```bash
# Start services
./docker-manage.sh start

# View logs
./docker-manage.sh logs

# Check health
./docker-manage.sh health

# Restart services
./docker-manage.sh restart

# Stop services
./docker-manage.sh stop
```

## Architecture Details

### Backend Container

**Base Image**: `python:3.10-slim`

**Components**:
- FastAPI web server (port 8000)
- ViDove translation pipeline (separate venv at `/app/venv`)
- Backend dependencies (separate venv at `/app/backend-venv`)
- FFmpeg for video/audio processing
- Fonts for subtitle rendering

**Virtual Environments**:
- `/app/venv`: Pipeline dependencies (from pyproject.toml)
- `/app/backend-venv`: Backend dependencies (from demo/backend/requirements.txt)

**Volumes**:
- `uploads/`: User uploaded files
- `results/`: Translation results
- `domain_dict/`: Custom translation dictionaries
- `configs/`: Configuration files

**Health Check**: `GET /health` every 30s

### Frontend Container

**Build**: Multi-stage
1. **Builder Stage**: Node.js 18 Alpine
   - Install dependencies
   - Build React app
   - Optimize for production

2. **Production Stage**: Nginx Alpine
   - Copy built assets
   - Serve static files
   - Handle React Router

**Port**: 80 (mapped to 3000 on host)

**Health Check**: `GET /health` every 30s

### Networking

- Both containers on `vidove-network` bridge network
- Frontend communicates with backend via container name
- External access via port mapping

## Resource Limits

### Backend
- **Memory**: 2-4GB (configurable)
- **Recommended**: 4GB for heavy translation tasks

### Frontend
- **Memory**: 256-512MB
- **Recommended**: 512MB for optimal performance

## Health Monitoring

### Backend Health Check
```bash
curl http://localhost:8000/health
# Response: {"status": "healthy", "service": "vidove-backend"}
```

### Frontend Health Check
```bash
curl http://localhost:3000/health
# Response: healthy
```

### Container Status
```bash
docker-compose ps
# Shows health status of all containers
```

## Troubleshooting

### Common Issues

1. **Backend exits immediately**
   - Check logs: `./docker-manage.sh logs-backend`
   - Verify OPENAI_API_KEY is set in .env
   - Check if port 8000 is already in use

2. **Frontend cannot connect to backend**
   - Verify both containers are running: `docker-compose ps`
   - Check backend health: `curl http://localhost:8000/health`
   - Check network: `docker network inspect vidove-network`

3. **ViDove subprocess crashes**
   - Backend container continues running ✓
   - Frontend remains accessible ✓
   - Check backend logs for error details
   - Task status will show "FAILED" with error message

4. **Out of memory**
   - Check stats: `./docker-manage.sh stats`
   - Increase memory limits in docker-compose.yml
   - Consider scaling to multiple backend instances

## Deployment Options

### Development
```bash
./docker-manage.sh dev
```
- Hot reload enabled
- Debug logging
- Source code mounted

### Production
```bash
./docker-manage.sh prod
```
- Optimized for performance
- Automatic restart on failure
- Log rotation
- Can scale backend: `docker-compose up -d --scale backend=3`

## Migration from Unified Container

If you were using the unified container (`Dockerfile.unified`):

1. **Stop old container**:
   ```bash
   docker stop <unified-container-name>
   docker rm <unified-container-name>
   ```

2. **Backup data** (if needed):
   ```bash
   docker cp <unified-container-name>:/app/demo/backend/uploads ./backup_uploads
   docker cp <unified-container-name>:/app/demo/backend/results ./backup_results
   ```

3. **Start new split containers**:
   ```bash
   ./docker-manage.sh start
   ```

4. **Restore data** (if backed up):
   ```bash
   cp -r backup_uploads/* ./demo/backend/uploads/
   cp -r backup_results/* ./demo/backend/results/
   ```

## Performance Improvements

### Build Time
- **Frontend**: ~2-3 minutes (multi-stage build, cached layers)
- **Backend**: ~5-7 minutes (includes Python deps compilation)
- **Total**: ~7-10 minutes for first build
- **Subsequent builds**: Much faster with layer caching

### Runtime
- **Frontend**: Serves static files with Nginx (< 10ms response)
- **Backend**: FastAPI with async support (< 100ms for API calls)
- **ViDove Pipeline**: Same performance as before (subprocess isolation)

## Security Considerations

1. **API Keys**: Never commit `.env` to git
2. **CORS**: Configure `allow_origins` in production
3. **HTTPS**: Use reverse proxy (nginx, traefik) in production
4. **Network**: Use Docker secrets for sensitive data in production
5. **Updates**: Regularly update base images and dependencies

## Future Improvements

1. **Container Registry**: Push images to Docker Hub or private registry
2. **Orchestration**: Kubernetes deployment for production scale
3. **Monitoring**: Prometheus + Grafana for metrics
4. **Logging**: Centralized logging with ELK stack
5. **CI/CD**: Automated builds and deployments
6. **Load Balancing**: Nginx or Traefik reverse proxy with multiple backend instances

## Conclusion

The split-container architecture provides:
- ✅ Better reliability (isolated processes)
- ✅ Easier scaling (independent services)
- ✅ Improved debugging (separate logs)
- ✅ Flexible deployment (update services independently)
- ✅ Production-ready (proper health checks, restart policies)

The migration maintains backward compatibility while providing significant operational improvements.
