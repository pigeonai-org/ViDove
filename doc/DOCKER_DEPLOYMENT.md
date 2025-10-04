# ViDove Docker Deployment Guide

This guide explains how to run ViDove with Docker using the new split-container architecture.

## Architecture Overview

The application is now split into two containers:

1. **Backend Container** (`vidove-backend`): 
   - FastAPI web server
   - ViDove translation pipeline
   - Handles all processing and API requests
   - Runs on port 8000

2. **Frontend Container** (`vidove-frontend`):
   - React application served by Nginx
   - User interface
   - Runs on port 3000 (mapped to port 80 in container)

## Benefits of Split Architecture

- **Isolation**: If ViDove subprocess crashes, only backend is affected, frontend stays responsive
- **Independent Scaling**: Scale backend and frontend separately
- **Better Resource Management**: Allocate resources appropriately to each service
- **Easier Debugging**: Issues are isolated to specific services
- **Independent Updates**: Update/restart one service without affecting the other

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 1.29+
- At least 4GB RAM available for containers
- OpenAI API key (or other LLM provider credentials)

## Quick Start

### 1. Set Up Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
# Add other API keys as needed
```

### 2. Build and Start Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 4. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clears uploads/results)
docker-compose down -v
```

## Configuration

### Environment Variables

Edit `.env` file to configure:

```env
# Required
OPENAI_API_KEY=your_key_here

# Optional - Backend
BACKEND_PORT=8000
MAX_UPLOAD_SIZE=1073741824  # 1GB in bytes

# Optional - Frontend
FRONTEND_PORT=3000
REACT_APP_API_URL=http://localhost:8000
```

### Docker Compose Customization

Edit `docker-compose.yml` to customize:

- **Ports**: Change port mappings
- **Resources**: Adjust memory limits
- **Volumes**: Add additional mount points
- **Environment**: Add more environment variables

### Persistent Storage

The following directories are mounted as volumes for persistence:

- `./demo/backend/uploads`: Uploaded files
- `./demo/backend/results`: Translation results
- `./domain_dict`: Custom translation dictionaries
- `./configs`: Configuration files

## Health Checks

Both containers include health checks:

- **Backend**: `curl http://localhost:8000/health`
- **Frontend**: `curl http://localhost:3000/health`

Check health status:

```bash
docker-compose ps
```

## Troubleshooting

### Backend Container Issues

**Check logs:**
```bash
docker-compose logs backend
```

**Restart backend only:**
```bash
docker-compose restart backend
```

**Access backend shell:**
```bash
docker-compose exec backend /bin/bash
```

**Check ViDove process:**
```bash
docker-compose exec backend ps aux | grep python
```

### Frontend Container Issues

**Check logs:**
```bash
docker-compose logs frontend
```

**Restart frontend only:**
```bash
docker-compose restart frontend
```

**Access frontend shell:**
```bash
docker-compose exec frontend /bin/sh
```

### Common Issues

1. **Container exits immediately**
   - Check logs: `docker-compose logs [service]`
   - Verify environment variables are set
   - Ensure ports are not already in use

2. **Cannot connect to backend from frontend**
   - Verify `REACT_APP_API_URL` environment variable
   - Check if backend is healthy: `docker-compose ps`
   - Ensure both containers are on same network

3. **ViDove subprocess fails**
   - Backend container will remain running
   - Check backend logs for error details
   - Verify API keys are correctly set
   - Check if required models are available

4. **Out of memory**
   - Increase memory limits in docker-compose.yml
   - Check container stats: `docker stats`

## Development

### Rebuild After Code Changes

```bash
# Rebuild specific service
docker-compose build backend
docker-compose build frontend

# Rebuild and restart
docker-compose up -d --build
```

### Local Development with Docker

You can mount your local code for development:

```yaml
# Add to docker-compose.yml under backend service
volumes:
  - ./src:/app/src
  - ./entries:/app/entries
```

## Production Deployment

### Security Considerations

1. **Update CORS settings** in `demo/backend/main.py`
2. **Use secrets management** for API keys (Docker secrets, Kubernetes secrets)
3. **Enable HTTPS** with reverse proxy (nginx, traefik)
4. **Set resource limits** appropriately
5. **Configure logging** and monitoring

### Recommended Production Setup

```bash
# Use production docker-compose override
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  backend:
    restart: always
    environment:
      - LOG_LEVEL=INFO
    deploy:
      replicas: 2
  
  frontend:
    restart: always
```

### Scaling

Scale backend to handle more requests:

```bash
docker-compose up -d --scale backend=3
```

## Monitoring

### View Container Stats

```bash
docker stats
```

### View Resource Usage

```bash
docker-compose exec backend top
```

### Export Logs

```bash
docker-compose logs --no-color > logs.txt
```

## Backup and Restore

### Backup Results and Uploads

```bash
tar -czf backup.tar.gz demo/backend/uploads demo/backend/results
```

### Restore

```bash
tar -xzf backup.tar.gz
```

## Additional Resources

- [ViDove Documentation](../README.md)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
