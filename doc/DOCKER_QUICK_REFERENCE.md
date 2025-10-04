# ViDove Docker Quick Reference

## Quick Commands

### Start/Stop
```bash
docker-compose up -d              # Start services
docker-compose down               # Stop services
docker-compose restart            # Restart all
docker-compose restart backend    # Restart backend only
docker-compose restart frontend   # Restart frontend only
```

### View Logs
```bash
docker-compose logs -f            # All logs (follow)
docker-compose logs -f backend    # Backend logs only
docker-compose logs -f frontend   # Frontend logs only
docker-compose logs --tail=100    # Last 100 lines
```

### Status & Health
```bash
docker-compose ps                 # Container status
docker stats                      # Resource usage
curl http://localhost:8000/health # Backend health
curl http://localhost:3000/health # Frontend health
```

### Build & Rebuild
```bash
docker-compose build              # Build images
docker-compose build backend      # Build backend only
docker-compose up -d --build      # Rebuild and start
docker-compose build --no-cache   # Force clean build
```

### Management Script
```bash
./docker-manage.sh start          # Start services
./docker-manage.sh stop           # Stop services
./docker-manage.sh logs           # View logs
./docker-manage.sh logs-backend   # Backend logs
./docker-manage.sh health         # Check health
./docker-manage.sh stats          # Resource usage
./docker-manage.sh shell-backend  # Backend shell
./docker-manage.sh rebuild        # Rebuild all
```

## Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **OpenAPI Spec**: http://localhost:8000/openapi.json

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs backend

# Verify .env file
cat .env | grep OPENAI_API_KEY

# Check port conflicts
lsof -i :8000  # Backend port
lsof -i :3000  # Frontend port
```

### ViDove Task Fails
```bash
# Backend stays up ✓
# Check backend logs for details
docker-compose logs -f backend

# Shell into backend
docker-compose exec backend /bin/bash

# Check processes
docker-compose exec backend ps aux | grep python
```

### Out of Memory
```bash
# Check usage
docker stats

# Edit docker-compose.yml memory limits:
# backend: 4G
# frontend: 512M
```

### Clean Reset
```bash
# Stop and remove everything (WARNING: deletes data)
docker-compose down -v

# Clean Docker system
docker system prune -a
```

## File Locations (Inside Containers)

### Backend
- **App**: `/app`
- **Pipeline venv**: `/app/venv`
- **Backend venv**: `/app/backend-venv`
- **Uploads**: `/app/demo/backend/uploads`
- **Results**: `/app/demo/backend/results`
- **Configs**: `/app/configs`

### Frontend
- **Built app**: `/usr/share/nginx/html`
- **Nginx config**: `/etc/nginx/conf.d/default.conf`

## Environment Variables

### Required
```bash
OPENAI_API_KEY=sk-...
```

### Optional
```bash
BACKEND_PORT=8000
FRONTEND_PORT=3000
REACT_APP_API_URL=http://localhost:8000
LOG_LEVEL=INFO
```

## Development Mode
```bash
# With hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Or use script
./docker-manage.sh dev
```

## Production Mode
```bash
# Optimized for production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or use script
./docker-manage.sh prod

# Scale backend
docker-compose up -d --scale backend=3
```

## Backup & Restore

### Backup
```bash
tar -czf vidove-backup-$(date +%Y%m%d).tar.gz \
  demo/backend/uploads \
  demo/backend/results
```

### Restore
```bash
tar -xzf vidove-backup-20241004.tar.gz
docker-compose restart
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Port already in use | Change ports in docker-compose.yml or stop conflicting service |
| Backend exits immediately | Check .env has OPENAI_API_KEY set |
| Frontend can't reach backend | Verify both containers are on same network |
| ViDove subprocess crashes | Backend stays up! Check logs for error details |
| Out of memory | Increase memory limits in docker-compose.yml |

## Network Inspection
```bash
# Show network details
docker network inspect vidove-network

# Show container IPs
docker-compose exec backend hostname -i
docker-compose exec frontend hostname -i

# Test connectivity
docker-compose exec frontend ping backend
```

## Performance Tips

1. **Use volumes for development**: Mount code for hot reload
2. **Use build cache**: Don't use --no-cache unless necessary
3. **Scale backend**: `--scale backend=N` for more workers
4. **Increase memory**: Edit docker-compose.yml limits
5. **Monitor stats**: Use `docker stats` to find bottlenecks

## Security Checklist

- [ ] Set strong API keys in .env
- [ ] Configure CORS for production (in main.py)
- [ ] Use HTTPS with reverse proxy
- [ ] Don't expose Docker socket
- [ ] Regularly update base images
- [ ] Use Docker secrets in production
- [ ] Limit resource usage
- [ ] Enable logging and monitoring

## Useful Docker Commands

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Clean everything
docker system prune -a --volumes

# Export container
docker export backend > backend.tar

# Show container details
docker inspect vidove-backend

# Copy files from container
docker cp vidove-backend:/app/results ./local-results
```

## More Resources

- Full Guide: [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
- Migration: [DOCKER_MIGRATION_SUMMARY.md](DOCKER_MIGRATION_SUMMARY.md)
- Main README: [README.md](README.md)
- Docker Docs: https://docs.docker.com/
