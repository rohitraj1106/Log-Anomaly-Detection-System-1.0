#!/usr/bin/env bash
# ==============================================================================
# Production Deployment Script — Log Anomaly Detection Platform
# ==============================================================================
# Usage:
#   ./deploy/deploy.sh local        — Docker Compose on local machine
#   ./deploy/deploy.sh cloud        — Deploy to cloud VM (EC2/GCP/Azure)
#   ./deploy/deploy.sh railway      — Deploy to Railway (free tier)
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ==============================================================================
# Pre-flight Checks
# ==============================================================================
preflight_check() {
    log_info "Running pre-flight checks..."

    # Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Install from https://docs.docker.com/get-docker/"
        exit 1
    fi
    log_ok "Docker found: $(docker --version)"

    # Docker Compose
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        log_error "Docker Compose is not installed."
        exit 1
    fi
    log_ok "Docker Compose found"

    # Check if .env exists
    if [ ! -f ".env" ]; then
        log_warn ".env file not found. Copying from .env.example..."
        cp .env.example .env
        log_ok "Created .env from .env.example — please review and update it"
    fi
}

# ==============================================================================
# Local Deployment (Docker Compose)
# ==============================================================================
deploy_local() {
    log_info "🚀 Starting LOCAL deployment with Docker Compose..."
    preflight_check

    # Build images
    log_info "Building Docker images..."
    $COMPOSE_CMD build --no-cache

    # Start services
    log_info "Starting services..."
    $COMPOSE_CMD up -d

    # Wait for health check
    log_info "Waiting for API to be healthy..."
    for i in {1..30}; do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            log_ok "API is healthy!"
            break
        fi
        if [ "$i" -eq 30 ]; then
            log_error "API did not become healthy within 60 seconds"
            $COMPOSE_CMD logs api
            exit 1
        fi
        sleep 2
    done

    echo ""
    log_ok "═══════════════════════════════════════════════"
    log_ok "  Deployment Complete! Services running:"
    log_ok "═══════════════════════════════════════════════"
    echo ""
    echo -e "  ${GREEN}API Docs${NC}     → http://localhost:8000/docs"
    echo -e "  ${GREEN}Health${NC}       → http://localhost:8000/health"
    echo -e "  ${GREEN}Dashboard${NC}    → http://localhost:8501"
    echo -e "  ${GREEN}Prometheus${NC}   → http://localhost:9090"
    echo -e "  ${GREEN}Grafana${NC}      → http://localhost:3000"
    echo ""
    echo -e "  Stop:  ${YELLOW}$COMPOSE_CMD down${NC}"
    echo -e "  Logs:  ${YELLOW}$COMPOSE_CMD logs -f api${NC}"
    echo ""
}

# ==============================================================================
# Cloud VM Deployment (EC2 / GCP / Azure)
# ==============================================================================
deploy_cloud() {
    log_info "🌩️  Cloud VM Deployment Guide"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Step-by-Step: Deploy to a Cloud VM"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  1. PROVISION A VM"
    echo "     • AWS:   aws ec2 run-instances --instance-type t3.medium ..."
    echo "     • GCP:   gcloud compute instances create log-anomaly --machine-type e2-medium"
    echo "     • Azure: az vm create --size Standard_B2s --name log-anomaly"
    echo ""
    echo "  2. SSH INTO THE VM"
    echo "     ssh -i your-key.pem ubuntu@<YOUR_VM_IP>"
    echo ""
    echo "  3. RUN THE SETUP SCRIPT ON THE VM"
    echo "     curl -sSL https://raw.githubusercontent.com/<your-repo>/main/deploy/cloud_setup.sh | bash"
    echo ""
    echo "  4. OR MANUALLY:"
    echo "     git clone https://github.com/<your-repo>.git"
    echo "     cd log-anomaly-detection-platform"
    echo "     cp .env.example .env"
    echo "     # Edit .env with production values"
    echo "     docker compose up -d"
    echo ""
    echo "  5. CONFIGURE FIREWALL"
    echo "     Open ports: 8000 (API), 8501 (Dashboard), 3000 (Grafana)"
    echo ""
    echo "  6. (OPTIONAL) SET UP DOMAIN + SSL"
    echo "     Use Caddy or Nginx as reverse proxy with Let's Encrypt"
    echo ""
}

# ==============================================================================
# Railway Deployment (PaaS — easiest)
# ==============================================================================
deploy_railway() {
    log_info "🚂 Railway Deployment"
    echo ""

    # Check if Railway CLI is installed
    if ! command -v railway &> /dev/null; then
        log_warn "Railway CLI not installed. Installing..."
        echo "  npm install -g @railway/cli"
        echo "  OR visit https://railway.app"
        echo ""
    fi

    echo "═══════════════════════════════════════════════════════════════"
    echo "  Deploy to Railway (free tier available)"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  1. Install Railway CLI:"
    echo "     npm install -g @railway/cli"
    echo ""
    echo "  2. Login & Initialize:"
    echo "     railway login"
    echo "     railway init"
    echo ""
    echo "  3. Deploy:"
    echo "     railway up"
    echo ""
    echo "  4. Set environment variables:"
    echo "     railway variables set LADP_ENV=production"
    echo "     railway variables set LADP_API_KEY=your-secret-key"
    echo "     railway variables set LADP_LOG_LEVEL=INFO"
    echo ""
    echo "  Railway auto-detects your Dockerfile and deploys!"
    echo "  Your API will be live at: https://<app>.railway.app"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ALTERNATIVE PLATFORMS (same process):"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  Render:  https://render.com  (connect GitHub, auto-deploy)"
    echo "  Fly.io:  flyctl launch       (uses Dockerfile)"
    echo ""
}

# ==============================================================================
# Stop & Cleanup
# ==============================================================================
deploy_stop() {
    log_info "Stopping all services..."
    preflight_check
    $COMPOSE_CMD down
    log_ok "All services stopped"
}

deploy_status() {
    preflight_check
    $COMPOSE_CMD ps
}

# ==============================================================================
# Main
# ==============================================================================
case "${1:-help}" in
    local)
        deploy_local
        ;;
    cloud)
        deploy_cloud
        ;;
    railway|render|paas)
        deploy_railway
        ;;
    stop)
        deploy_stop
        ;;
    status)
        deploy_status
        ;;
    *)
        echo ""
        echo "Usage: ./deploy/deploy.sh <command>"
        echo ""
        echo "Commands:"
        echo "  local     Deploy locally with Docker Compose"
        echo "  cloud     Show cloud VM deployment guide"
        echo "  railway   Show Railway/Render/Fly.io deployment guide"
        echo "  stop      Stop all running services"
        echo "  status    Show status of running services"
        echo ""
        ;;
esac
