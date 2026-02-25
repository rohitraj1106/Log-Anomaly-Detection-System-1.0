#!/usr/bin/env bash
# ==============================================================================
# Cloud VM Setup Script — Run this on a fresh Ubuntu VM
# ==============================================================================
# Usage: curl -sSL <url>/cloud_setup.sh | bash
# ==============================================================================

set -euo pipefail

echo "═══════════════════════════════════════════════"
echo "  Log Anomaly Detection Platform — VM Setup"
echo "═══════════════════════════════════════════════"

# Update system
echo "[1/5] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# Install Docker
echo "[2/5] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "  ✅ Docker installed"
else
    echo "  ✅ Docker already installed"
fi

# Install Docker Compose (V2 plugin)
echo "[3/5] Installing Docker Compose..."
sudo apt-get install -y -qq docker-compose-plugin 2>/dev/null || true
echo "  ✅ Docker Compose ready"

# Clone repository (user needs to customize this URL)
echo "[4/5] Cloning repository..."
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/log-anomaly-detection-platform.git}"
DEPLOY_DIR="${DEPLOY_DIR:-/opt/log-anomaly-platform}"

if [ ! -d "$DEPLOY_DIR" ]; then
    sudo git clone "$REPO_URL" "$DEPLOY_DIR"
    sudo chown -R "$USER:$USER" "$DEPLOY_DIR"
else
    echo "  Directory $DEPLOY_DIR already exists, pulling latest..."
    cd "$DEPLOY_DIR" && git pull
fi

cd "$DEPLOY_DIR"

# Setup environment
echo "[5/5] Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    # Set production defaults
    sed -i 's/LADP_ENV=development/LADP_ENV=production/' .env
    sed -i 's/LADP_DEBUG=true/LADP_DEBUG=false/' .env
    sed -i 's/LADP_LOG_LEVEL=DEBUG/LADP_LOG_LEVEL=INFO/' .env
    echo "  ✅ Created .env with production defaults"
    echo "  ⚠️  Please edit .env to set LADP_API_KEY and LADP_ALLOWED_ORIGINS"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup complete! Next steps:"
echo "═══════════════════════════════════════════════"
echo ""
echo "  1. Edit the .env file:"
echo "     nano $DEPLOY_DIR/.env"
echo ""
echo "  2. Start the platform:"
echo "     cd $DEPLOY_DIR"
echo "     docker compose up -d"
echo ""
echo "  3. Open firewall ports:"
echo "     sudo ufw allow 8000/tcp  # API"
echo "     sudo ufw allow 8501/tcp  # Dashboard"
echo "     sudo ufw allow 3000/tcp  # Grafana"
echo ""
echo "  4. Check status:"
echo "     docker compose ps"
echo "     curl http://localhost:8000/health"
echo ""
