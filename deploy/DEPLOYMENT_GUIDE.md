# ============================================================================
# 🚀 Deployment Guide — Log Anomaly Detection Platform
# ============================================================================
#
# Three deployment paths: Local → Cloud → PaaS
# Choose based on your needs.
#

# ============================================================================
# 📋 TABLE OF CONTENTS
# ============================================================================
# 1. Local Deployment (Docker Compose)
# 2. Cloud VM Deployment (AWS / GCP / Azure)
# 3. PaaS Deployment (Railway / Render / Fly.io)
# 4. Production Checklist
# ============================================================================


# ============================================================================
# 1️⃣  LOCAL DEPLOYMENT (Docker Compose)
# ============================================================================
#
# Prerequisites: Docker Desktop installed
# Time: ~5 minutes
#
# WINDOWS (PowerShell):
#
#   Step 1: Copy environment file
#     Copy-Item .env.example .env
#
#   Step 2: (Optional) Edit .env with your settings
#     notepad .env
#
#   Step 3: Build and start all services
#     docker-compose up -d --build
#
#   Step 4: Verify health
#     curl http://localhost:8000/health
#     # Or open in browser: http://localhost:8000/docs
#
#   Step 5: View logs
#     docker-compose logs -f api
#
#   Step 6: Stop everything
#     docker-compose down
#
#
# LINUX / MAC (bash):
#
#   cp .env.example .env
#   docker compose up -d --build
#   curl http://localhost:8000/health
#
#
# SERVICES:
#   ┌──────────────┬──────┬──────────────────────────────┐
#   │ Service      │ Port │ URL                          │
#   ├──────────────┼──────┼──────────────────────────────┤
#   │ FastAPI      │ 8000 │ http://localhost:8000/docs   │
#   │ Dashboard    │ 8501 │ http://localhost:8501        │
#   │ Prometheus   │ 9090 │ http://localhost:9090        │
#   │ Grafana      │ 3000 │ http://localhost:3000        │
#   └──────────────┴──────┴──────────────────────────────┘


# ============================================================================
# 2️⃣  CLOUD VM DEPLOYMENT (AWS EC2 / GCP / Azure)
# ============================================================================
#
# Best for: Full control, custom networking, multi-service stack
# Cost: ~$15-30/month for a t3.medium
#
# A. PROVISION A VM
#
#   AWS EC2:
#     aws ec2 run-instances \
#       --image-id ami-0c55b159cbfafe1f0 \
#       --instance-type t3.medium \
#       --key-name your-key \
#       --security-group-ids sg-xxxxx \
#       --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=log-anomaly}]'
#
#   GCP:
#     gcloud compute instances create log-anomaly \
#       --machine-type e2-medium \
#       --image-family ubuntu-2204-lts \
#       --image-project ubuntu-os-cloud
#
#   Azure:
#     az vm create \
#       --resource-group myResourceGroup \
#       --name log-anomaly \
#       --size Standard_B2s \
#       --image Ubuntu2204
#
#
# B. SSH IN AND SETUP
#
#   ssh -i your-key.pem ubuntu@<VM_IP>
#
#   # Install Docker
#   curl -fsSL https://get.docker.com | sh
#   sudo usermod -aG docker $USER
#   newgrp docker
#
#   # Clone and deploy
#   git clone https://github.com/<your-username>/log-anomaly-detection-platform.git
#   cd log-anomaly-detection-platform
#   cp .env.example .env
#   nano .env                    # Set production values
#   docker compose up -d --build
#
#
# C. OPEN FIREWALL PORTS
#
#   AWS:  Add inbound rules for ports 8000, 8501, 3000 in Security Group
#   GCP:  gcloud compute firewall-rules create allow-app --allow tcp:8000,tcp:8501,tcp:3000
#   Azure: az vm open-port --port 8000,8501,3000 --name log-anomaly
#
#
# D. (OPTIONAL) SSL WITH CADDY REVERSE PROXY
#
#   sudo apt install caddy
#   # /etc/caddy/Caddyfile:
#   # api.yourdomain.com {
#   #     reverse_proxy localhost:8000
#   # }
#   # dashboard.yourdomain.com {
#   #     reverse_proxy localhost:8501
#   # }
#   sudo systemctl restart caddy


# ============================================================================
# 3️⃣  PaaS DEPLOYMENT (Easiest — Zero DevOps)
# ============================================================================
#
# RAILWAY (recommended for portfolio):
#
#   1. Install CLI:  npm install -g @railway/cli
#   2. Login:        railway login
#   3. Init:         railway init
#   4. Deploy:       railway up
#   5. Set vars:     railway variables set LADP_ENV=production
#                    railway variables set LADP_API_KEY=<your-secret>
#
#   Your app → https://<app-name>.railway.app
#
#
# RENDER (auto-deploy from GitHub):
#
#   1. Push code to GitHub
#   2. Go to https://render.com → New Web Service
#   3. Connect your GitHub repo
#   4. Render reads the render.yaml file and auto-configures everything
#   5. Set environment variables in Render dashboard
#
#   Your app → https://<app-name>.onrender.com
#
#
# FLY.IO:
#
#   1. Install CLI:  curl -L https://fly.io/install.sh | sh
#   2. Login:        flyctl auth login
#   3. Launch:       flyctl launch    (auto-detects Dockerfile)
#   4. Deploy:       flyctl deploy
#   5. Set secrets:  flyctl secrets set LADP_API_KEY=<your-secret>
#
#   Your app → https://<app-name>.fly.dev


# ============================================================================
# 4️⃣  PRODUCTION CHECKLIST
# ============================================================================
#
# Before going live, ensure:
#
#  ✅ Security
#     [ ] Set a strong LADP_API_KEY in .env
#     [ ] Restrict LADP_ALLOWED_ORIGINS to your frontend domain
#     [ ] Set LADP_ENV=production
#     [ ] Set LADP_DEBUG=false
#     [ ] Set LADP_LOG_LEVEL=INFO (not DEBUG)
#
#  ✅ Model
#     [ ] Train the model before starting the API:
#         python main.py --mode train --logs 50000
#     [ ] Verify model artifacts exist in models/artifacts/latest/
#
#  ✅ Monitoring
#     [ ] Prometheus is scraping /metrics endpoint
#     [ ] Grafana dashboards are configured
#     [ ] Alert rules set for anomaly rate > threshold
#
#  ✅ Infrastructure
#     [ ] Health check endpoint responds: GET /health
#     [ ] SSL/TLS configured (HTTPS)
#     [ ] Firewall allows only necessary ports
#     [ ] Log rotation configured
#     [ ] Backups for model artifacts
#
