FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (layer-cached) ─────────────────────────────────────
# NOTE: build context must be the PROJECT ROOT, not deployment/ — see
# docker-compose.yml (`context: ..`). This is required so this Dockerfile
# can COPY the backend/ and ai/ folders below.
COPY deployment/requirements.txt .

# Core web stack + auth + DB (what backend_api.py actually needs to run today)
RUN pip install --no-cache-dir \
    fastapi==0.111.0 "uvicorn[standard]==0.29.0" \
    python-multipart websockets \
    "python-jose[cryptography]" "passlib[bcrypt]" "bcrypt==4.0.1" \
    pydantic==2.7.1 pydantic-settings \
    httpx sqlalchemy aiosqlite asyncpg anyio

# PyTorch — Phase 2: the live API optionally loads models/biomirror_pinn.pt
# at startup for near-term glucose prediction (backend/ai_bridge.py). Also
# used by the ai/ research pipeline (ai/train.py, ai/train_checkpoint.py)
# if you run those inside this image (e.g. the `trainer` profile below).
# The live API degrades gracefully to the Bergman-only engine if this is
# ever removed to shrink the image.
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    numpy pandas scikit-learn scipy matplotlib optuna

# ── Copy application code (matches the real backend/ai/frontend layout) ──
COPY backend/  ./backend/
COPY ai/       ./ai/
COPY frontend/ ./frontend/
COPY data/     ./data/
COPY models/   ./models/

# ── Model + SQLite-fallback directories ────────────────────────────
RUN mkdir -p /app/models /app/data

# ── Non-root user (security best practice) ─────────────────────────
RUN useradd -m -u 1000 biomirror && chown -R biomirror:biomirror /app
USER biomirror

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run as a package (backend.backend_api:app) — matches QUICKSTART.md and
# is what makes `from database import db` resolve correctly inside
# backend_api.py's own sys.path fix.
CMD ["uvicorn", "backend.backend_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
