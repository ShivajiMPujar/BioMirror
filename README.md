# BioMirror

A physics-informed Digital Twin platform for personalized Type-2 diabetes glucose monitoring, forecasting, and lifestyle analytics.

## Overview

BioMirror models a user's metabolic state as a live "Digital Twin" driven by the Bergman Minimal Model (a well-established glucose-insulin ODE system), personalizes that model with each user's own logged health data, and layers a Physics-Informed Neural Network (PINN) on top for short-horizon glucose prediction. Every prediction, recommendation, and analytics view is computed from the logged-in user's own history — there is no synthetic or randomly-generated health data anywhere in the running application.

## Core Features

- **Digital Twin simulation** — real-time metabolic state (glucose, insulin action, insulin sensitivity) driven by the Bergman Minimal Model, with daily/weekly historical snapshots and a confidence/calibration score that improves as more personal data accumulates.
- **PINN-assisted glucose forecasting** — a trained neural network layered on the physics engine for near-term prediction, with an automatic plausibility check that falls back to the physics-only engine whenever the model's output isn't physiologically reasonable (see [Limitations](#limitations)).
- **Personal Learning Engine** — recomputes user-specific statistical features (morning/night glucose averages, exercise effectiveness, sleep effect on glucose, logged activity trends, glucose variability, and more) from each user's own stored history after every log.
- **Personalized recommendations** — generated from correlation analysis against the user's own data (e.g. "glucose averaged X mg/dL lower on days with more than 7,000 steps"), not static templates.
- **Health logging** — glucose, meals, activity, sleep, weight, blood pressure, water intake, HbA1c, and medications.
- **What-If Simulator** — models the glucose impact of a hypothetical meal/exercise/sleep scenario before it happens.
- **Reversal Engine** — a multi-factor scoring system (reversal score, lifestyle score, insulin sensitivity score, metabolic health score) with human-readable explanations.
- **SHAP-based explainability** — factor-level attribution for glucose predictions.
- **AI Coach** — conversational guidance grounded in the user's own recent history; uses Groq when `GROQ_API_KEY` is configured, with Anthropic and an offline rule-based fallback available.
- **Goals & analytics** — target tracking, weekly summaries, trend charts, and correlation analysis.
- **Secure multi-user data isolation** — JWT authentication, per-user data scoping enforced on every endpoint, hashed passwords, revocable refresh tokens.

## Architecture

```
Frontend (SPA)
      ↓
FastAPI Backend  ──────────────┐
      ↓                        │
SQLAlchemy / SQLite|Postgres    │
      ↓                        │
AI Engine (feature engineering, PINN, reversal engine, SHAP)
      ↓                        │
Prediction + Personalized Analytics
      ↓
Dashboard
```

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | HTML/CSS/JavaScript single-page application (Chart.js for visualization) |
| Backend | FastAPI (Python), Uvicorn |
| Database | SQLAlchemy (async) — SQLite by default, PostgreSQL for production |
| AI/ML | PyTorch (PINN model), scikit-learn, pandas, NumPy — optional; the API degrades gracefully to the physics-only engine if unavailable |
| Authentication | JWT (python-jose), bcrypt password hashing |
| Deployment | Docker, Docker Compose, Nginx |

## Project Structure

```
BioMirror/
├── backend/
│   ├── backend_api.py       # FastAPI application — all routes
│   ├── database.py          # SQLAlchemy models and async data access
│   ├── ai_bridge.py         # Connects the ai/ modules to the live API
│   └── food_database.py     # Glycemic index / meal-response lookup
├── ai/
│   ├── pinn_model.py            # PINN architecture + training
│   ├── feature_engineering.py   # Derived health features
│   ├── reversal_engine.py       # Reversal/lifestyle/health scoring
│   ├── evaluation_metrics.py    # Model evaluation utilities (MAE/RMSE/MARD/R²/SHAP)
│   ├── federated_learning.py    # Federated learning simulation (offline research module)
│   ├── optuna_hypertuning.py    # Hyperparameter search (offline research module)
│   ├── train.py                 # Full training/evaluation pipeline
│   ├── train_checkpoint.py      # Produces the deployable model checkpoint
│   ├── demo.py                  # Offline CLI demo (no server/DB required)
│   └── tests.py                 # Automated test suite
├── frontend/
│   └── biomirror_app.html   # The application (open directly in a browser)
├── data/
│   └── diabetes_lifestyle_dataset_500.csv
├── models/
│   └── biomirror_pinn.pt    # Trained PINN checkpoint, loaded at startup
├── deployment/
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── nginx.conf
│   ├── setup_windows.bat
│   └── .env.example
└── README.md
```

## Setup

1. **Create and activate a virtual environment**
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

2. **Install dependencies**
   ```
   pip install -r deployment/requirements.txt
   ```
   PyTorch is optional and installed separately — it enables PINN-assisted prediction; the API runs fine without it, falling back to the physics-only engine:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Configure environment variables**
   ```
   copy deployment\.env.example .env      # Windows
   cp deployment/.env.example .env        # macOS/Linux
   ```
   Fill in only the values you need (see [Environment Variables](#environment-variables) below). SQLite persistence works out of the box with no configuration.

4. **Start the backend**
   ```
   python -m uvicorn backend.backend_api:app --host 127.0.0.1 --port 8000 --reload
   ```

5. **Start the frontend**
   Open `frontend/biomirror_app.html` directly in a browser (double-click, or serve it with any static file server).

6. **Open the application**
   The frontend connects to `http://127.0.0.1:8000` by default. API documentation is available at `http://127.0.0.1:8000/docs`.

## Environment Variables

Only variables the current code actually reads. Copy `deployment/.env.example` and fill in the ones relevant to your setup — everything has a safe default for local development.

```
# Security
BIOMIRROR_SECRET=

# Database — leave empty for local SQLite (created automatically); set for PostgreSQL
DATABASE_URL=

# AI Coach — optional, falls back to an offline rule-based coach if unset
# AI Coach — Groq is preferred; the key is read by the backend only
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b
# Optional fallback provider
ANTHROPIC_API_KEY=

# Optional: only read by the offline training pipeline / trainer container
PINN_MODEL_PATH=
```

Never commit a real `.env` file — `deployment/.env.example` contains placeholders only.

## API / Application

The backend exposes a REST + WebSocket API covering authentication, health logging, the Digital Twin, forecasting, the What-If simulator, the AI Coach, goals, analytics, and data export. Interactive documentation for the full route list is generated automatically at `/docs` once the server is running.

## AI Prediction Pipeline

```
User's manually logged health data
        ↓
Personal feature engineering (computed from that user's own history)
        ↓
Digital Twin state (Bergman Minimal Model)
        ↓
PINN inference (if a checkpoint is loaded and its output passes a plausibility check)
        ↓
Prediction + SHAP explanation
        ↓
Dashboard / Analytics
```

**Training is offline, inference is online.** The PINN checkpoint in `models/biomirror_pinn.pt` is trained ahead of time via `ai/train_checkpoint.py` against the shipped dataset; the live server only ever performs inference against the already-trained checkpoint. The model is **not** continuously retrained on live user data — each glucose, meal, activity, or sleep log updates the Digital Twin's physics state and this user's personal statistical features immediately, but does not re-train the neural network. Retraining is a separate, manual, offline step.

## Limitations

- **PINN prediction accuracy is currently poor.** The shipped checkpoint's measured test-set performance (MAE ≈ 23.8 mg/dL, R² ≈ −603) is well below usable — it performs worse than a naive baseline. The application does not rely on this blindly: every PINN prediction is checked against physiologically plausible bounds before use, and the Digital Twin automatically falls back to the Bergman physics-only engine whenever the check fails, which in practice is currently always. Retraining with a corrected training configuration is required before PINN output should be trusted for prediction.
- **No clinical validation.** BioMirror is a software prototype, not a validated medical device. It does not perform diagnosis and should not be used as a substitute for clinical glucose monitoring or medical advice.
- **No continuous glucose monitoring (CGM) integration.** Glucose values are either manually logged or derived from the Digital Twin's simulation — there is no live sensor feed.
