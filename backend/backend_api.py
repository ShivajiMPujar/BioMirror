"""
BioMirror — FastAPI Backend (Complete v3 — Phase 1: Real Persistence)
=======================================================================
All endpoints required by the full SPA frontend. Same routes, same request/
response schemas as before — the ONLY architectural change in this pass is
that every in-memory dict has been replaced by real SQLAlchemy-backed
storage via database.py (SQLite by default, PostgreSQL if DATABASE_URL is
set). No user data is lost on restart.

Phase 1 changes vs. the previous version:
  • Fixed the `from database import db` import (it silently failed before —
    see the sys.path fix right below the imports).
    • USERS_DB, LOGS_DB, EVENTS_LOG, CHAT_HISTORY, NOTIFICATIONS,
    SIM_HISTORY, GOALS_DB → all removed. Replaced by real DB reads/writes.
  • TWIN_STATE → replaced by a write-through cache (TWIN_CACHE) whose source
    of truth is the `twin_state` DB table: every mutation is saved to the DB
    immediately, and any cache miss (e.g. after a restart) is lazily
    rehydrated from the DB. This keeps the hot simulation loop (used by the
    WebSocket tick and every glucose log) fast while guaranteeing nothing is
    lost across restarts.
  • WS_CONNECTIONS remains in-memory ONLY — these are live socket objects,
    not user data, and cannot be serialized to a database by design.
  • Added two small ADDITIVE auth endpoints (do not change any existing
    route): POST /auth/reset-password and GET /auth/verify-email/{token}.
    These complete the persistence story for "password reset" and "email
    verification" that Phase 1 was asked to persist, without touching any
    endpoint the frontend already calls.
"""

import sys, os
# Make sure this file's own directory (backend/) is importable as a plain
# module path regardless of how the app is launched (e.g.
# `uvicorn backend.backend_api:app` from the project root). This is the
# fix for the DB import that previously failed silently.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import (
    FastAPI, HTTPException, Depends, WebSocket,
    WebSocketDisconnect, status, Query, Request, Body
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncio, json, math, time, hashlib, uuid, httpx, re, logging

from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger("biomirror")

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
try:
    from database import db as _db
    _DB_AVAILABLE = True
except ImportError as e:
    _db = None
    _DB_AVAILABLE = False
    logger.error(f"[DB] Failed to import database.py: {e}")

# ─────────────────────────────────────────────
# AI MODULES (Phase 2) — feature engineering, PINN, reversal engine, SHAP,
# food database. All loaded/constructed once here at import time; see
# backend/ai_bridge.py for the full integration + fallback design.
# ─────────────────────────────────────────────
try:
    import ai_bridge as _ai
    _AI_BRIDGE_AVAILABLE = True
except ImportError as e:
    _ai = None
    _AI_BRIDGE_AVAILABLE = False
    logger.error(f"[AI] Failed to import ai_bridge.py: {e}")

try:
    from food_database import calculate_nutrition, list_foods as list_food_catalog
except ImportError as e:
    calculate_nutrition = None
    list_food_catalog = None
    logger.error(f"[Food] food_database.py could not be imported: {e}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SECRET_KEY            = os.getenv("BIOMIRROR_SECRET", "biomirror-dev-secret-change-in-production")
ALGORITHM             = "HS256"
ACCESS_TOKEN_MINUTES  = 60 * 24        # 24 h
REFRESH_TOKEN_DAYS    = 30

ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL       = "claude-sonnet-4-20250514"
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL            = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_URL          = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI(
    title="BioMirror API v3",
    description="Physics-Informed Digital Twin for Type-2 Diabetes Reversal",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ─────────────────────────────────────────────
# RUNTIME-ONLY STATE
# (Everything else now lives in the database — see database.py)
# ─────────────────────────────────────────────
# TWIN_CACHE: write-through cache. Source of truth is the `twin_state` table.
# Every write here is immediately persisted via save_twin(); a cache miss
# (e.g. right after a server restart) is transparently rehydrated from the DB.
TWIN_CACHE:     Dict[str, dict]              = {}
# WS_CONNECTIONS: live socket objects only — inherently runtime state, not
# user data, and not something a database can (or should) store.
WS_CONNECTIONS: Dict[str, List[WebSocket]]   = {}
# Populated once at startup by ai_bridge.initialize() — see startup() below.
AI_STATUS: Dict[str, Any] = {'feature_engineering': False, 'pinn': False,
                              'pinn_status': {'available': False},
                              'reversal_engine': False, 'shap': False, 'food_database': False}


# ─────────────────────────────────────────────
# BERGMAN EDGE ENGINE  (CPU, no PyTorch) — unchanged from previous version
# ─────────────────────────────────────────────
class BergmanEngine:
    P1 = 0.028; P2 = 0.025; GB = 90.0; IB = 7.0; DT = 5.0

    def __init__(self, p3: float = 3.5e-5):
        self.p3 = p3

    def step(self, G, X, I, meal):
        dG = -(self.P1 + X) * G + self.P1 * self.GB + meal
        dX = -self.P2 * X + self.p3 * (I - self.IB)
        return max(40.0, min(400.0, G + dG * self.DT)), max(0.0, X + dX * self.DT)

    def meal_bolus(self, carbs, elapsed, peak=45):
        if carbs <= 0: return 0.0
        return carbs * 0.6 * math.exp(-0.5 * ((elapsed - peak) / 15) ** 2) / 15

    @staticmethod
    def classify(g):
        if g < 54:   return "severe_low"
        if g < 70:   return "low"
        if g <= 140: return "normal"
        if g <= 180: return "elevated"
        if g <= 250: return "high"
        return "critical"


# ─────────────────────────────────────────────
# AUTH UTILITIES
# ─────────────────────────────────────────────
def hash_pw(pw):  return pwd_context.hash(pw)
def verify_pw(plain, hashed): return pwd_context.verify(plain, hashed)

def make_token(data: dict, expires_delta: timedelta) -> str:
    payload = {**data, "exp": datetime.utcnow() + expires_delta, "iat": datetime.utcnow()}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def make_access_token(sub: str) -> str:
    return make_token({"sub": sub, "type": "access"}, timedelta(minutes=ACCESS_TOKEN_MINUTES))

def make_refresh_token(sub: str) -> str:
    # jti ensures uniqueness even if two refresh tokens are minted for the
    # same user within the same second (iat/exp alone are not unique enough
    # once tokens are persisted to a DB column with a UNIQUE constraint).
    return make_token({"sub": sub, "type": "refresh", "jti": uuid.uuid4().hex},
                       timedelta(days=REFRESH_TOKEN_DAYS))

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(401, "Invalid token")
        user = await _db.get_user_by_username(username)
        if not user:
            raise HTTPException(401, "Invalid token")
        return user
    except JWTError:
        raise HTTPException(401, "Token expired or invalid")

async def get_optional_user(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    try:
        return await get_current_user(token)
    except Exception:
        return None


# ─────────────────────────────────────────────
# TWIN STATE HELPERS  (write-through cache over the DB)
# ─────────────────────────────────────────────
def build_initial_twin(patient_id: str, profile: dict) -> dict:
    bmi = profile.get("bmi", 27.0)
    p3  = max(1e-5, 7e-5 - (bmi - 22) * 2e-6)
    baseline_g = profile.get("baseline_glucose", 140.0)
    return {
        "patient_id": patient_id,
        "G": baseline_g, "X": 0.02, "I": 12.0,
        "p3": p3, "p3_index": round(p3 / 5e-5 * 100, 1),
        "meal_queue": [], "sim_time": 0.0,
        "history": [], "last_event": None,
        "created_at": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat(),
    }

async def load_twin(patient_id: str) -> Optional[dict]:
    """Read-through: cache first, DB on miss (e.g. right after a restart)."""
    if patient_id in TWIN_CACHE:
        return TWIN_CACHE[patient_id]
    state = await _db.get_twin_state(patient_id)
    if state:
        TWIN_CACHE[patient_id] = state
    return state

async def save_twin(patient_id: str, state: dict):
    """Write-through: update cache AND persist to the DB immediately.
    Used by every explicit user action (glucose/meal/activity logs, profile
    updates, etc.) — these must always persist right away."""
    TWIN_CACHE[patient_id] = state
    await _db.save_twin_state(patient_id, state)

def touch_twin_cache(patient_id: str, state: dict):
    """Cache-only update — no DB write. Used by the high-frequency WebSocket
    simulation tick (every 2s) so the live dashboard still gets instant
    updates via the socket push itself, without writing to disk on every
    single tick. The DB is still checkpointed periodically (see ws_twin)
    and immediately on disconnect, so no user data is lost — only a few
    seconds of pure ambient simulation drift in the rare case of an
    ungraceful crash between checkpoints, which is negligible since every
    explicit user action (the data that actually matters) still persists
    instantly via save_twin() above."""
    TWIN_CACHE[patient_id] = state


def compute_metrics(state: dict) -> dict:
    G, p3_idx = state["G"], state["p3_index"]
    hist = state.get("history", [])
    tir  = round(sum(1 for h in hist if 70 <= h["G"] <= 180) / max(len(hist), 1) * 100, 1)
    recent = [h["G"] for h in hist[-12:]]
    if len(recent) >= 2:
        mean_r = sum(recent) / len(recent)
        vol    = round((sum((g - mean_r) ** 2 for g in recent) / len(recent)) ** 0.5, 1)
    else:
        vol = 15.0
    stress  = round(min(100, vol * 2 + max(0, G - 140) * 0.3), 1)
    rec     = round(max(0, 100 - stress * 0.6 - max(0, G - 140) * 0.15), 1)
    rev     = round(p3_idx * 0.35 + min(tir, 100) * 0.35 + rec * 0.30, 1)
    hba1c   = round((G + 46.7) / 28.7, 2)
    m_age   = round(max(28, 52 - p3_idx * 0.24), 0)
    return {
        "glucose": round(G, 1),
        "X_insulin_action": round(state["X"], 5),
        "insulin_sensitivity": round(p3_idx, 1),
        "tir_pct": tir,
        "glucose_volatility": vol,
        "metabolic_stress": stress,
        "recovery_score": rec,
        "reversal_score": rev,
        "hba1c_estimate": hba1c,
        "metabolic_age": m_age,
        "glucose_status": BergmanEngine.classify(G),
        "last_updated": datetime.utcnow().isoformat(),
    }

def generate_nudges(metrics: dict) -> list:
    nudges = []
    glucose = metrics["glucose"]
    volatility = metrics.get("glucose_volatility", 0)
    sensitivity = metrics.get("insulin_sensitivity", 0)
    recovery = metrics.get("recovery_score", 0)
    stress = metrics.get("metabolic_stress", 0)
    tir = metrics.get("tir_pct", 0)

    if glucose < 70:
        nudges.append({"type": "urgent", "icon": "⚠",
                       "message": f"Current glucose is {glucose} mg/dL, below 70. Recheck soon and follow your clinician's low-glucose plan.",
                       "priority": 1})
    elif glucose > 180:
        nudges.append({"type": "urgent", "icon": "🚶",
                       "message": f"Current glucose is {glucose} mg/dL. A gentle 10-15 minute walk may help if you feel well and it is safe for you.",
                       "priority": 1})
    elif glucose > 140:
        nudges.append({"type": "glucose", "icon": "🥗",
                       "message": f"Current glucose is {glucose} mg/dL. Keep your next meal lower in refined carbohydrates and include fiber or protein.",
                       "priority": 1})
    else:
        nudges.append({"type": "glucose", "icon": "✅",
                       "message": f"Current glucose is {glucose} mg/dL and in the target zone. Keep the routine that is working for you.",
                       "priority": 1})

    if volatility >= 20:
        nudges.append({"type": "variability", "icon": "📉",
                       "message": f"Your recent glucose variability is {volatility} mg/dL. Log meals and readings around spikes to identify your personal triggers.",
                       "priority": 2})
    elif sensitivity < 80:
        nudges.append({"type": "sensitivity", "icon": "👟",
                       "message": f"Your current insulin-sensitivity index is {sensitivity}%. A short walk after meals may support your next readings if appropriate for you.",
                       "priority": 2})
    elif recovery < 65 or stress >= 35:
        nudges.append({"type": "recovery", "icon": "🌙",
                       "message": f"Your recovery score is {recovery}/100 with metabolic stress at {stress}. Prioritize sleep, hydration, and a lower-intensity day.",
                       "priority": 2})

    if tir < 65:
        nudges.append({"type": "range", "icon": "🎯",
                       "message": "Your recent readings need closer follow-up. Log them with meal context so the next recommendation can be more specific.",
                       "priority": 3})
    else:
        nudges.append({"type": "progress", "icon": "💡",
                       "message": f"Your current state supports a reversal score of {metrics['reversal_score']}/100. Continue the habits supporting this trend.",
                       "priority": 3})
    return nudges[:3]

async def broadcast(patient_id: str, payload: dict):
    dead = []
    for ws in WS_CONNECTIONS.get(patient_id, []):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        WS_CONNECTIONS[patient_id].remove(ws)

async def push_notification(patient_id: str, title: str, body: str, ntype: str = "info"):
    await _db.push_notification(patient_id, title, body, ntype)


async def evaluate_recent_predictions(patient_id: str):
    """
    Phase 2 — retrospective model evaluation. Matches each recent
    forecast-source prediction's 5-minute-ahead value against the actual
    glucose logged closest to that time (+/- 5 min tolerance), then runs
    the real evaluation_metrics.py compute_all_metrics() over the matched
    pairs. Called after every glucose log so /metrics can show a live-
    updating MAE/RMSE/MARD/R² rather than a static number. Cheap no-op if
    there isn't enough matched data yet or the AI bridge is unavailable.
    """
    if not _AI_BRIDGE_AVAILABLE or not AI_STATUS.get('shap'):
        return
    try:
        cutoff = datetime.utcnow() - timedelta(hours=6)
        preds = await _db.get_predictions(patient_id, limit=50, source="forecast")
        preds = [p for p in preds if p.get("timestamp", "") >= cutoff.isoformat()
                 and p.get("predicted_glucose")]
        if len(preds) < 3:
            return
        actual_logs = await _db.get_logs(patient_id, "glucose", limit=500, since=cutoff)
        if not actual_logs:
            return
        pairs = []
        for p in preds:
            try:
                pred_time = datetime.fromisoformat(p["timestamp"])
            except (ValueError, TypeError):
                continue
            target_time = pred_time + timedelta(minutes=5)
            best, best_diff = None, timedelta(minutes=6)
            for log in actual_logs:
                try:
                    log_time = datetime.fromisoformat(log["timestamp"])
                except (ValueError, TypeError):
                    continue
                diff = abs(log_time - target_time)
                if diff < best_diff:
                    best, best_diff = log, diff
            if best is not None and best.get("glucose") is not None:
                pairs.append((p["predicted_glucose"][0], best["glucose"]))
        if len(pairs) < 3:
            return
        metrics = _ai.evaluate_predictions(pairs)
        if metrics:
            metrics["n_samples"] = len(pairs)
            await _db.save_evaluation_metrics(patient_id, metrics)
    except Exception as e:
        logger.warning(f"[AI] evaluate_recent_predictions failed for {patient_id}: {e}")


# ─────────────────────────────────────────────
# PERSONAL LEARNING ENGINE (Phase 3, Part C)
# ─────────────────────────────────────────────
async def get_personal_timeline(patient_id: str, days: int = 90) -> dict:
    """
    The continuously-growing personal history described in Part C —
    every log type for this patient over the requested window, grouped
    by type. This is what compute_personal_
    features() analyzes below, and what the personalized prediction/
    recommendation paths pull from instead of population-wide constants.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    logs = await _db.get_logs(patient_id, None, limit=10000, since=cutoff)
    by_type: Dict[str, list] = {}
    for l in logs:
        by_type.setdefault(l["type"], []).append(l)
    return {"by_type": by_type, "days": days, "total_entries": len(logs)}


def _hour_of(ts: str) -> Optional[int]:
    try:
        return datetime.fromisoformat(ts).hour
    except (ValueError, TypeError):
        return None


def _mean(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _stdev(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return round((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5, 2)


async def build_personalized_context(patient_id: str, user: dict, state: dict) -> dict:
    """
    Phase 3 Part E — builds the "current context" dict passed into every
    ai_bridge call (feature engineering, PINN). Before Phase 3 this used
    fixed constants (steps=4000, heart_rate=80, exercise_min=0) for every
    patient on every call. It now uses this patient's manually logged
    activity and heart-rate data, falling back to conservative constants
    when personal measurements are unavailable.
    """
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    recent_logs = await _db.get_logs(
        patient_id, None, limit=5000, since=datetime.utcnow() - timedelta(days=14)
    ) if _DB_AVAILABLE else []
    steps_by_day = _logged_steps_by_day(recent_logs)
    today_steps = steps_by_day.get(today_str)
    avg_steps = _mean(list(steps_by_day.values()))
    heart_rates = [
        l.get("heart_rate") or l.get("heart_rate_avg")
        for l in recent_logs
        if l.get("type") in ("glucose", "activity")
        and (l.get("heart_rate") or l.get("heart_rate_avg"))
    ]
    today_activity = [l for l in recent_logs if l.get("type") == "activity"
                      and l.get("timestamp", "")[:10] == today_str]
    today_meals = [l for l in recent_logs if l.get("type") == "meal"
                   and l.get("timestamp", "")[:10] == today_str]
    stress_entries = [l for l in recent_logs if l.get("type") == "stress"]
    steps = today_steps or avg_steps or 4000
    hr = _mean(heart_rates) or 80
    stress_level = stress_entries[-1].get("stress_level", 2) if stress_entries else 2
    carbs_today = sum(l.get("carbs_g", 0) or 0 for l in today_meals)
    exercise_today = sum(l.get("duration_min", 0) or 0 for l in today_activity)
    latest_food = today_meals[-1].get("food_name", "") if today_meals else ""

    return {
        "glucose": state["G"], "carbs": carbs_today, "sugar": 0,
        "steps": round(steps), "exercise_min": exercise_today, "heart_rate": round(hr),
        "stress_level": stress_level, "food": latest_food,
        "bmi": user.get('bmi') or 27.0,
        "diabetes_risk": user.get('diabetes_risk') or 1,
        "family_history": user.get('family_history') or False,
        "health_score": 70, "hour": datetime.now().hour,
    }


def _logged_steps_by_day(logs: list) -> dict:
    glucose_steps: Dict[str, int] = {}
    activity_steps: Dict[str, int] = {}
    for log in logs:
        day = log.get("timestamp", "")[:10]
        if not day:
            continue
        steps = int(log.get("steps") or 0)
        if log.get("type") == "glucose":
            glucose_steps[day] = max(glucose_steps.get(day, 0), steps)
        elif log.get("type") == "activity":
            activity_steps[day] = activity_steps.get(day, 0) + steps
    return {
        day: max(glucose_steps.get(day, 0), activity_steps.get(day, 0))
        for day in glucose_steps.keys() | activity_steps.keys()
    }


async def compute_steps_glucose_correlation(patient_id: str, days: int = 30,
                                              step_threshold: int = 7000) -> Optional[dict]:
    """
    Phase 3 Part F — the exact style of analysis the brief asked for:
    "your glucose decreased by approximately X mg/dL on days you logged
    more than 7000 steps." Joins this patient's own manually logged step
    counts with their own logged glucose, day by day.
    Returns None (not a fabricated number) if there isn't enough overlap
    between the two data sources yet.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    logs = await _db.get_logs(patient_id, None, limit=10000, since=cutoff)
    glucose_logs = [l for l in logs if l.get("type") == "glucose"]
    steps_by_day = _logged_steps_by_day(logs)
    if not steps_by_day or not glucose_logs:
        return None

    g_by_day: Dict[str, list] = {}
    for l in glucose_logs:
        g_by_day.setdefault(l["timestamp"][:10], []).append(l["glucose"])

    high_step_days, low_step_days = [], []
    for day, steps in steps_by_day.items():
        day_glucose = g_by_day.get(day)
        if not day_glucose:
            continue
        avg_g = sum(day_glucose) / len(day_glucose)
        (high_step_days if steps > step_threshold else low_step_days).append(avg_g)

    if len(high_step_days) < 2 or len(low_step_days) < 2:
        return None

    high_mean = round(sum(high_step_days) / len(high_step_days), 1)
    low_mean = round(sum(low_step_days) / len(low_step_days), 1)
    return {
        "step_threshold": step_threshold,
        "high_step_days_count": len(high_step_days),
        "low_step_days_count": len(low_step_days),
        "avg_glucose_high_step_days": high_mean,
        "avg_glucose_low_step_days": low_mean,
        "difference_mg_dl": round(low_mean - high_mean, 1),   # positive = high-step days were better
        "days_analyzed": days,
    }


async def compute_personal_features(patient_id: str) -> dict:
    """
    Phase 3, Part D — the 15 named user-specific features, each computed
    from THIS patient's own stored history (get_personal_timeline above),
    not population averages or hardcoded constants. Stored via
    database.save_personal_features() and consumed by the personalized
    prediction path (Part E) and recommendation engine (Part F).

    Every feature is None (not a fabricated default) when there isn't
    enough of the relevant log type yet — callers must handle that, and
    the recommendation engine explicitly checks for it before referencing
    a number in generated text, so no feature is ever presented as real
    when it's actually a placeholder.
    """
    timeline = await get_personal_timeline(patient_id, days=90)
    by_type = timeline["by_type"]
    glucose_logs = by_type.get("glucose", [])
    meal_logs    = by_type.get("meal", [])
    activity_logs = by_type.get("activity", [])
    sleep_logs   = by_type.get("sleep", [])
    water_logs   = by_type.get("water", [])
    hba1c_logs   = by_type.get("hba1c", [])
    weight_logs  = by_type.get("weight", [])

    # 1-2. Morning vs night glucose (morning = 05:00-11:59, night = 20:00-04:59)
    morning_g = [l["glucose"] for l in glucose_logs if l.get("glucose") is not None
                 and (h := _hour_of(l["timestamp"])) is not None and 5 <= h < 12]
    night_g   = [l["glucose"] for l in glucose_logs if l.get("glucose") is not None
                 and (h := _hour_of(l["timestamp"])) is not None and (h >= 20 or h < 5)]

    # 3. Average meal response (post_G - glucose, for meal-associated readings)
    meal_responses = [l["post_G"] - l["glucose"] for l in glucose_logs
                       if l.get("post_G") is not None and (l.get("carbs") or 0) > 0]

    # 4. Exercise effectiveness: does logged exercise blunt the glucose response?
    #    Compares the average response (post_G - glucose) for readings logged
    #    alongside >=20min exercise vs. readings with none.
    with_ex = [l["post_G"] - l["glucose"] for l in glucose_logs
               if l.get("post_G") is not None and (l.get("exercise_min") or 0) >= 20]
    without_ex = [l["post_G"] - l["glucose"] for l in glucose_logs
                  if l.get("post_G") is not None and (l.get("exercise_min") or 0) == 0]
    exercise_effectiveness = (round(_mean(with_ex) - _mean(without_ex), 1)
                               if with_ex and without_ex else None)

    # 5. Sleep effect: next-calendar-day mean glucose after a short-sleep night
    #    vs. after a 7h+ night.
    sleep_effect = None
    if sleep_logs and glucose_logs:
        short_nights = {l["timestamp"][:10] for l in sleep_logs if (l.get("hours") or 8) < 6}
        long_nights  = {l["timestamp"][:10] for l in sleep_logs if (l.get("hours") or 0) >= 7}
        g_by_day: Dict[str, list] = {}
        for l in glucose_logs:
            g_by_day.setdefault(l["timestamp"][:10], []).append(l["glucose"])
        def _next_day_mean(nights):
            vals = []
            for n in nights:
                nxt = (datetime.fromisoformat(n) + timedelta(days=1)).date().isoformat()
                vals.extend(g_by_day.get(nxt, []))
            return _mean(vals)
        short_mean, long_mean = _next_day_mean(short_nights), _next_day_mean(long_nights)
        if short_mean is not None and long_mean is not None:
            sleep_effect = round(short_mean - long_mean, 1)

    # 6. Weekly activity trend — logged activity duration this week vs. last week.
    weekly_activity_trend = None
    recent_cutoff = datetime.utcnow() - timedelta(days=7)
    prior_cutoff = datetime.utcnow() - timedelta(days=14)
    recent_minutes = sum(l.get("duration_min", 0) or 0 for l in activity_logs
                         if l.get("timestamp", "") >= recent_cutoff.isoformat())
    prior_minutes = sum(l.get("duration_min", 0) or 0 for l in activity_logs
                        if prior_cutoff.isoformat() <= l.get("timestamp", "") < recent_cutoff.isoformat())
    if prior_minutes > 0:
        weekly_activity_trend = round((recent_minutes - prior_minutes) / prior_minutes * 100, 1)

    # 8. Glucose variability — coefficient of variation (%), a standard clinical metric
    g_vals = [l["glucose"] for l in glucose_logs if l.get("glucose") is not None]
    g_mean, g_std = _mean(g_vals), _stdev(g_vals)
    glucose_cv = round(g_std / g_mean * 100, 1) if g_mean and g_std else None

    # 9-10. Average carbs / calories
    avg_carb_intake = _mean([l.get("carbs_g") for l in meal_logs])
    avg_calories = _mean([l.get("calories") for l in meal_logs])

    # 11. Average daily water intake
    avg_water_intake = None
    if water_logs:
        by_day: Dict[str, int] = {}
        for l in water_logs:
            d = l["timestamp"][:10]
            by_day[d] = by_day.get(d, 0) + (l.get("amount_ml") or 0)
        avg_water_intake = _mean(list(by_day.values()))

    # 12. Stress trend — recent vs. prior average logged stress_level
    stress_trend = None
    stress_vals = [(l["timestamp"], l["stress_level"]) for l in glucose_logs if l.get("stress_level")]
    stress_vals.extend((l["timestamp"], l["stress_level"])
                       for l in by_type.get("stress", []) if l.get("stress_level"))
    if len(stress_vals) >= 6:
        stress_vals.sort(key=lambda x: x[0])
        half = len(stress_vals) // 2
        prior_mean = _mean([v for _, v in stress_vals[:half]])
        recent_mean = _mean([v for _, v in stress_vals[half:]])
        if prior_mean and recent_mean:
            stress_trend = round(recent_mean - prior_mean, 2)

    # 13. Weekly TIR
    week_cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
    week_g = [l["glucose"] for l in glucose_logs if l["timestamp"] >= week_cutoff]
    weekly_tir = round(sum(1 for g in week_g if 70 <= g <= 180) / len(week_g) * 100, 1) if week_g else None

    # 14. Monthly HbA1c trend
    monthly_hba1c_trend = None
    if len(hba1c_logs) >= 2:
        monthly_hba1c_trend = round(hba1c_logs[-1]["hba1c"] - hba1c_logs[0]["hba1c"], 2)

    # 15. Weight trend
    weight_trend = None
    if len(weight_logs) >= 2:
        weight_trend = round(weight_logs[-1]["weight_kg"] - weight_logs[0]["weight_kg"], 1)

    features = {
        "avg_morning_glucose": _mean(morning_g),
        "avg_night_glucose": _mean(night_g),
        "avg_meal_response": _mean(meal_responses),
        "exercise_effectiveness": exercise_effectiveness,
        "sleep_effect_on_glucose": sleep_effect,
        "weekly_activity_trend_pct": weekly_activity_trend,
        "glucose_variability_cv_pct": glucose_cv,
        "avg_carb_intake_g": avg_carb_intake,
        "avg_calories": avg_calories,
        "avg_water_intake_ml": avg_water_intake,
        "stress_trend": stress_trend,
        "weekly_tir_pct": weekly_tir,
        "monthly_hba1c_trend": monthly_hba1c_trend,
        "weight_trend_kg": weight_trend,
    }
    days_of_data = len({entry["timestamp"][:10]
                        for entries in timeline["by_type"].values()
                        for entry in entries})
    if _DB_AVAILABLE:
        await _db.save_personal_features(patient_id, features, days_of_data)
    return {"features": features, "days_of_data": days_of_data,
            "computed_at": datetime.utcnow().isoformat()}


# ─────────────────────────────────────────────
# PYDANTIC MODELS  (unchanged — API contract preserved)
# ─────────────────────────────────────────────
class UserRegister(BaseModel):
    username:       str   = Field(..., min_length=3, max_length=50)
    email:          str   = Field(..., pattern=r".+@.+\..+")
    password:       str   = Field(..., min_length=6)
    full_name:      str   = Field("", max_length=100)

class PatientProfile(BaseModel):
    full_name:      Optional[str]   = None
    age:            Optional[int]   = Field(None, ge=18, le=100)
    gender:         Optional[str]   = None
    height_cm:      Optional[float] = None
    weight_kg:      Optional[float] = None
    bmi:            Optional[float] = None
    diabetes_years: Optional[int]   = None
    hba1c:          Optional[float] = None
    baseline_glucose: Optional[float] = None
    medications:    Optional[List[str]] = None
    activity_level: Optional[str]   = None  # sedentary/moderate/active
    goal:           Optional[str]   = None  # reversal/management/monitoring
    diabetes_risk:  Optional[int]   = Field(None, ge=0, le=2)
    family_history: Optional[bool]  = None

class GlucoseLog(BaseModel):
    glucose:     float = Field(..., ge=40, le=400)
    meal_type:   Optional[str] = None
    carbs:       float = Field(0, ge=0, le=300)
    sugar:       float = Field(0, ge=0, le=200)
    food:        Optional[str] = None
    steps:       int   = Field(0, ge=0)
    exercise_min:int   = Field(0, ge=0)
    heart_rate:  int   = Field(80, ge=40, le=220)
    stress_level:int   = Field(2, ge=1, le=5)
    notes:       Optional[str] = None
    timestamp:   Optional[datetime] = None

class MealLog(BaseModel):
    food_name:   str
    quantity:    Optional[float] = Field(None, gt=0, le=100)
    serving_unit:Optional[str] = None
    carbs_g:     Optional[float] = Field(None, ge=0, le=500)
    protein_g:   Optional[float] = Field(None, ge=0)
    fat_g:       Optional[float] = Field(None, ge=0)
    fiber_g:     Optional[float] = Field(None, ge=0)
    calories:    Optional[float] = Field(None, ge=0)
    meal_type:   str   = "meal"   # breakfast/lunch/dinner/snack
    gi_score:    Optional[float] = None
    notes:       Optional[str] = None

class ActivityLog(BaseModel):
    activity:    str
    duration_min:int   = Field(..., ge=0, le=480)
    steps:       int   = Field(0, ge=0)
    calories:    float = Field(0, ge=0)
    heart_rate_avg: int = Field(0, ge=0)
    intensity:   str   = "moderate"   # light/moderate/intense

class SleepLog(BaseModel):
    hours:       float = Field(..., ge=0, le=24)
    quality:     int   = Field(3, ge=1, le=5)   # 1=poor 5=excellent
    bed_time:    Optional[str] = None
    wake_time:   Optional[str] = None

class MedicationLog(BaseModel):
    medication:  str
    dose:        Optional[str] = None
    taken:       bool = True
    notes:       Optional[str] = None

class WeightLog(BaseModel):
    weight_kg:   float = Field(..., ge=20, le=300)
    bmi:         Optional[float] = None
    body_fat_pct:Optional[float] = None

class BloodPressureLog(BaseModel):
    systolic:    int   = Field(..., ge=60, le=250, description="Upper number e.g. 130")
    diastolic:   int   = Field(..., ge=40, le=150, description="Lower number e.g. 85")
    pulse:       int   = Field(80, ge=30, le=220)
    notes:       Optional[str] = None

class WaterLog(BaseModel):
    amount_ml:   int   = Field(..., ge=50, le=5000, description="Volume in ml e.g. 250")
    time_of_day: Optional[str] = None   # morning/afternoon/evening/night

class HbA1cLog(BaseModel):
    hba1c:       float = Field(..., ge=3.0, le=20.0, description="HbA1c percentage e.g. 7.2")
    lab_name:    Optional[str] = None
    test_date:   Optional[str] = None   # YYYY-MM-DD

class StressLog(BaseModel):
    level: str = Field(..., pattern="^(low|moderate|high)$")

class NutritionRequest(BaseModel):
    food_name: str = Field(..., min_length=1, max_length=160)
    quantity: float = Field(1, gt=0, le=100)
    serving_unit: Optional[str] = None

class PasswordChange(BaseModel):
    current_password: str
    new_password:     str = Field(..., min_length=6)

class PatientGoal(BaseModel):
    target_hba1c:    Optional[float] = Field(None, ge=4.0, le=12.0)
    target_weight:   Optional[float] = Field(None, ge=30.0, le=250.0)
    target_steps:    Optional[int]   = Field(None, ge=1000, le=50000)
    target_tir:      Optional[float] = Field(None, ge=0.0, le=100.0)
    target_date:     Optional[str]   = None   # YYYY-MM-DD deadline

class WhatIfRequest(BaseModel):
    patient_id:  str
    carbs:       float = Field(50, ge=0, le=300)
    exercise_min:int   = Field(0, ge=0, le=180)
    sleep_hours: float = Field(7, ge=3, le=12)
    stress_level:int   = Field(2, ge=1, le=5)
    current_glucose: Optional[float] = None

class CoachMessage(BaseModel):
    patient_id:      str
    message:         str = Field(..., max_length=500)
    include_context: bool = True

class MarkNotificationsRead(BaseModel):
    notification_ids: Optional[List[str]] = None  # None = mark all

class ResetPassword(BaseModel):
    """Additive — completes the forgot-password flow now that reset tokens
    are actually persisted. Does not change any existing endpoint."""
    token:        str
    new_password: str = Field(..., min_length=6)


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logging.basicConfig(level=logging.INFO)
    logger.info("BioMirror API v3 starting…")
    if _DB_AVAILABLE:
        await _db.init()
        h = await _db.health_check()
        logger.info(f"[DB] type={h.get('type','unknown')} status={h.get('status','?')}")
        if h.get('warning'):
            logger.warning(f"[DB] {h['warning']}")
        if not _db.ready:
            logger.warning("[DB] SQLAlchemy not installed — falling back to in-memory store. "
                            "Install with: pip install sqlalchemy aiosqlite --break-system-packages")
    else:
        logger.error("[DB] database.py could not be imported at all — persistence is completely disabled. "
                      "Check that backend/database.py exists and its dependencies are installed.")
    if GROQ_API_KEY:
        logger.info(f"GROQ_API_KEY configured — AI Coach uses Groq ({GROQ_MODEL})")
    elif ANTHROPIC_API_KEY:
        logger.info("GROQ_API_KEY not set — AI Coach uses Anthropic")
    else:
        logger.warning("No AI provider key set — AI Coach uses offline fallback")

    global AI_STATUS
    if _AI_BRIDGE_AVAILABLE:
        AI_STATUS = _ai.initialize()
        logger.info(f"[AI] feature_engineering={AI_STATUS['feature_engineering']} "
                    f"reversal_engine={AI_STATUS['reversal_engine']} "
                    f"shap={AI_STATUS['shap']} food_database={AI_STATUS['food_database']}")
        if AI_STATUS['pinn']:
            logger.info(f"[AI] PINN model loaded: {AI_STATUS['pinn_status']['model_version']} "
                        f"(trained {AI_STATUS['pinn_status']['trained_at']})")
        else:
            logger.warning(f"[AI] PINN model not loaded ({AI_STATUS['pinn_status'].get('load_error')}) "
                            f"— /forecast will use the Bergman engine only")
    else:
        AI_STATUS = {'feature_engineering': False, 'pinn': False, 'pinn_status': {'available': False},
                     'reversal_engine': False, 'shap': False, 'food_database': False}
        logger.error("[AI] ai_bridge.py could not be imported — all Phase 2 AI features disabled, "
                     "falling back to Phase 1 inline calculations everywhere")

    logger.info("BioMirror API ready. Docs: /docs")


@app.on_event("shutdown")
async def shutdown():
    if _DB_AVAILABLE and _db:
        await _db.close()


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────
@app.post("/auth/register", tags=["Auth"])
async def register(user: UserRegister):
    if await _db.get_user_by_username(user.username) or await _db.get_user_by_email(user.email):
        raise HTTPException(400, "Username or email already registered")
    patient_id = "P-" + hashlib.md5(user.username.encode()).hexdigest()[:8].upper()
    verify_token = uuid.uuid4().hex
    profile = {
        "username":    user.username,
        "email":       user.email,
        "password":    hash_pw(user.password),
        "full_name":   user.full_name or user.username,
        "patient_id":  patient_id,
        "role":        "patient",
        "bmi":         27.0,
        "baseline_glucose": 140.0,
        "diabetes_risk": 1,
        "profile_complete": False,
        "email_verification_token": verify_token,
    }
    created = await _db.create_user(profile)
    await save_twin(patient_id, build_initial_twin(patient_id, created))
    await push_notification(patient_id, "Welcome to BioMirror! 🧬",
                      "Your Digital Twin has been initialized. Complete your profile to personalize it.")
    refresh = make_refresh_token(user.username)
    await _db.save_refresh_token(user.username, refresh, datetime.utcnow() + timedelta(days=REFRESH_TOKEN_DAYS))
    logger.info(f"[DEV] Email verification token for {user.email}: {verify_token} "
                f"(no SMTP configured — verify via GET /auth/verify-email/{{token}})")
    return {
        "access_token":  make_access_token(user.username),
        "refresh_token": refresh,
        "token_type":    "bearer",
        "patient_id":    patient_id,
        "full_name":     created["full_name"],
        "profile_complete": False,
    }


@app.post("/auth/login", tags=["Auth"])
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = await _db.get_user_by_username(form.username)
    if not user:
        user = await _db.get_user_by_email(form.username)
    if not user or not verify_pw(form.password, user["password"]):
        raise HTTPException(401, "Invalid credentials")
    refresh = make_refresh_token(user["username"])
    await _db.save_refresh_token(user["username"], refresh, datetime.utcnow() + timedelta(days=REFRESH_TOKEN_DAYS))
    return {
        "access_token":  make_access_token(user["username"]),
        "refresh_token": refresh,
        "token_type":    "bearer",
        "patient_id":    user["patient_id"],
        "full_name":     user.get("full_name", user["username"]),
        "profile_complete": user.get("profile_complete", False),
    }


@app.post("/auth/refresh", tags=["Auth"])
async def refresh_token_route(body: dict):
    token = body.get("refresh_token", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(401, "Not a refresh token")
        if not await _db.is_refresh_token_valid(token):
            raise HTTPException(401, "Refresh token has been revoked or is unknown")
        username = payload.get("sub")
        user = await _db.get_user_by_username(username) if username else None
        if not user:
            raise HTTPException(401, "User not found")
        return {
            "access_token": make_access_token(username),
            "token_type":   "bearer",
            "patient_id":   user["patient_id"],
        }
    except JWTError:
        raise HTTPException(401, "Invalid or expired refresh token")


@app.post("/auth/logout", tags=["Auth"])
async def logout(user=Depends(get_current_user)):
    # Revoke all persisted refresh tokens for this user — real logout now,
    # not just client-side token deletion.
    await _db.revoke_all_user_tokens(user["username"])
    return {"message": "Logged out successfully"}


@app.post("/auth/forgot-password", tags=["Auth"])
async def forgot_password(body: dict):
    email = body.get("email", "")
    user  = await _db.get_user_by_email(email)
    if user:
        token = await _db.create_reset_token(email)
        # No SMTP configured by default (matches previous behaviour) — log
        # the token so it's usable in development. In production, send it
        # via email instead of logging it.
        logger.info(f"[DEV] Password reset token for {email}: {token} "
                    f"(use POST /auth/reset-password with this token)")
    # Always return success to prevent user enumeration
    return {"message": "If that email is registered, a reset link has been sent."}


@app.post("/auth/reset-password", tags=["Auth"])
async def reset_password(body: ResetPassword):
    """Additive endpoint completing the forgot-password flow now that reset
    tokens are actually persisted (see database.py PasswordResetTokenRow)."""
    email = await _db.consume_reset_token(body.token)
    if not email:
        raise HTTPException(400, "Invalid or expired reset token")
    user = await _db.get_user_by_email(email)
    if not user:
        raise HTTPException(404, "User not found")
    await _db.update_user(user["username"], {"password": hash_pw(body.new_password),
                                              "password_changed_at": datetime.utcnow()})
    await _db.revoke_all_user_tokens(user["username"])
    return {"success": True, "message": "Password has been reset. Please log in again."}


@app.get("/auth/verify-email/{token}", tags=["Auth"])
async def verify_email(token: str):
    """Additive endpoint — confirms the email verification token issued at
    registration. Non-blocking: login is not gated on this."""
    ok = await _db.verify_email_token(token)
    if not ok:
        raise HTTPException(400, "Invalid or expired verification token")
    return {"success": True, "message": "Email verified successfully"}


# ─────────────────────────────────────────────
# PROFILE ROUTES
# ─────────────────────────────────────────────
@app.get("/profile/{patient_id}", tags=["Profile"])
async def get_profile(patient_id: str, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id and user.get("role") != "admin":
        raise HTTPException(403, "Access denied")
    safe = {k: v for k, v in user.items() if k != "password"}
    return safe


@app.put("/profile/{patient_id}", tags=["Profile"])
async def update_profile(patient_id: str, profile: PatientProfile,
                         user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    uname = user["username"]
    updates = profile.dict(exclude_none=True)

    # Auto-calculate BMI if height+weight provided
    if "height_cm" in updates and "weight_kg" in updates:
        h_m = updates["height_cm"] / 100
        updates["bmi"] = round(updates["weight_kg"] / (h_m ** 2), 1)
    elif "weight_kg" in updates and user.get("height_cm"):
        h_m = user["height_cm"] / 100
        updates["bmi"] = round(updates["weight_kg"] / (h_m ** 2), 1)

    updates["profile_complete"] = True
    updated = await _db.update_user(uname, updates)

    # Re-initialize twin with updated profile
    bmi = updated.get("bmi", 27.0)
    p3  = max(1e-5, 7e-5 - (bmi - 22) * 2e-6)
    state = await load_twin(patient_id)
    if state:
        state["p3"]       = p3
        state["p3_index"] = round(p3 / 5e-5 * 100, 1)
        bg = updates.get("baseline_glucose") or updated.get("baseline_glucose", 140.0)
        state["G"] = bg
        await save_twin(patient_id, state)

    await push_notification(patient_id, "Profile Updated ✓",
                      "Your Digital Twin has been recalibrated with your new profile data.")
    safe = {k: v for k, v in updated.items() if k != "password"}
    return {"success": True, "profile": safe}


# ─────────────────────────────────────────────
# HEALTH LOGGING ROUTES
# ─────────────────────────────────────────────
@app.get("/foods", tags=["Nutrition"])
async def get_food_catalog(query: str = Query("", max_length=80),
                           category: str = Query("", max_length=40),
                           user=Depends(get_current_user)):
    if not list_food_catalog:
        raise HTTPException(503, "Food database is unavailable")
    return {"foods": list_food_catalog(query, category)}


@app.post("/foods/nutrition", tags=["Nutrition"])
async def get_food_nutrition(request: NutritionRequest,
                             user=Depends(get_current_user)):
    if not calculate_nutrition:
        raise HTTPException(503, "Nutrition calculator is unavailable")
    return calculate_nutrition(request.food_name, request.quantity, request.serving_unit)


async def add_log(patient_id: str, log_type: str, data: dict):
    return await _db.add_log(patient_id, log_type, data)


@app.post("/logs/{patient_id}/glucose", tags=["Logging"])
async def log_glucose(patient_id: str, log: GlucoseLog, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    state = await load_twin(patient_id)
    if not state:
        raise HTTPException(404, "Digital Twin not found")

    eng = BergmanEngine(p3=state["p3"])
    G_prev = state["G"]

    recent_stress = await _db.get_logs(patient_id, "stress", limit=1)
    effective_stress = (recent_stress[-1].get("stress_level", log.stress_level)
                        if recent_stress and log.stress_level == 2 else log.stress_level)

    # Update twin with new anchor reading
    state["G"] = log.glucose
    state["I"] = min(100, max(0, 7 + (log.glucose - 90) * 0.1))

    # Queue meal if carbs logged
    if log.carbs > 0:
        state["meal_queue"].append({"carbs": log.carbs, "start": state["sim_time"], "elapsed": 0.0})

    # Exercise adjusts p3
    if log.exercise_min > 0:
        state["p3"] = min(state["p3"] * (1 + log.exercise_min * 0.005), 8e-5)
        state["p3_index"] = round(state["p3"] / 5e-5 * 100, 1)

    # ODE step
    meal_in = sum(eng.meal_bolus(m["carbs"], m["elapsed"]) for m in state["meal_queue"])
    for m in state["meal_queue"]:
        m["elapsed"] += eng.DT
    G_new, X_new = eng.step(state["G"], state["X"], state["I"], meal_in)
    state["G"] = G_new; state["X"] = X_new
    state["sim_time"] += eng.DT
    state["history"].append({"t": state["sim_time"], "G": G_new})
    if len(state["history"]) > 288:
        state["history"].pop(0)
    state["last_updated"] = datetime.utcnow().isoformat()
    state["last_event"] = {"type": "glucose", "value": log.glucose,
                            "timestamp": datetime.utcnow().isoformat()}
    await save_twin(patient_id, state)

    entry = await add_log(patient_id, "glucose", {
        "glucose": log.glucose, "meal_type": log.meal_type,
        "carbs": log.carbs, "food": log.food,
        "steps": log.steps, "exercise_min": log.exercise_min,
        "heart_rate": log.heart_rate, "stress_level": effective_stress,
        "notes": log.notes, "post_G": round(G_new, 1),
    })

    # Phase 2: opportunistically update retrospective evaluation metrics
    # (matches recent forecast predictions against this and other actuals)
    await evaluate_recent_predictions(patient_id)

    # Phase 3 Part C: every logged data point grows the personal timeline
    try:
        await compute_personal_features(patient_id)
    except Exception as e:
        logger.warning(f"[Personal Learning] feature recompute after glucose log failed for {patient_id}: {e}")

    metrics = compute_metrics(state)

    # High glucose notification
    if log.glucose > 200:
        await push_notification(patient_id, "⚠ High Glucose Alert",
                          f"Glucose reading {log.glucose} mg/dL is above safe range. "
                          "Consider a 15-min walk to bring it down.", ntype="alert")

    await broadcast(patient_id, {"event": "twin_update", **metrics})
    return {"success": True, "log_id": entry["id"],
            "glucose_change": round(G_new - G_prev, 1),
            "current_state": metrics,
            "precision_nudges": generate_nudges(metrics)}


@app.post("/logs/{patient_id}/meal", tags=["Logging"])
async def log_meal(patient_id: str, log: MealLog, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    state = await load_twin(patient_id)
    current_glucose = state["G"] if state else 140.0
    p3_index = state["p3_index"] if state else 68.0
    hour = datetime.utcnow().hour

    nutrition = None
    log_data = log.dict(exclude_none=True)
    prediction_food = log.food_name
    if log.quantity is not None and calculate_nutrition:
        nutrition = calculate_nutrition(log.food_name, log.quantity, log.serving_unit)
        if nutrition["available"]:
            prediction_food = nutrition["canonicalName"]
            unit_label = nutrition["servingUnit"]
            if nutrition["quantity"] != 1:
                unit_label += "s"
            log_data["food_name"] = (
                f"{nutrition['quantity']:g} {unit_label} "
                f"{nutrition['canonicalName'].replace('_', ' ').title()}"
            )
            log_data.update({
                "carbs_g": nutrition["carbohydrates"],
                "protein_g": nutrition["protein"],
                "fat_g": nutrition["fat"],
                "fiber_g": nutrition["fiber"],
                "calories": nutrition["calories"],
                "gi_score": nutrition["glycemicIndex"] / 100,
            })
        else:
            log_data.update({"carbs_g": None, "protein_g": None, "fat_g": None,
                             "fiber_g": None, "calories": None, "gi_score": None})
        serving = f"{nutrition['quantity']:g} {nutrition['servingUnit']}"
        log_data["notes"] = "; ".join(filter(None, [
            log.notes,
            f"Serving: {serving}",
            f"Nutrition: {nutrition['confidence']} ({nutrition['source']})",
        ]))

    carbs_g = float(log_data.get("carbs_g") or 0)
    prediction = None
    if _AI_BRIDGE_AVAILABLE and carbs_g > 0:
        prediction = _ai.predict_meal_from_carbs(prediction_food, carbs_g,
                                                   current_glucose, p3_index, hour)

    if prediction:
        log_data["glycemic_load"] = prediction["glycemic_load"]
        log_data["predicted_peak_glucose"] = prediction["peak_glucose"]
        log_data["spike_risk"] = prediction["spike_risk"]
        log_data["time_to_peak_min"] = prediction["time_to_peak_min"]
        if not log_data.get("gi_score"):
            log_data["gi_score"] = prediction["glycemic_index"]
    entry = await add_log(patient_id, "meal", log_data)

    # Update twin meal queue
    if state and carbs_g > 0:
        state["meal_queue"].append({"carbs": carbs_g, "start": state["sim_time"], "elapsed": 0.0})
        await save_twin(patient_id, state)

    try:
        await compute_personal_features(patient_id)
    except Exception as e:
        logger.warning(f"[Personal Learning] feature recompute after meal log failed for {patient_id}: {e}")

    if prediction:
        if prediction["spike_risk"] == "HIGH":
            await push_notification(patient_id, "⚠ High Glucose Spike Predicted",
                f"{log.food_name} is predicted to spike glucose to "
                f"{prediction['peak_glucose']} mg/dL. "
                f"{prediction.get('tip', '')}", ntype="alert")
        return {"success": True, "log_id": entry["id"], "nutrition": nutrition,
                "estimated_glucose_impact": f"+{round(prediction['peak_glucose']-current_glucose,0):.0f} mg/dL peak",
                "prediction": prediction}
    return {"success": True, "log_id": entry["id"], "nutrition": nutrition,
            "estimated_glucose_impact": None}


@app.post("/logs/{patient_id}/activity", tags=["Logging"])
async def log_activity(patient_id: str, log: ActivityLog, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    entry = await add_log(patient_id, "activity", log.dict())
    state = await load_twin(patient_id)
    if state:
        boost = log.duration_min * 0.005
        state["p3"] = min(state["p3"] * (1 + boost), 8e-5)
        state["p3_index"] = round(state["p3"] / 5e-5 * 100, 1)
        await save_twin(patient_id, state)
    try:
        await compute_personal_features(patient_id)
    except Exception as e:
        logger.warning(f"[Personal Learning] feature recompute after activity log failed for {patient_id}: {e}")
    return {"success": True, "log_id": entry["id"],
            "sensitivity_boost": f"+{round(log.duration_min * 0.5, 1)}% insulin sensitivity"}


@app.post("/logs/{patient_id}/sleep", tags=["Logging"])
async def log_sleep(patient_id: str, log: SleepLog, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    entry = await add_log(patient_id, "sleep", log.dict())
    state = await load_twin(patient_id)
    if state and log.hours < 6:
        # Sleep deficit increases insulin resistance
        state["p3"] = max(state["p3"] * 0.92, 1e-5)
        state["p3_index"] = round(state["p3"] / 5e-5 * 100, 1)
        await save_twin(patient_id, state)
        await push_notification(patient_id, "Sleep Deficit Detected 💤",
                          f"Only {log.hours}h sleep reduces sensitivity by ~{round((6-log.hours)*3)}%. "
                          "Aim for 7-9h for optimal glucose control.")
    try:
        await compute_personal_features(patient_id)
    except Exception as e:
        logger.warning(f"[Personal Learning] feature recompute after sleep log failed for {patient_id}: {e}")
    return {"success": True, "log_id": entry["id"]}


@app.post("/logs/{patient_id}/stress", tags=["Logging"])
async def log_stress(patient_id: str, log: StressLog, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    normalized = {"low": 1, "moderate": 3, "high": 5}[log.level]
    entry = await add_log(patient_id, "stress", {
        "level": log.level, "stress_level": normalized,
    })
    state = await load_twin(patient_id)
    if state:
        state["last_event"] = {
            "type": "stress", "level": log.level,
            "stress_level": normalized, "timestamp": datetime.utcnow().isoformat(),
        }
        await save_twin(patient_id, state)
        await broadcast(patient_id, {"event": "stress_logged", "level": log.level,
                                     "stress_level": normalized})
    try:
        await compute_personal_features(patient_id)
    except Exception as e:
        logger.warning(f"[Personal Learning] feature recompute after stress log failed for {patient_id}: {e}")
    return {"success": True, "log_id": entry["id"], "level": log.level,
            "stress_level": normalized}


@app.post("/logs/{patient_id}/medication", tags=["Logging"])
async def log_medication(patient_id: str, log: MedicationLog, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    entry = await add_log(patient_id, "medication", log.dict())
    try:
        await compute_personal_features(patient_id)
    except Exception as e:
        logger.warning(f"[Personal Learning] feature recompute after medication log failed for {patient_id}: {e}")
    return {"success": True, "log_id": entry["id"]}


@app.post("/logs/{patient_id}/weight", tags=["Logging"])
async def log_weight(patient_id: str, log: WeightLog, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    uname = user["username"]
    # Auto-compute BMI
    h_cm = user.get("height_cm")
    if h_cm and not log.bmi:
        log = log.copy(update={"bmi": round(log.weight_kg / (h_cm / 100) ** 2, 1)})
    updates = {"weight_kg": log.weight_kg}
    if log.bmi:
        updates["bmi"] = log.bmi
    await _db.update_user(uname, updates)
    state = await load_twin(patient_id)
    if state and log.bmi:
        state["p3"] = max(1e-5, 7e-5 - (log.bmi - 22) * 2e-6)
        state["p3_index"] = round(state["p3"] / 5e-5 * 100, 1)
        await save_twin(patient_id, state)
    entry = await add_log(patient_id, "weight", log.dict())
    try:
        await compute_personal_features(patient_id)
    except Exception as e:
        logger.warning(f"[Personal Learning] feature recompute after weight log failed for {patient_id}: {e}")
    return {"success": True, "log_id": entry["id"], "bmi": log.bmi}


# ─────────────────────────────────────────────
# HISTORY ROUTES
# ─────────────────────────────────────────────
@app.get("/history/{patient_id}", tags=["History"])
async def get_history(
    patient_id: str,
    log_type: Optional[str] = Query(None, description="glucose/meal/activity/sleep/medication/weight"),
    limit:    int = Query(50, ge=1, le=500),
    days:     int = Query(30, ge=1, le=365),
    user=Depends(get_current_user),
):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    cutoff = datetime.utcnow() - timedelta(days=days)
    filtered = await _db.get_logs(patient_id, log_type, limit=limit, since=cutoff)
    return {"logs": filtered, "total": len(filtered),
            "days": days, "type": log_type or "all"}


@app.get("/history/{patient_id}/glucose", tags=["History"])
async def get_glucose_history(patient_id: str, days: int = Query(7),
                              user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    cutoff = datetime.utcnow() - timedelta(days=days)
    glucose_logs = await _db.get_logs(patient_id, "glucose", limit=2000, since=cutoff)
    values = [l["glucose"] for l in glucose_logs if l.get("glucose") is not None]
    if not values:
        # Return demo data from twin history
        state = await load_twin(patient_id) or {}
        hist  = state.get("history", [])
        return {"readings": [{"timestamp": datetime.utcnow().isoformat(),
                               "glucose": h["G"], "source": "twin"}
                              for h in hist[-48:]], "stats": {}}
    mean_g = round(sum(values) / len(values), 1)
    tir    = round(sum(1 for g in values if 70 <= g <= 180) / len(values) * 100, 1)
    hba1c  = round((mean_g + 46.7) / 28.7, 2)
    return {
        "readings":   glucose_logs,
        "stats": {
            "mean":   mean_g, "min": min(values), "max": max(values),
            "tir_pct": tir, "hba1c_estimate": hba1c, "count": len(values),
            "hypo": sum(1 for g in values if g < 70),
            "hyper": sum(1 for g in values if g > 180),
        },
    }


@app.get("/history/{patient_id}/summary", tags=["History"])
async def get_summary(patient_id: str, user=Depends(get_current_user)):
    """Weekly summary across all log types."""
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    cutoff = datetime.utcnow() - timedelta(days=7)
    week   = await _db.get_logs(patient_id, None, limit=5000, since=cutoff)
    by_type = {}
    for l in week:
        by_type.setdefault(l["type"], []).append(l)
    summary = {t: len(v) for t, v in by_type.items()}
    g_vals  = [l["glucose"] for l in by_type.get("glucose", []) if l.get("glucose") is not None]
    return {
        "patient_id":   patient_id,
        "period_days":  7,
        "logs_by_type": summary,
        "total_logs":   len(week),
        "glucose_mean": round(sum(g_vals)/len(g_vals), 1) if g_vals else None,
        "glucose_tir":  round(sum(1 for g in g_vals if 70 <= g <= 180)/len(g_vals)*100, 1) if g_vals else None,
    }


# ─────────────────────────────────────────────
# DIGITAL TWIN ROUTES
# ─────────────────────────────────────────────
async def compute_twin_confidence_calibration(patient_id: str) -> dict:
    """
    Phase 3 Part G — "the Digital Twin should improve as more user data
    becomes available." Confidence grows with days_of_data (more personal
    history = more trustworthy personalization); calibration is derived
    from evaluation_metrics_history (Phase 2's retrospective MAE tracking)
    — a twin whose past predictions were close to actual outcomes gets a
    higher calibration score than one that's been consistently off.
    """
    pf = await _db.get_personal_features(patient_id) if _DB_AVAILABLE else None
    days_of_data = (pf or {}).get("days_of_data", 0)
    # Confidence ramps from 0.3 (brand new patient) toward 0.95 as personal
    # history accumulates, saturating around 60 days — a deliberately
    # conservative curve so a 3-day-old account isn't shown as "highly confident."
    confidence = round(min(0.95, 0.3 + 0.65 * min(days_of_data, 60) / 60), 3)

    eval_metrics = await _db.get_latest_evaluation_metrics(patient_id) if _DB_AVAILABLE else None
    calibration = None
    if eval_metrics and eval_metrics.get("mae") is not None:
        # MAE of 0 -> calibration 1.0; MAE of 40+ mg/dL -> calibration floors at 0.1
        calibration = round(max(0.1, 1.0 - min(eval_metrics["mae"], 40) / 40), 3)

    return {"confidence": confidence, "calibration": calibration,
            "days_of_data": days_of_data}


@app.get("/twin/{patient_id}", tags=["Digital Twin"])
async def get_twin(patient_id: str, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id and user.get("role") != "admin":
        raise HTTPException(403, "Access denied")
    state = await load_twin(patient_id)
    if not state:
        raise HTTPException(404, f"Digital Twin not found for {patient_id}")
    metrics = compute_metrics(state)
    
    if _DB_AVAILABLE:
        today_start = datetime.combine(datetime.utcnow().date(), datetime.min.time())
        today_water_logs = await _db.get_logs(patient_id, "water", limit=200, since=today_start)
        metrics["water_today_ml"] = sum(l.get("amount_ml", 0) or 0 for l in today_water_logs)
    # Phase 2: sync the real reversal-engine snapshot into the twin state
    # itself (not just the response) — persisted via save_twin() so the
    # twin and the reversal engine stay in sync across restarts too.
    ctx = await build_personalized_context(patient_id, user, state)   # Phase 3: no more hardcoded steps
    if _AI_BRIDGE_AVAILABLE:
        rev = _ai.reversal_quick(glucose=state["G"], p3_index=state["p3_index"],
                                  tir=metrics.get("tir_pct", 63), steps=ctx["steps"],
                                  bmi=user.get('bmi') or 27.0)
        if rev:
            state["reversal_snapshot"] = rev
            await save_twin(patient_id, state)
            metrics["reversal_engine"] = rev

    # Phase 3 Part G: twin confidence/calibration + daily memory snapshot
    conf_cal = await compute_twin_confidence_calibration(patient_id)
    metrics["twin_confidence"] = conf_cal["confidence"]
    metrics["twin_calibration"] = conf_cal["calibration"]
    if _DB_AVAILABLE and not await _db.has_twin_snapshot_today(patient_id, "daily"):
        await _db.add_twin_snapshot(patient_id, "daily", {
            "G": state["G"], "p3_index": state["p3_index"],
            "reversal_score": metrics.get("reversal_score"), "tir_pct": metrics.get("tir_pct"),
            "confidence": conf_cal["confidence"], "calibration": conf_cal["calibration"],
        })

    # Phase 3 Part I: a fast "today's personalized insight" for the
    # dashboard, built from the already-computed/cached PersonalFeaturesRow
    # (not re-running the full correlation analysis on every dashboard
    # poll — that stays in /recommendations, which the frontend calls
    # less frequently).
    personal_insight = None
    if _DB_AVAILABLE:
        pf_row = await _db.get_personal_features(patient_id)
        if pf_row and pf_row.get("days_of_data", 0) >= 5:
            f = pf_row["features"]
            if f.get("sleep_effect_on_glucose") and f["sleep_effect_on_glucose"] > 5:
                personal_insight = (f"Short sleep nights have cost you about "
                                     f"{f['sleep_effect_on_glucose']} mg/dL the next day, based on your own history.")
            elif f.get("exercise_effectiveness") and f["exercise_effectiveness"] < -3:
                personal_insight = (f"Exercise measurably helps you — readings after 20+ min activity "
                                     f"run {abs(f['exercise_effectiveness'])} mg/dL lower in your data.")
            elif f.get("weekly_tir_pct") is not None:
                personal_insight = f"Your time-in-range this week is {f['weekly_tir_pct']}%."
        elif not pf_row:
            personal_insight = "Keep logging — personalized insights unlock after about 5 days of data."

    return {
        "patient_id": patient_id,
        "bergman_state": {
            "G": state["G"], "X": state["X"],
            "p3": state["p3"], "p3_index": state["p3_index"],
            "sim_time_min": state["sim_time"],
        },
        "metrics": metrics,
        "nudges":  generate_nudges(metrics),
        "personal_insight": personal_insight,
        "history_length": len(state.get("history", [])),
        "history_24h": state.get("history", [])[-288:],
    }


@app.post("/twin/{patient_id}/log", tags=["Digital Twin"])
async def twin_log(patient_id: str, log: GlucoseLog, user=Depends(get_current_user)):
    """Alias for /logs/{patient_id}/glucose — kept for backwards compatibility."""
    return await log_glucose(patient_id, log, user)


@app.get("/twin/{patient_id}/memory", tags=["Digital Twin"])
async def twin_memory(patient_id: str, user=Depends(get_current_user)):
    """
    Phase 3 Part G — Digital Twin memory: daily + weekly historical twin
    state, prediction/reversal/simulation history, and confidence/
    calibration, all in one place. Additive endpoint — doesn't change
    /twin's existing response shape.
    """
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    daily = await _db.get_twin_snapshots(patient_id, period="daily", limit=90)
    weekly_daily_pool = await _db.get_twin_snapshots(patient_id, period="daily", limit=400)
    # Weekly rollup computed from daily snapshots (no separate weekly-write
    # path needed — one daily snapshot per day already gives us everything
    # a weekly aggregate needs).
    weekly: List[dict] = []
    by_week: Dict[str, list] = {}
    for d in weekly_daily_pool:
        try:
            wk = datetime.fromisoformat(d["timestamp"]).strftime("%Y-W%W")
        except (ValueError, TypeError):
            continue
        by_week.setdefault(wk, []).append(d)
    for wk, snaps in sorted(by_week.items())[-12:]:
        vals = lambda k: [s[k] for s in snaps if s.get(k) is not None]
        weekly.append({
            "week": wk, "n_days": len(snaps),
            "avg_G": _mean(vals("G")), "avg_reversal_score": _mean(vals("reversal_score")),
            "avg_tir_pct": _mean(vals("tir_pct")), "avg_confidence": _mean(vals("confidence")),
        })

    predictions = await _db.get_predictions(patient_id, limit=30, source="forecast")
    simulations = await _db.get_predictions(patient_id, limit=30, source="simulate")
    reversal_hist = await _db.get_reversal_history(patient_id, limit=30)
    conf_cal = await compute_twin_confidence_calibration(patient_id)

    return {
        "patient_id": patient_id,
        "daily_snapshots": daily,
        "weekly_snapshots": weekly,
        "prediction_history": predictions,
        "simulation_history": simulations,
        "reversal_history": reversal_hist,
        "twin_confidence": conf_cal["confidence"],
        "twin_calibration": conf_cal["calibration"],
        "days_of_personal_data": conf_cal["days_of_data"],
    }


# ─────────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────────
@app.get("/forecast/{patient_id}", tags=["Forecast"])
async def get_forecast(patient_id: str, hours: int = Query(24, ge=1, le=72),
                       user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    state = await load_twin(patient_id)
    if not state:
        raise HTTPException(404, "Patient twin not found")
    eng = BergmanEngine(p3=state["p3"])
    G0, X0 = state["G"], state["X"]
    N = hours * 12

    # Long-horizon physics-based trajectory (Bergman ODE). This part is
    # unchanged from Phase 1: the PINN checkpoint's prediction horizon is
    # 6 steps x 5min = 30 minutes (see ai/pinn_model.py HyperParams), so it
    # cannot replace a 24-72h simulation — it's used below for the
    # near-term prediction instead, which is what it's actually designed for.
    traj = []
    G, X = G0, X0
    for k in range(N):
        G, X = eng.step(G, X, 7 + max(0, (G - 90) * 0.08), 0)
        traj.append(round(G, 1))

    meal_times = {7: 50, 13: 65, 19: 55}
    forecast = traj.copy()
    for hour_idx, carbs in meal_times.items():
        start = hour_idx * 12
        for s in range(start, min(start + 48, N)):
            elapsed = (s - start) * 5
            boost = eng.meal_bolus(carbs, elapsed, peak=45)
            forecast[s] = round(min(400, forecast[s] + boost * 5), 1)

    tir = round(sum(1 for g in forecast if 70 <= g <= 180) / N * 100, 1)

    # ── Phase 2: near-term (next 30 min) prediction via the real PINN
    # model where available, real feature engineering, and real SHAP —
    # replacing the previous hardcoded "shap" formulas. Falls back to a
    # Bergman-derived short-term estimate + analytical SHAP whenever the
    # PINN is unavailable or its output fails the plausibility check in
    # ai_bridge.py (see that file's docstring for why/when that happens).
    glucose_hist = [h["G"] for h in state.get("history", [])]
    ctx = await build_personalized_context(patient_id, user, state)   # Phase 3 Part E
    engineered = {}
    if _AI_BRIDGE_AVAILABLE:
        engineered = _ai.compute_live_features(glucose_hist, {**ctx, 'glucose': G0})

    pinn_result = None
    if _AI_BRIDGE_AVAILABLE:
        pinn_result = _ai.PINN.predict(glucose_hist, {
            'glucose': G0, 'carbs': ctx['carbs'], 'steps': ctx['steps'],
            'exercise_min': ctx['exercise_min'], 'heart_rate': ctx['heart_rate'],
            'bmi': ctx['bmi'], 'X': X0, 'I': state.get('I', 12.0),
            'hour_sin': math.sin(2 * math.pi * datetime.utcnow().hour / 24),
            'hour_cos': math.cos(2 * math.pi * datetime.utcnow().hour / 24),
            'glycemic_load': engineered.get('glycemic_load_score', 20.0),
            'stress_proxy': ctx['stress_level'] / 5,
        })

    if pinn_result:
        model_status  = "pinn"
        near_term     = pinn_result['predicted_glucose_mg_dl']
        confidence    = pinn_result['confidence']
        uncertainty   = pinn_result['uncertainty_std']
        inference_ms  = pinn_result['inference_time_ms']
        model_version = pinn_result['model_version']
    else:
        model_status  = "bergman_fallback" if (_AI_BRIDGE_AVAILABLE and AI_STATUS.get('pinn')) else "pinn_unavailable"
        near_term     = forecast[:6]
        confidence    = 0.55   # honest, lower confidence for the non-ML physics fallback
        uncertainty   = None
        inference_ms  = None
        model_version = "bergman_v1"

    shap_result = None
    if _AI_BRIDGE_AVAILABLE:
        latest_sleep = await _db.get_logs(patient_id, "sleep", limit=1)
        sleep_quality = ((latest_sleep[-1].get("quality", 3) / 5)
                         if latest_sleep else 0.7)
        shap_result = _ai.explain_prediction(
            glucose=G0, meal_carbs=ctx["carbs"], steps=ctx["steps"],
            sleep_quality=sleep_quality, stress=ctx["stress_level"])
    shap_out = shap_result if shap_result else {
        "meal_carbohydrates": round(0.38 + (G0-140)*0.001, 3),
        "previous_glucose":   round(0.26 + (G0-140)*0.0005, 3),
        "insulin_sensitivity": round(-0.22 - (state['p3_index']-50)*0.002, 3),
        "activity":           -0.18,
        "sleep_quality":      -0.12,
        "stress_cortisol":    0.09,
    }

    result = {
        "patient_id":  patient_id,
        "forecast_h":  hours,
        "baseline":    traj,
        "with_meals":  forecast,
        "labels":      [f"{k*5//60:02d}:{k*5%60:02d}" for k in range(N)],
        "summary": {
            "current_glucose": round(G0, 1),
            "tir_forecast":    tir,
            "peak_forecast":   max(forecast),
            "spike_risk":      "HIGH" if max(forecast) > 200 else
                               "MEDIUM" if max(forecast) > 160 else "LOW",
            "p3_index":        state["p3_index"],
        },
        "shap": shap_out,
        # Phase 2 additions — additive only, nothing above changed shape
        "near_term_prediction": {
            "predicted_glucose_mg_dl": [round(v, 1) for v in near_term],
            "horizon_minutes": 30,
            "model_status":    model_status,
            "model_version":   model_version,
            "confidence":      confidence,
            "uncertainty_std": uncertainty,
            "inference_time_ms": inference_ms,
        },
        "engineered_features": engineered,
    }

    if _DB_AVAILABLE:
        await _db.add_prediction(patient_id, {
            "source": "forecast",
            "inputs": {"hours": hours, "current_glucose": G0, "p3_index": state["p3_index"]},
            "engineered_features": engineered,
            "predicted_glucose": [round(v, 1) for v in near_term],
            "peak_glucose": max(forecast),
            "tir_pct": tir,
            "confidence": confidence,
            "uncertainty_std": uncertainty,
            "inference_time_ms": inference_ms,
            "model_version": model_version,
            "model_status": model_status,
            "shap_explanation": shap_out,
        })

    return result


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
@app.get("/metrics/{patient_id}", tags=["Metrics"])
async def get_metrics(patient_id: str, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    state = await load_twin(patient_id)
    if not state:
        raise HTTPException(404, "Patient twin not found")
    m = compute_metrics(state)
    G = state["G"]

    # Phase 2: derived metrics come from the real ai/feature_engineering.py
    # pipeline. Phase 3: the context fed into it is now this patient's own
    # recent data (build_personalized_context), not fixed constants.
    ctx = await build_personalized_context(patient_id, user, state)
    engineered = _ai.compute_live_features(
        [h["G"] for h in state.get("history", [])], {**ctx, 'glucose': G}
    ) if _AI_BRIDGE_AVAILABLE else {}

    if engineered:
        m.update({
            "glycemic_load_score":        engineered["glycemic_load_score"],
            "glucose_volatility_mg_dl":   engineered["glucose_volatility"],
            "circadian_factor":           engineered["circadian_rhythm_factor"],
            "meal_impact_factor":         engineered["meal_impact_factor"],
            "activity_adjusted_response": engineered["activity_adjusted_glucose_response"],
            "sleep_impact_coefficient":   engineered["sleep_impact_coefficient"],
            "inflammation_proxy":         engineered["inflammation_proxy"],
            "bergman_p3":                 round(state["p3"], 8),
            "bergman_G":                  round(state["G"], 1),
            "bergman_X":                  round(state["X"], 5),
            # Phase 2 additions (additive)
            "metabolic_age_fe":           engineered["metabolic_age"],
            "hba1c_estimate_fe":          engineered["hba1c_estimate"],
            "coefficient_of_variation":   engineered["coefficient_of_variation"],
        })
    else:
        m.update({
            "glycemic_load_score":             round(G * 0.09 + 20, 1),
            "glucose_volatility_mg_dl":        round(abs(G - 140) * 0.3, 1),
            "circadian_factor":                round(0.85 + 0.1 * math.sin(
                                                   math.pi * datetime.now().hour / 12), 3),
            "meal_impact_factor":              round(30 + G * 0.05, 1),
            "activity_adjusted_response":      round(-12 - state["p3_index"] * 0.06, 1),
            "sleep_impact_coefficient":        round(1.0 + 1 * 0.18, 2),
            "inflammation_proxy":              round(state.get("metabolic_stress", 35), 1),
            "bergman_p3":                      round(state["p3"], 8),
            "bergman_G":                       round(state["G"], 1),
            "bergman_X":                       round(state["X"], 5),
        })

    # Phase 2: real reversal-engine quick snapshot, layered on top of the
    # inline reversal_score compute_metrics() already produces (kept for
    # compatibility) — this adds the engine's own grading/staging/nudge.
    if _AI_BRIDGE_AVAILABLE:
        rev = _ai.reversal_quick(
            glucose=G, p3_index=state["p3_index"], tir=m.get("tir_pct", 63),
            steps=4000, bmi=user.get('bmi') or 27.0)
        if rev:
            m["reversal_engine"] = rev

    # Phase 2: latest retrospective evaluation metrics, if any have been computed
    if _DB_AVAILABLE:
        eval_metrics = await _db.get_latest_evaluation_metrics(patient_id)
        if eval_metrics:
            m["evaluation_metrics"] = {
                "mae": eval_metrics.get("mae"), "rmse": eval_metrics.get("rmse"),
                "mard_pct": eval_metrics.get("mard"), "r2": eval_metrics.get("r2"),
                "n_samples": eval_metrics.get("n_samples"),
                "computed_at": eval_metrics.get("timestamp"),
            }

    return {"patient_id": patient_id, "metrics": m,
            "timestamp": datetime.utcnow().isoformat()}


# ─────────────────────────────────────────────
# REVERSAL ENGINE — detailed breakdown (Phase 2, additive endpoint)
# ─────────────────────────────────────────────
@app.get("/reversal/{patient_id}", tags=["Reversal Engine"])
async def get_reversal_detail(patient_id: str, user=Depends(get_current_user)):
    """
    Full reversal-engine breakdown: Reversal Score, Lifestyle Score,
    Recovery Index, Insulin Sensitivity Score, Metabolic Health Score, and
    Weekly Improvement Score — each with an explanation. Heavier than
    /metrics' quick reversal snapshot (runs reversal_engine.py's full
    8-sub-system analysis), so this is its own endpoint rather than
    something called on every dashboard poll.
    """
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    if not _AI_BRIDGE_AVAILABLE or not AI_STATUS.get('reversal_engine'):
        raise HTTPException(503, "Reversal engine unavailable")
    state = await load_twin(patient_id)
    if not state:
        raise HTTPException(404, "Patient twin not found")

    cutoff = datetime.utcnow() - timedelta(days=30)
    glucose_logs = await _db.get_logs(patient_id, "glucose", limit=200, since=cutoff)
    profile_dict = {**user, 'patient_id': patient_id, 'p3': state.get('p3')}
    analysis = _ai.reversal_full(profile_dict, glucose_logs)
    if not analysis:
        raise HTTPException(503, "Reversal analysis failed")

    scoring = analysis['recovery_scoring']
    comp = scoring['component_scores']

    ctx = await build_personalized_context(patient_id, user, state)
    engineered = _ai.compute_live_features(
        [h["G"] for h in state.get("history", [])], {**ctx, 'glucose': state["G"]})

    recovery_index = engineered.get('recovery_score', 70.0)
    metabolic_health_score = round(
        (comp['glucose_stability'] + comp['hba1c_improvement']
         + (100 - engineered.get('metabolic_stress_score', 40))) / 3, 1)

    # Weekly improvement: compare against a snapshot from ~7 days ago
    week_ago = datetime.utcnow() - timedelta(days=8)
    week_recent = datetime.utcnow() - timedelta(days=6)
    past = await _db.get_reversal_history(patient_id, since=week_ago, limit=50)
    past_in_window = [p for p in past if p["timestamp"] <= week_recent.isoformat()]
    if past_in_window:
        prior_score = past_in_window[0]["reversal_score"]
        weekly_improvement = round(scoring['reversal_score'] - prior_score, 1)
        improvement_note = (f"Up {weekly_improvement} points vs. ~1 week ago"
                             if weekly_improvement > 0 else
                             f"Down {abs(weekly_improvement)} points vs. ~1 week ago"
                             if weekly_improvement < 0 else "Unchanged vs. ~1 week ago")
    else:
        weekly_improvement = None
        improvement_note = "Not enough history yet (need a snapshot from ~7 days ago) — check back next week"

    result = {
        "patient_id": patient_id,
        "scores": {
            "reversal_score": scoring['reversal_score'],
            "lifestyle_score": comp['lifestyle_score'],
            "recovery_index": recovery_index,
            "insulin_sensitivity_score": comp['insulin_sensitivity'],
            "metabolic_health_score": metabolic_health_score,
            "weekly_improvement_score": weekly_improvement,
        },
        "explanations": {
            "reversal_score": f"Weighted composite of insulin sensitivity (28%), time-in-range (25%), "
                               f"HbA1c progress (20%), glucose stability (12%), lifestyle (10%), weight (5%). "
                               f"Currently {scoring['recovery_stage'].replace('_',' ').title()}.",
            "lifestyle_score": f"From steps, sleep, stress, and hydration consistency over recent logs "
                                f"({comp['lifestyle_score']}/100).",
            "recovery_index": f"Real-time recovery proxy from feature_engineering.py: combines TIR tendency, "
                               f"activity, and risk factors ({recovery_index}/100).",
            "insulin_sensitivity_score": f"Derived from your current p3 (insulin sensitivity) index "
                                          f"relative to the healthy reference range ({comp['insulin_sensitivity']}/100).",
            "metabolic_health_score": f"Composite of glucose stability, HbA1c progress, and inverse metabolic "
                                      f"stress ({metabolic_health_score}/100).",
            "weekly_improvement_score": improvement_note,
        },
        "recovery_stage": scoring['recovery_stage'],
        "clinical_remission_pct": scoring['clinical_remission_pct'],
        "top_habit_interventions": analysis['top_habit_interventions'],
        "precision_nudges": analysis['precision_nudges'],
        "reversal_summary": analysis['reversal_summary'],
    }

    # Persist this snapshot so next week's weekly_improvement_score has
    # something real to compare against.
    await _db.add_reversal_snapshot(patient_id, {
        "reversal_score": scoring['reversal_score'],
        "lifestyle_score": comp['lifestyle_score'],
        "recovery_index": recovery_index,
        "insulin_sensitivity_score": comp['insulin_sensitivity'],
        "metabolic_health_score": metabolic_health_score,
        "recovery_stage": scoring['recovery_stage'],
        "component_scores": comp,
    })

    return result


# ─────────────────────────────────────────────
# WHAT-IF SIMULATOR
# ─────────────────────────────────────────────
@app.post("/simulate", tags=["Simulator"])
async def simulate(req: WhatIfRequest, user=Depends(get_current_user)):
    if user["patient_id"] != req.patient_id:
        raise HTTPException(403, "Access denied")
    state = await load_twin(req.patient_id)
    if not state:
        raise HTTPException(404, "Patient twin not found")

    G0  = req.current_glucose or state["G"]
    X0  = state["X"]
    p3b = state["p3"]

    # Adjust p3 for scenario
    p3_sim = p3b * (1 + req.exercise_min * 0.008) * (req.sleep_hours / 7)
    p3_sim = max(1e-5, min(8e-5, p3_sim))
    stress_mult = 1 + (req.stress_level - 1) * 0.04

    sim_eng  = BergmanEngine(p3=p3_sim)
    base_eng = BergmanEngine(p3=p3b)

    N = 48
    meal_trace = [sim_eng.meal_bolus(req.carbs, k * 5) for k in range(N)]
    base_trace = [0.0] * N
    I_trace    = [7 + max(0, (G0 - 90) * 0.08)] * N

    sim_G, base_G = G0, G0
    sim_X, base_X = X0, X0
    sim_traj, base_traj = [], []
    for k in range(N):
        sim_G,  sim_X  = sim_eng.step(sim_G,  sim_X,  I_trace[k], meal_trace[k])
        base_G, base_X = base_eng.step(base_G, base_X, I_trace[k], base_trace[k])
        sim_traj.append(round(sim_G * stress_mult, 1))
        base_traj.append(round(base_G, 1))

    peak    = max(sim_traj)
    spike   = round(sum(1 for g in sim_traj if g > 180) / N * 100, 1)
    tir_sim = round(sum(1 for g in sim_traj if 70 <= g <= 180) / N * 100, 1)
    recov_i = next((N - 1 - i for i, g in enumerate(reversed(sim_traj)) if g > 120), 0)
    recov_h = round(recov_i * 5 / 60, 1)

    advice = []
    if req.carbs > 80:
        advice.append(f"High carb load ({req.carbs}g) will spike glucose. "
                      f"Consider reducing to 50g or choosing brown rice/dal.")
    if req.exercise_min >= 30:
        advice.append(f"{req.exercise_min} min exercise boosts sensitivity by "
                      f"~{round(req.exercise_min * 0.5, 0):.0f}%. Best timed 15-30 min post-meal.")
    if req.sleep_hours < 6:
        advice.append(f"Only {req.sleep_hours}h sleep increases insulin resistance 18-22%. Aim for 7-9h.")
    if req.stress_level >= 4:
        advice.append("High stress triggers cortisol → hepatic glucose output. "
                      "Try 5-min box breathing to cut glucose by 8-12 mg/dL.")
    if not advice:
        advice.append("Your planned scenario looks balanced. Keep this pattern for sustained reversal.")

    # Save to simulation/prediction history
    sim_record = await _db.add_prediction(req.patient_id, {
        "inputs":      req.dict(exclude={"patient_id"}),
        "peak_glucose": peak,
        "spike_pct":   spike,
        "tir_pct":     tir_sim,
        "recovery_h":  recov_h,
    })

    return {
        "baseline":  base_traj,
        "simulated": sim_traj,
        "labels":    [f"{k*5}m" for k in range(N)],
        "results": {
            "peak_glucose":      round(peak, 1),
            "spike_probability": spike,
            "recovery_hours":    recov_h,
            "tir_pct":           tir_sim,
            "p3_adjusted":       round(p3_sim / 5e-5 * 100, 1),
        },
        "advice": advice,
        "sim_id": sim_record["id"],
    }


@app.get("/simulate/history/{patient_id}", tags=["Simulator"])
async def sim_history(patient_id: str, limit: int = Query(20),
                      user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    return {"history": await _db.get_predictions(patient_id, limit=limit)}


# ─────────────────────────────────────────────
# AI COACH  (proxied server-side — no CORS)
# ─────────────────────────────────────────────
@app.post("/coach/ask", tags=["AI Coach"])
async def coach_ask(msg: CoachMessage, user=Depends(get_current_user)):
    if user["patient_id"] != msg.patient_id:
        raise HTTPException(403, "Access denied")
    state   = await load_twin(msg.patient_id) or {}
    metrics = compute_metrics(state) if state else {}
    profile = {k: v for k, v in user.items()
               if k in ("full_name","age","bmi","hba1c","goal","activity_level")}

    question = (msg.message or "").strip()
    question_lower = question.lower()
    greeting = question_lower in {"hi", "hello", "hey", "hey there", "hi there"}
    health_summary = any(w in question_lower for w in [
        "how am i doing", "how is my health", "summarize my", "my overall progress", "my health status"
    ])

    def add_if_not_blank(target: list, value: Optional[str]) -> None:
        if value and str(value).strip():
            target.append(str(value).strip())

    # Build a focused, question-specific context instead of dumping the whole profile.
    ctx = f"Current authenticated user: {profile.get('full_name', 'Patient')} (patient_id={msg.patient_id})."
    relevant_context = []

    question_type = "general"
    if greeting:
        question_type = "greeting"
    elif health_summary:
        question_type = "summary"
    elif any(w in question_lower for w in ["eat", "food", "meal", "diet", "nutrition"]):
        question_type = "food"
    elif any(w in question_lower for w in ["exercise", "walk", "workout", "gym", "activity"]):
        question_type = "exercise"
    elif any(w in question_lower for w in ["sleep", "rest", "tired"]):
        question_type = "sleep"
    elif any(w in question_lower for w in ["reversal", "reverse", "score", "goal"]):
        question_type = "reversal"
    elif any(w in question_lower for w in ["digital twin", "twin", "insulin sensitivity", "p3", "personalized", "model"]):
        question_type = "twin"
    elif any(w in question_lower for w in ["glucose", "sugar", "high", "low", "spike", "trend", "current glucose"]):
        question_type = "glucose"

    if _DB_AVAILABLE:
        try:
            cutoff = datetime.utcnow() - timedelta(days=7)
            week_logs = await _db.get_logs(msg.patient_id, None, limit=1000, since=cutoff)
            by_type: Dict[str, list] = {}
            for l in week_logs:
                by_type.setdefault(l["type"], []).append(l)

            glucose_logs = [l for l in by_type.get("glucose", []) if l.get("glucose") is not None]
            meal_logs = by_type.get("meal", [])
            activity_logs = by_type.get("activity", [])
            sleep_logs = by_type.get("sleep", [])
            weight_logs = by_type.get("weight", [])
            goals = await _db.get_goals(msg.patient_id)

            if question_type == "glucose":
                if glucose_logs:
                    latest_glucose = glucose_logs[-1]
                    latest_val = latest_glucose.get("glucose")
                    latest_ts = latest_glucose.get("timestamp")
                    add_if_not_blank(relevant_context, f"latest glucose reading: {latest_val} mg/dL at {latest_ts}")
                    recent_series = [float(item.get("glucose")) for item in glucose_logs[-7:] if item.get("glucose") is not None]
                    if len(recent_series) >= 2:
                        avg_recent = round(sum(recent_series) / len(recent_series), 1)
                        add_if_not_blank(relevant_context, f"recent 7-day glucose average: {avg_recent} mg/dL")
                        tir = round(sum(70 <= value <= 180 for value in recent_series) / len(recent_series) * 100, 1)
                        add_if_not_blank(relevant_context, f"time in range from recent logged readings: {tir}%")
                    if len(glucose_logs) >= 2:
                        recent_vals = ", ".join(str(round(float(l.get("glucose")), 1)) for l in glucose_logs[-3:] if l.get("glucose") is not None)
                        add_if_not_blank(relevant_context, f"recent glucose values: {recent_vals} mg/dL")
                else:
                    add_if_not_blank(relevant_context, "No glucose readings have been logged by this user in the last 7 days.")
            elif question_type == "food":
                if meal_logs:
                    recent_meals = [l.get("food_name") for l in meal_logs[-5:] if l.get("food_name")]
                    if recent_meals:
                        add_if_not_blank(relevant_context, f"recent meals: {', '.join(recent_meals)}")
            elif question_type == "exercise":
                if activity_logs:
                    act_min = sum(l.get("duration_min", 0) or 0 for l in activity_logs)
                    if act_min:
                        add_if_not_blank(relevant_context, f"activity in last 7 days: {act_min} minutes")
                if sleep_logs:
                    avg_sleep = round(sum(l.get("hours", 0) or 0 for l in sleep_logs) / len(sleep_logs), 1)
                    add_if_not_blank(relevant_context, f"average sleep: {avg_sleep} hours")
            elif question_type == "sleep":
                if sleep_logs:
                    avg_sleep = round(sum(l.get("hours", 0) or 0 for l in sleep_logs) / len(sleep_logs), 1)
                    add_if_not_blank(relevant_context, f"average sleep: {avg_sleep} hours")
            elif question_type == "summary":
                if glucose_logs:
                    latest_glucose = glucose_logs[-1]
                    add_if_not_blank(
                        relevant_context,
                        f"latest logged glucose: {latest_glucose.get('glucose')} mg/dL at {latest_glucose.get('timestamp')}",
                    )
                if meal_logs:
                    recent_meals = [l.get("food_name") for l in meal_logs[-3:] if l.get("food_name")]
                    if recent_meals:
                        add_if_not_blank(relevant_context, f"recent meals: {', '.join(recent_meals)}")
                if activity_logs:
                    act_min = sum(l.get("duration_min", 0) or 0 for l in activity_logs)
                    if act_min:
                        add_if_not_blank(relevant_context, f"recent activity: {act_min} minutes")
                if sleep_logs:
                    avg_sleep = round(sum(l.get("hours", 0) or 0 for l in sleep_logs) / len(sleep_logs), 1)
                    add_if_not_blank(relevant_context, f"average sleep: {avg_sleep} hours")

            if weight_logs:
                recent_weight = weight_logs[-1].get("weight_kg")
                if question_type == "summary":
                    add_if_not_blank(relevant_context, f"latest weight: {recent_weight} kg")

            if goals and question_type in {"summary", "food", "reversal"}:
                goal_strs = [f"target {k.replace('target_', '')}={v}" for k, v in goals.items() if k.startswith("target_")]
                if goal_strs:
                    add_if_not_blank(relevant_context, f"active goal targets (not current measurements): {', '.join(goal_strs)}")

            if state and question_type == "twin":
                twin_summary = (
                    f"Digital Twin model state (simulated, not logged measurements): glucose={state.get('G')}, insulin={state.get('I')}, "
                    f"sensitivity={state.get('X')}, p3={state.get('p3')}"
                )
                add_if_not_blank(relevant_context, twin_summary)

            if metrics and question_type in {"summary", "reversal", "twin"}:
                if metrics.get("insulin_sensitivity") is not None and question_type in {"summary", "reversal", "twin"}:
                    add_if_not_blank(relevant_context, f"BioMirror model-estimated insulin sensitivity: {metrics.get('insulin_sensitivity')}%")
                if metrics.get("reversal_score") is not None and question_type in {"summary", "reversal"}:
                    add_if_not_blank(relevant_context, f"BioMirror model-estimated reversal score: {metrics.get('reversal_score')}/100")
                if question_type == "reversal":
                    add_if_not_blank(
                        relevant_context,
                        "BioMirror score formula: model-estimated insulin sensitivity 35%, Digital Twin time-in-range 35%, model-estimated recovery 30%.",
                    )
                    add_if_not_blank(
                        relevant_context,
                        f"Score components: sensitivity={metrics.get('insulin_sensitivity')}%, twin time-in-range={metrics.get('tir_pct')}%, recovery={metrics.get('recovery_score')}/100.",
                    )

        except Exception as e:
            logger.warning(f"[Coach] context enrichment failed: {e}")

    if relevant_context:
        ctx += " " + " ".join(relevant_context) + "."
    else:
        ctx += " No personal health data is available yet for this user."

    system_prompt = (
        "You are BioMirror's AI Metabolic Coach. Answer the actual user question directly. "
        "Use ONLY the current authenticated user's real BioMirror data. Never use another user's data, generic examples, or fabricated values. "
        "Use only the data relevant to the question. Do not repeat glucose, sensitivity, or reversal score information for unrelated questions. "
        "If the required user data is missing, say that the data is not available instead of guessing. "
        "Never present model estimates or goal targets as direct measurements; label estimates and distinguish targets from current values. "
        "For glucose, use logged glucose readings only; never substitute a simulated twin state for a logged reading. "
        "Never say the user is on track, improving, meeting a target, or making progress unless a current logged measurement supports that comparison. "
        "When explaining a score, use only the formula and components supplied in the user context; do not invent contributors or claim logged inputs unless identified as logged. "
        "Do not invent numerical effects, dates, readings, targets, or personal history. "
        "Do not dump the full health profile in every answer. "
        f"Current question: {question}. "
        f"Relevant current user data: {ctx} "
        f"Question type: {question_type}. "
        "Answer in one to three concise bullet points, at most 22 words each; never repeat a point or pad the answer. "
        "For general health knowledge, answer the question directly but do not claim it is based on this user's history unless supported by the supplied data. "
        "Treat digital-twin state as simulated, not a direct measurement. For greetings, reply briefly. Never recommend medication changes."
    )

    # Keep this request isolated so old answers cannot contaminate current facts.
    messages = [{"role": "user", "content": question}]

    ai_text = None

    # Try Groq server-side first; the API key never leaves this process.
    provider_model = "offline-rule-based"
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                resp = await client.post(
                    GROQ_API_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": GROQ_MODEL, "temperature": 0.3,
                          "max_tokens": 500, "reasoning_effort": "low",
                          "messages": [{"role": "system", "content": system_prompt}, *messages]},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    ai_text = data["choices"][0]["message"]["content"].strip()
                    if ai_text:
                        provider_model = GROQ_MODEL
                    else:
                        logger.warning(
                            "[Coach] Groq returned empty visible content (finish_reason=%s)",
                            data["choices"][0].get("finish_reason"),
                        )
                else:
                    logger.warning(f"[Coach] Groq request failed ({resp.status_code}): {resp.text[:300]}")
        except Exception as e:
            logger.warning(f"[Coach] Groq request error: {e}")

    # Try Claude API server-side if Groq is not configured or unavailable.
    if not ai_text and ANTHROPIC_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": ANTHROPIC_API_KEY,
                             "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json={"model": ANTHROPIC_MODEL, "max_tokens": 250,
                          "system": system_prompt, "messages": messages},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    ai_text = data["content"][0]["text"]
                    provider_model = ANTHROPIC_MODEL
                else:
                    logger.warning(f"[Coach] Anthropic request failed ({resp.status_code})")
        except Exception as e:
            logger.warning(f"[Coach] Anthropic request error: {e}")

    if not ai_text:
        raise HTTPException(
            status_code=503,
            detail="AI Coach is temporarily unavailable because no AI provider returned a response. Please try again.",
        )

    # Save to history (persisted)
    await _db.add_chat_message(msg.patient_id, "user", msg.message)
    await _db.add_chat_message(msg.patient_id, "assistant", ai_text)
    new_total = len(await _db.get_chat_history(msg.patient_id, limit=10000))

    return {
        "patient_id": msg.patient_id,
        "user_message": msg.message,
        "ai_response":  ai_text,
        "model": provider_model,
        "context_used": ctx,
        "history_length": new_total,
    }


@app.get("/coach/history/{patient_id}", tags=["AI Coach"])
async def coach_history(patient_id: str, limit: int = Query(20),
                         user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    hist = await _db.get_chat_history(patient_id, limit=limit * 2)
    hist = [{"role": h["role"], "content": h["message"]} for h in hist]
    return {"history": hist, "total_turns": len(hist) // 2}


@app.delete("/coach/history/{patient_id}", tags=["AI Coach"])
async def clear_coach_history(patient_id: str, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    await _db.clear_chat_history(patient_id)
    return {"success": True}


# ─────────────────────────────────────────────
# NOTIFICATIONS
# ─────────────────────────────────────────────
@app.get("/notifications/{patient_id}", tags=["Notifications"])
async def get_notifications(patient_id: str, unread_only: bool = Query(False),
                             user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    notifs = await _db.get_notifications(patient_id, unread_only=unread_only)
    unread = await _db.count_unread_notifications(patient_id)
    return {"notifications": notifs, "unread_count": unread}


@app.post("/notifications/{patient_id}/mark-read", tags=["Notifications"])
async def mark_read(patient_id: str, body: MarkNotificationsRead,
                    user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    marked = await _db.mark_notifications_read(patient_id, ids=body.notification_ids)
    return {"success": True, "marked": marked}


# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────
@app.get("/analytics/{patient_id}", tags=["Analytics"])
async def get_analytics(patient_id: str, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    logs   = await _db.get_logs(patient_id, None, limit=5000)
    state  = await load_twin(patient_id) or {}
    metrics = compute_metrics(state) if state else {}

    glucose_logs = [l for l in logs if l.get("type") == "glucose"]
    g_values     = [l["glucose"] for l in glucose_logs if l.get("glucose") is not None]
    meal_logs    = [l for l in logs if l.get("type") == "meal"]
    act_logs     = [l for l in logs if l.get("type") == "activity"]

    mean_g = round(sum(g_values)/len(g_values), 1) if g_values else metrics.get("glucose", 140)
    tir    = round(sum(1 for g in g_values if 70<=g<=180)/len(g_values)*100, 1) if g_values else metrics.get("tir_pct", 63)
    hba1c  = round((mean_g + 46.7) / 28.7, 2)
    avg_steps = round(sum(l.get("steps", 0) or 0 for l in glucose_logs) / max(len(glucose_logs), 1))
    avg_carbs = round(sum((l.get("carbs_g") or l.get("carbs") or 0) for l in meal_logs + glucose_logs)
                      / max(len(meal_logs) + len(glucose_logs), 1), 1)

    sim_count = await _db.count_predictions(patient_id)

    return {
        "patient_id":    patient_id,
        "glucose": {
            "mean":      mean_g,
            "tir_pct":   tir,
            "hba1c":     hba1c,
            "readings":  len(g_values),
            "hypo":      sum(1 for g in g_values if g < 70),
            "hyper":     sum(1 for g in g_values if g > 180),
        },
        "reversal": {
            "score":              metrics.get("reversal_score", 60),
            "insulin_sensitivity": metrics.get("insulin_sensitivity", 68),
            "metabolic_age":      metrics.get("metabolic_age", 44),
        },
        "lifestyle": {
            "avg_steps":    avg_steps,
            "avg_carbs_g":  avg_carbs,
            "meals_logged": len(meal_logs),
            "activity_sessions": len(act_logs),
        },
        "totals": {"logs": len(logs), "simulations": sim_count},
    }


# ─────────────────────────────────────────────
# FHIR
# ─────────────────────────────────────────────
@app.get("/fhir/{patient_id}", tags=["FHIR"])
async def fhir_bundle(patient_id: str, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    events = await _db.get_logs(patient_id, "glucose", limit=10)
    observations = []
    for i, ev in enumerate(events):
        g = ev.get("glucose", 0)
        observations.append({
            "resourceType": "Observation",
            "id": f"obs-{patient_id}-{i}",
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                       "code": "laboratory"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "2339-0",
                                  "display": "Glucose [Mass/volume] in Blood"}]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": ev.get("timestamp", datetime.utcnow().isoformat()),
            "valueQuantity": {"value": g, "unit": "mg/dL",
                              "system": "http://unitsofmeasure.org", "code": "mg/dL"},
            "interpretation": [{"coding": [{"code": BergmanEngine.classify(g)}]}],
        })
    bundle = {"resourceType": "Bundle", "type": "collection",
              "id": f"biomirror-{patient_id}",
              "entry": [{"resource": o} for o in observations]}
    return bundle


@app.get("/fhir/{patient_id}/download", tags=["FHIR"])
async def fhir_download(patient_id: str, user=Depends(get_current_user)):
    """Returns FHIR bundle as downloadable JSON file."""
    bundle = await fhir_bundle(patient_id, user)
    content = json.dumps(bundle, indent=2)
    return Response(
        content=content,
        media_type="application/fhir+json",
        headers={"Content-Disposition": f"attachment; filename=biomirror-{patient_id}.json"},
    )


# ─────────────────────────────────────────────
# WEBSOCKET — REAL-TIME TWIN
# ─────────────────────────────────────────────
@app.websocket("/ws/{patient_id}")
async def ws_twin(websocket: WebSocket, patient_id: str):
    await websocket.accept()
    WS_CONNECTIONS.setdefault(patient_id, []).append(websocket)
    state = await load_twin(patient_id)
    if not state:
        await websocket.send_json({"error": "Twin not found"})
        await websocket.close(); return
    tick_count = 0
    try:
        eng = BergmanEngine(p3=state["p3"])
        while True:
            meal_in = sum(eng.meal_bolus(m["carbs"], m["elapsed"])
                          for m in state.get("meal_queue", []))
            for m in state.get("meal_queue", []):
                m["elapsed"] += eng.DT
            G, X = eng.step(state["G"], state["X"], state["I"], meal_in)
            state["G"] = G; state["X"] = X; state["sim_time"] += eng.DT
            state["history"].append({"t": state["sim_time"], "G": G})
            if len(state["history"]) > 288:
                state["history"].pop(0)
            tick_count += 1
            # Checkpoint to the DB every ~30s (every 15th 2s tick) instead of
            # on every single tick. Every-tick writes previously rewrote
            # biomirror.db roughly every 2 seconds for as long as any
            # dashboard was open — harmless to the app itself, but file
            # churn that frequency looks exactly like a source-file change
            # to any file-watching dev/live-reload tool pointed at the
            # project directory, which explains the reload loop reported
            # after Phase 1. The live dashboard is unaffected: it still
            # gets this exact update every 2s via the socket push below,
            # regardless of DB write frequency. A final checkpoint also
            # fires on disconnect (see the except block) so no state is
            # lost when the user actually navigates away.
            if tick_count % 15 == 0:
                await save_twin(patient_id, state)
            else:
                touch_twin_cache(patient_id, state)
            metrics = compute_metrics(state)
            await websocket.send_json({"event": "tick", "patient_id": patient_id,
                                       **metrics})
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        WS_CONNECTIONS[patient_id].remove(websocket)
        # Final checkpoint on disconnect — ensures state since the last
        # periodic checkpoint isn't lost when the user navigates away.
        await save_twin(patient_id, state)


# ─────────────────────────────────────────────
# NEW LOGGING ENDPOINTS (v3)
# ─────────────────────────────────────────────

@app.post("/logs/{patient_id}/bp", tags=["Logging"])
async def log_blood_pressure(patient_id: str, log: BloodPressureLog,
                              user=Depends(get_current_user)):
    """Log blood pressure reading. Alerts on hypertension (>140/90)."""
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    entry = await add_log(patient_id, "blood_pressure", log.dict())
    # BP category
    if log.systolic >= 180 or log.diastolic >= 120:
        category = "Hypertensive Crisis"
        await push_notification(patient_id, "🚨 Hypertensive Crisis",
            f"BP {log.systolic}/{log.diastolic} mmHg is dangerously high. "
            "Seek immediate medical attention.", ntype="alert")
    elif log.systolic >= 140 or log.diastolic >= 90:
        category = "High (Stage 2)"
        await push_notification(patient_id, "⚠ High Blood Pressure",
            f"BP {log.systolic}/{log.diastolic} mmHg is above normal. "
            "High BP increases insulin resistance by 15–20%.", ntype="alert")
    elif log.systolic >= 130 or log.diastolic >= 80:
        category = "High (Stage 1)"
    elif log.systolic >= 120:
        category = "Elevated"
    else:
        category = "Normal"
    # BP also modulates twin stress
    state = await load_twin(patient_id)
    if state and log.systolic > 140:
        # High BP → cortisol → raises glucose
        state["G"] = min(400, state["G"] * 1.05)
        await save_twin(patient_id, state)
    try:
        await compute_personal_features(patient_id)
    except Exception as e:
        logger.warning(f"[Personal Learning] feature recompute after BP log failed for {patient_id}: {e}")
    return {"success": True, "log_id": entry["id"],
            "bp_category": category,
            "reading": f"{log.systolic}/{log.diastolic} mmHg",
            "pulse": log.pulse}


@app.post("/logs/{patient_id}/water", tags=["Logging"])
async def log_water(patient_id: str, log: WaterLog, user=Depends(get_current_user)):
    """Log water intake. Dehydration raises glucose 5–10%."""
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    entry = await add_log(patient_id, "water", log.dict())
    # Calculate today's running total
    today_start = datetime.combine(datetime.utcnow().date(), datetime.min.time())
    today_logs = await _db.get_logs(patient_id, "water", limit=200, since=today_start)
    daily_total = sum(l.get("amount_ml", 0) or 0 for l in today_logs)
    target_ml = 2500
    pct = round(daily_total / target_ml * 100, 0)
    if daily_total >= target_ml:
        await push_notification(patient_id, "💧 Hydration Goal Met!",
            f"You've had {daily_total}ml today. Great hydration keeps glucose stable!")
    try:
        await compute_personal_features(patient_id)
    except Exception as e:
        logger.warning(f"[Personal Learning] feature recompute after water log failed for {patient_id}: {e}")
    return {"success": True, "log_id": entry["id"],
            "daily_total_ml": daily_total,
            "target_ml": target_ml,
            "target_pct": min(100, pct),
            "remaining_ml": max(0, target_ml - daily_total),
            "tip": "Well hydrated!" if daily_total >= 2000 else
                   f"Drink {target_ml - daily_total}ml more — dehydration raises glucose 5–10%"}


@app.post("/logs/{patient_id}/hba1c", tags=["Logging"])
async def log_hba1c(patient_id: str, log: HbA1cLog, user=Depends(get_current_user)):
    """Record lab HbA1c result and recalibrate Digital Twin baseline."""
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    uname = user["username"]
    prev_hba1c = user.get("hba1c")
    # Update profile
    await _db.update_user(uname, {"hba1c": log.hba1c,
                                   "hba1c_date": log.test_date or datetime.utcnow().date().isoformat()})
    # Recalibrate twin baseline glucose using Nathan formula inverse
    mean_g_estimate = log.hba1c * 28.7 - 46.7
    state = await load_twin(patient_id)
    if state:
        state["G"] = round(min(400, max(40, mean_g_estimate)), 1)
        state["last_updated"] = datetime.utcnow().isoformat()
        await save_twin(patient_id, state)
    entry = await add_log(patient_id, "hba1c", log.dict())
    # Status classification
    if log.hba1c >= 6.5:
        status_str = "Diabetic"
    elif log.hba1c >= 5.7:
        status_str = "Pre-diabetic"
    else:
        status_str = "Normal"
    improvement = None
    if prev_hba1c:
        improvement = round(prev_hba1c - log.hba1c, 1)
        if improvement > 0:
            await push_notification(patient_id, "🎉 HbA1c Improved!",
                f"HbA1c dropped from {prev_hba1c}% to {log.hba1c}% "
                f"(−{improvement}%). Keep up the great work!")
        elif improvement < 0:
            await push_notification(patient_id, "⚠ HbA1c Increased",
                f"HbA1c rose from {prev_hba1c}% to {log.hba1c}%. "
                "Review your diet and activity plan with your coach.", ntype="alert")
    try:
        await compute_personal_features(patient_id)
    except Exception as e:
        logger.warning(f"[Personal Learning] feature recompute after HbA1c log failed for {patient_id}: {e}")
    return {"success": True, "log_id": entry["id"],
            "hba1c": log.hba1c,
            "status": status_str,
            "change_from_last": improvement,
            "estimated_avg_glucose_mg_dl": round(mean_g_estimate, 1),
            "twin_recalibrated": True}


# ─────────────────────────────────────────────
# AUTH ENHANCEMENTS (v3)
# ─────────────────────────────────────────────

@app.post("/auth/change-password", tags=["Auth"])
async def change_password(body: PasswordChange, user=Depends(get_current_user)):
    """Change account password. Requires current password verification."""
    if not verify_pw(body.current_password, user["password"]):
        raise HTTPException(400, "Current password is incorrect")
    await _db.update_user(user["username"], {"password": hash_pw(body.new_password),
                                              "password_changed_at": datetime.utcnow()})
    await push_notification(user["patient_id"], "🔒 Password Changed",
        "Your password was changed successfully. "
        "If this was not you, contact support immediately.")
    return {"success": True, "message": "Password changed successfully"}


# ─────────────────────────────────────────────
# PATIENT GOALS (v3)
# ─────────────────────────────────────────────

@app.post("/goals/{patient_id}", tags=["Goals"])
async def set_goals(patient_id: str, goals: PatientGoal, user=Depends(get_current_user)):
    """Set personalized health goals for the patient."""
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    goal_data = goals.dict(exclude_none=True)
    saved = await _db.set_goals(patient_id, goal_data)
    return {"success": True, "goals": saved}


@app.get("/goals/{patient_id}", tags=["Goals"])
async def get_goals(patient_id: str, user=Depends(get_current_user)):
    """Get current patient goals with progress vs current metrics."""
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    goals = await _db.get_goals(patient_id)
    state = await load_twin(patient_id) or {}
    metrics = compute_metrics(state) if state else {}
    profile = user
    # Calculate progress for each goal
    progress = {}
    if "target_hba1c" in goals:
        current = profile.get("hba1c") or metrics.get("hba1c_estimate", 7.5)
        target  = goals["target_hba1c"]
        pct = max(0, min(100, round((7.5 - current) / (7.5 - target) * 100, 1))) if target < current else 0
        progress["hba1c"] = {"current": current, "target": target, "progress_pct": pct}
    if "target_tir" in goals:
        current = metrics.get("tir_pct", 60)
        target  = goals["target_tir"]
        pct = max(0, min(100, round(current / target * 100, 1)))
        progress["tir"] = {"current": current, "target": target, "progress_pct": pct}
    if "target_weight" in goals:
        current = profile.get("weight_kg") or 75
        target  = goals["target_weight"]
        pct = max(0, min(100, round((75 - current) / (75 - target) * 100, 1))) if target < current else 0
        progress["weight"] = {"current": current, "target": target, "progress_pct": pct}
    return {"patient_id": patient_id, "goals": goals, "progress": progress,
            "goals_set": len(goals) > 1}


@app.delete("/goals/{patient_id}/{goal_field}", tags=["Goals"])
async def remove_goal(patient_id: str, goal_field: str, user=Depends(get_current_user)):
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    allowed_fields = {"target_hba1c", "target_weight", "target_steps", "target_tir", "target_date"}
    if goal_field not in allowed_fields:
        raise HTTPException(404, "Goal not found")
    remaining_goals = await _db.remove_goal(patient_id, goal_field)
    return {"success": True, "goals": remaining_goals}


# ─────────────────────────────────────────────
# PERSONALIZED RECOMMENDATIONS (Phase 2, additive endpoint)
# ─────────────────────────────────────────────
@app.get("/recommendations/{patient_id}", tags=["Recommendations"])
async def get_recommendations(patient_id: str, user=Depends(get_current_user)):
    """
    Adaptive recommendation engine. Every recommendation below is
    conditioned on actual stored data for THIS patient over the last 14
    days — glucose, meals, activity, sleep, hydration, HbA1c, weight,
    blood pressure, prediction history, and goals.
    Two patients (or the same patient at two different times) will get
    different output because the underlying data is different — this is
    deliberately not a static text block.
    """
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    cutoff = datetime.utcnow() - timedelta(days=14)
    logs = await _db.get_logs(patient_id, None, limit=2000, since=cutoff)
    goals = await _db.get_goals(patient_id)
    predictions = await _db.get_predictions(patient_id, limit=20, source="forecast")
    logged_steps = _logged_steps_by_day(logs)
    avg_logged_steps = _mean(list(logged_steps.values()))

    by_type: Dict[str, list] = {}
    for l in logs:
        by_type.setdefault(l["type"], []).append(l)

    recs, basis = [], {}

    # ── Correlation-based recommendations using this patient's own
    # manually logged step counts and glucose history — the "your glucose
    # decreased by X mg/dL whenever you walked more than 7000 steps"
    # style explicitly requested, not a generic "exercise more" template.
    step_corr = await compute_steps_glucose_correlation(patient_id, days=30)
    if step_corr:
        basis["steps_glucose_correlation"] = step_corr
        diff = step_corr["difference_mg_dl"]
        if diff > 3:
            recs.append({"category": "activity", "priority": "high",
                "text": f"During the last {step_corr['days_analyzed']} days, your glucose averaged "
                        f"{diff} mg/dL lower on days you walked more than {step_corr['step_threshold']:,} "
                        f"steps ({step_corr['avg_glucose_high_step_days']} vs. "
                        f"{step_corr['avg_glucose_low_step_days']} mg/dL on lower-step days, based on "
                        f"{step_corr['high_step_days_count']} high-step and {step_corr['low_step_days_count']} "
                        f"lower-step days in your own history). Based on your personal pattern, a walk "
                        f"today would likely help."})

    personal = await compute_personal_features(patient_id)
    pf = personal["features"]
    if personal["days_of_data"] >= 5:
        basis["personal_features_days"] = personal["days_of_data"]
        if pf.get("exercise_effectiveness") is not None and pf["exercise_effectiveness"] < -3:
            recs.append({"category": "activity", "priority": "medium",
                "text": f"Looking at your own logs, glucose readings taken after 20+ minutes of "
                        f"exercise have averaged {abs(pf['exercise_effectiveness'])} mg/dL lower than "
                        f"readings with no exercise. This is specific to your data, not a general rule — "
                        f"exercise is measurably working for you."})
        if pf.get("sleep_effect_on_glucose") is not None and pf["sleep_effect_on_glucose"] > 5:
            recs.append({"category": "sleep", "priority": "high",
                "text": f"On days following a night with under 6 hours of sleep, your average glucose "
                        f"has run {pf['sleep_effect_on_glucose']} mg/dL higher than after nights with "
                        f"7+ hours, based on your own logged sleep and glucose. This is one of the "
                        f"largest personal levers visible in your data."})
        if pf.get("avg_morning_glucose") is not None and pf.get("avg_night_glucose") is not None:
            gap = round(pf["avg_night_glucose"] - pf["avg_morning_glucose"], 1)
            if abs(gap) > 15:
                basis["morning_night_gap"] = gap
                recs.append({"category": "glucose", "priority": "medium",
                    "text": f"Your average night-time glucose ({pf['avg_night_glucose']} mg/dL) runs "
                            f"{abs(gap)} mg/dL {'higher' if gap>0 else 'lower'} than your morning average "
                            f"({pf['avg_morning_glucose']} mg/dL) — worth reviewing your evening meal "
                            f"timing or composition."})
        if pf.get("weekly_activity_trend_pct") is not None and pf["weekly_activity_trend_pct"] < -20:
            recs.append({"category": "activity", "priority": "medium",
            "text": f"Your logged activity time is down {abs(pf['weekly_activity_trend_pct'])}% versus "
                f"the week before. Activity is one of the "
                        f"strongest levers in your own data — worth getting back on track."})

    g_vals = [l["glucose"] for l in by_type.get("glucose", []) if l.get("glucose") is not None]
    if g_vals:
        tir = sum(1 for g in g_vals if 70 <= g <= 180) / len(g_vals) * 100
        basis["glucose_tir_pct"] = round(tir, 1)
        basis["glucose_mean"] = round(sum(g_vals) / len(g_vals), 1)
        if tir < 60:
            recs.append({"category": "glucose", "priority": "high",
                "text": f"Your time-in-range over the last 2 weeks is {tir:.0f}% (below the 70% target). "
                        f"A 15-minute walk within 30 minutes of your largest meal is consistently "
                        f"your fastest lever based on your logged data."})
        elif tir >= 80:
            recs.append({"category": "glucose", "priority": "low",
                "text": f"Excellent control — {tir:.0f}% time-in-range over the last 2 weeks. "
                        f"Keep your current meal and activity pattern."})

    meal_logs = by_type.get("meal", [])
    if meal_logs:
        high_gi = [l for l in meal_logs if (l.get("gi_score") or 0) > 0.65]
        basis["high_gi_meal_pct"] = round(len(high_gi) / len(meal_logs) * 100, 1)
        if len(high_gi) / len(meal_logs) > 0.4:
            recs.append({"category": "nutrition", "priority": "medium",
                "text": f"{len(high_gi)} of your last {len(meal_logs)} logged meals were high-GI. "
                        f"Swapping white rice for brown rice or a dal-based meal could meaningfully "
                        f"lower your post-meal spikes."})

    act_logs = by_type.get("activity", [])
    total_active_min = sum(l.get("duration_min", 0) or 0 for l in act_logs)
    basis["active_minutes_14d"] = total_active_min
    if total_active_min < 150:
        recs.append({"category": "activity", "priority": "high",
            "text": f"Only {total_active_min} minutes of logged activity in 2 weeks "
                    f"(WHO target is 150/week, i.e. 300/2 weeks). A 20-minute daily walk "
                    f"would roughly double your current activity level."})

    sleep_logs = by_type.get("sleep", [])
    if sleep_logs:
        avg_sleep = sum(l.get("hours", 0) or 0 for l in sleep_logs) / len(sleep_logs)
        basis["avg_sleep_hours"] = round(avg_sleep, 1)
        if avg_sleep < 6.5:
            recs.append({"category": "sleep", "priority": "high",
                "text": f"Average sleep of {avg_sleep:.1f}h over your logged nights is below the "
                        f"7-9h target. Poor sleep raises insulin resistance 18-22% — this may be "
                        f"your highest-leverage change right now."})

    water_logs = by_type.get("water", [])
    if water_logs:
        days_logged = max(len(set(l["timestamp"][:10] for l in water_logs)), 1)
        avg_water = sum(l.get("amount_ml", 0) or 0 for l in water_logs) / days_logged
        basis["avg_daily_water_ml"] = round(avg_water)
        if avg_water < 1500:
            recs.append({"category": "hydration", "priority": "medium",
                "text": f"Your logged water intake averages ~{avg_water:.0f}ml/day, below the "
                        f"2000-2500ml target. Dehydration alone can raise glucose 5-10%."})

    hba1c_logs = by_type.get("hba1c", [])
    if len(hba1c_logs) >= 2:
        delta = hba1c_logs[-1]["hba1c"] - hba1c_logs[0]["hba1c"]
        basis["hba1c_trend"] = round(delta, 2)
        if delta > 0:
            recs.append({"category": "clinical", "priority": "high",
                "text": f"Your HbA1c has risen by {delta:.1f}% across your logged tests. "
                        f"Worth discussing your plan with your coach or physician."})
        elif delta < -0.2:
            recs.append({"category": "clinical", "priority": "low",
                "text": f"Your HbA1c has improved by {abs(delta):.1f}% — your current approach is working."})

    bp_logs = by_type.get("blood_pressure", [])
    if bp_logs:
        avg_sys = sum(l.get("systolic", 0) or 0 for l in bp_logs) / len(bp_logs)
        basis["avg_systolic"] = round(avg_sys, 1)
        if avg_sys >= 130:
            recs.append({"category": "cardiovascular", "priority": "medium",
                "text": f"Average systolic BP of {avg_sys:.0f} mmHg is above optimal. High BP "
                        f"compounds insulin resistance — worth monitoring alongside glucose."})

    if avg_logged_steps is not None:
        basis["avg_logged_steps"] = round(avg_logged_steps)
        if avg_logged_steps < 5000:
            recs.append({"category": "activity", "priority": "medium",
                "text": f"Your manually logged activity shows an average of {avg_logged_steps:.0f} steps/day. "
                        f"Small increases toward 7,000-8,000 meaningfully improve insulin sensitivity."})

    if predictions:
        confs = [p.get("confidence") for p in predictions if p.get("confidence") is not None]
        if confs:
            basis["avg_prediction_confidence"] = round(sum(confs) / len(confs), 2)

    if goals:
        basis["active_goals"] = list(goals.keys())
        if "target_steps" in goals and basis.get("avg_logged_steps"):
            gap = goals["target_steps"] - basis["avg_logged_steps"]
            if gap > 0:
                recs.append({"category": "goals", "priority": "medium",
                    "text": f"You're about {int(gap)} steps/day short of your "
                            f"{goals['target_steps']}-step goal on average."})

    if not recs:
        recs.append({"category": "general", "priority": "low",
            "text": "Not enough recent data to generate personalized recommendations yet — "
                    "log a few more days of glucose, meals, activity, and sleep."})

    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: order.get(r["priority"], 3))

    if _DB_AVAILABLE:
        await _db.save_recommendations(patient_id, recs, basis)

    return {"patient_id": patient_id, "recommendations": recs, "basis": basis,
            "generated_at": datetime.utcnow().isoformat()}


# ─────────────────────────────────────────────
# DATA EXPORT (v3)
# ─────────────────────────────────────────────

@app.get("/export/{patient_id}/csv", tags=["Export"])
async def export_logs_csv(patient_id: str, days: int = Query(30),
                           user=Depends(get_current_user)):
    """Export all health logs as a downloadable CSV file (Excel-compatible)."""
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    cutoff = datetime.utcnow() - timedelta(days=days)
    filtered = await _db.get_logs(patient_id, None, limit=5000, since=cutoff)
    if not filtered:
        raise HTTPException(404, "No logs to export")
    # Build CSV
    headers = ["id", "type", "timestamp", "glucose", "carbs", "steps",
               "exercise_min", "hours", "weight_kg", "systolic", "diastolic",
               "amount_ml", "hba1c", "food_name", "activity", "medication",
               "heart_rate", "stress_level", "notes"]
    rows = [",".join(headers)]
    for lg in filtered:
        row = []
        for h in headers:
            val = lg.get(h, "")
            # Escape commas in strings
            if isinstance(val, str) and "," in val:
                val = f'"{val}"'
            row.append(str(val) if val is not None else "")
        rows.append(",".join(row))
    csv_content = "\n".join(rows)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition":
                 f"attachment; filename=biomirror-{patient_id}-{days}days.csv"}
    )


@app.get("/export/{patient_id}/json", tags=["Export"])
async def export_logs_json(patient_id: str, days: int = Query(30),
                            user=Depends(get_current_user)):
    """Export all health logs as a downloadable JSON file."""
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    cutoff = datetime.utcnow() - timedelta(days=days)
    filtered = await _db.get_logs(patient_id, None, limit=5000, since=cutoff)
    profile = {k: v for k, v in user.items() if k not in ("password",)}
    export = {
        "export_date":  datetime.utcnow().isoformat(),
        "patient_id":   patient_id,
        "profile":      profile,
        "logs":         filtered,
        "total_logs":   len(filtered),
        "days":         days,
        "generated_by": "BioMirror v3",
    }
    return Response(
        content=json.dumps(export, indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition":
                 f"attachment; filename=biomirror-{patient_id}-export.json"}
    )


# ─────────────────────────────────────────────
# WEEKLY SUMMARY (v3)
# ─────────────────────────────────────────────

@app.get("/summary/{patient_id}/weekly", tags=["Analytics"])
async def weekly_summary(patient_id: str, user=Depends(get_current_user)):
    """Comprehensive 7-day health summary with trends."""
    if user["patient_id"] != patient_id:
        raise HTTPException(403, "Access denied")
    cutoff = datetime.utcnow() - timedelta(days=7)
    week_logs = await _db.get_logs(patient_id, None, limit=5000, since=cutoff)
    # Group by type
    by_type: Dict[str, list] = {}
    for l in week_logs:
        by_type.setdefault(l["type"], []).append(l)
    # Glucose stats
    g_vals = [l["glucose"] for l in by_type.get("glucose", []) if l.get("glucose") is not None]
    tir = round(sum(1 for g in g_vals if 70 <= g <= 180) / max(len(g_vals), 1) * 100, 1)
    mean_g = round(sum(g_vals) / len(g_vals), 1) if g_vals else None
    # Activity stats
    act_mins = sum(l.get("duration_min", 0) or 0 for l in by_type.get("activity", []))
    steps    = [l.get("steps", 0) for l in by_type.get("glucose", []) if l.get("steps")]
    avg_steps = round(sum(steps) / len(steps), 0) if steps else 0
    # Sleep stats
    sleep_hrs = [l.get("hours", 0) for l in by_type.get("sleep", [])]
    avg_sleep = round(sum(sleep_hrs) / len(sleep_hrs), 1) if sleep_hrs else None
    # Water stats
    water_ml = sum(l.get("amount_ml", 0) or 0 for l in by_type.get("water", []))
    # BP stats
    bp_logs = by_type.get("blood_pressure", [])
    avg_bp = None
    if bp_logs:
        avg_sys = round(sum(l.get("systolic", 0) or 0 for l in bp_logs) / len(bp_logs))
        avg_dia = round(sum(l.get("diastolic", 0) or 0 for l in bp_logs) / len(bp_logs))
        avg_bp  = f"{avg_sys}/{avg_dia}"
    meal_logs = by_type.get("meal", [])
    known_meals = [item for item in meal_logs
                   if item.get("carbs_g") is not None
                   and "unknown_food" not in (item.get("notes") or "")]
    meal_carbs = round(sum(item.get("carbs_g", 0) or 0 for item in known_meals), 1)
    meal_calories = round(sum(item.get("calories", 0) or 0 for item in known_meals
                              if item.get("calories") is not None))
    stress_logs = by_type.get("stress", [])
    high_stress = sum(1 for item in stress_logs if item.get("level") == "high")
    weight_logs = by_type.get("weight", [])
    latest_weight = weight_logs[-1] if weight_logs else None
    weight_change = (round((latest_weight.get("weight_kg") or 0) - (weight_logs[0].get("weight_kg") or 0), 1)
                     if len(weight_logs) > 1 else None)
    medication_logs = by_type.get("medication", [])
    medication_names = sorted({item["medication"] for item in medication_logs if item.get("medication")})
    # Insights
    insights = []
    if mean_g and mean_g > 160:
        insights.append(f"Average glucose ({mean_g} mg/dL) is elevated. Reduce carbs at lunch.")
    if tir < 65:
        insights.append(f"TIR of {tir}% is below the 70% target. Add a 15-min post-meal walk.")
    if avg_sleep and avg_sleep < 6.5:
        insights.append(f"Average sleep ({avg_sleep}h) is low. Poor sleep raises glucose 15–22%.")
    if act_mins < 150:
        insights.append(f"Only {act_mins}min exercise this week. WHO target is 150min/week.")
    if avg_steps < 5000:
        insights.append(f"Average {int(avg_steps)} steps/day. Aim for 8,000+ for best glucose control.")
    if meal_logs:
        insights.append(
            f"You logged {len(meal_logs)} meal(s) this week. Nutrition values are estimates "
            f"where available; {len(meal_logs) - len(known_meals)} meal(s) had no database match."
        )
    if stress_logs:
        insights.append(
            f"You recorded {len(stress_logs)} self-reported stress check-in(s), "
            f"including {high_stress} marked high."
        )
    stress_logs = by_type.get("stress", [])
    stress_high = sum(1 for item in stress_logs if item.get("level") == "high")
    if stress_logs:
        insights.append(
            f"You recorded stress {len(stress_logs)} times this week, including "
            f"{stress_high} high-stress check-in(s). These are self-reported observations."
        )
    if weight_logs:
        trend = f" Net change: {weight_change:+.1f} kg." if weight_change is not None else ""
        insights.append(f"You logged weight {len(weight_logs)} time(s) this week.{trend}")
    if medication_logs:
        insights.append(
            f"You recorded {len(medication_logs)} medication log(s) this week. "
            "This is a record only; discuss medication changes with your clinician."
        )
    if not insights:
        insights.append("Excellent week! Keep maintaining your healthy habits.")
    state = await load_twin(patient_id) or {}
    metrics = compute_metrics(state) if state else {}
    return {
        "patient_id": patient_id,
        "period": "Last 7 days",
        "logs_total": len(week_logs),
        "glucose": {
            "mean_mg_dl": mean_g, "tir_pct": tir,
            "readings": len(g_vals),
            "hypo_events": sum(1 for g in g_vals if g < 70),
            "hyper_events": sum(1 for g in g_vals if g > 180),
        },
        "activity": {"total_minutes": act_mins, "avg_daily_steps": int(avg_steps),
                     "sessions": len(by_type.get("activity", []))},
        "sleep":    {"avg_hours": avg_sleep,
                     "nights_logged": len(by_type.get("sleep", []))},
        "meals":    {"count": len(meal_logs), "known_nutrition_count": len(known_meals),
                 "estimated_carbohydrates_g": meal_carbs,
                 "estimated_calories": meal_calories},
        "stress":   {"check_ins": len(stress_logs), "high_check_ins": high_stress},
        "stress":   {"check_ins": len(stress_logs), "high_check_ins": stress_high},
        "hydration": {"total_ml": water_ml, "avg_daily_ml": round(water_ml/7)},
        "blood_pressure": {"avg": avg_bp, "readings": len(bp_logs)},
        "weight": {
            "entries": len(weight_logs),
            "latest_kg": latest_weight.get("weight_kg") if latest_weight else None,
            "latest_bmi": latest_weight.get("bmi") if latest_weight else None,
            "change_kg": weight_change,
        },
        "medication": {
            "entries": len(medication_logs),
            "taken_entries": sum(1 for item in medication_logs if item.get("taken")),
            "names": medication_names,
        },
        "current_reversal_score": metrics.get("reversal_score"),
        "current_hba1c_estimate": metrics.get("hba1c_estimate"),
        "insights": insights,
    }


# ─────────────────────────────────────────────
# ADMIN (v3)
# ─────────────────────────────────────────────

@app.get("/admin/users", tags=["Admin"])
async def list_users(user=Depends(get_current_user)):
    """Admin: list all registered users (role=admin required)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    all_users = await _db.list_users()
    users = []
    for u in all_users:
        state = await load_twin(u["patient_id"])
        m = compute_metrics(state) if state else {}
        logs_count = await _db.count_logs(u["patient_id"])
        users.append({
            "patient_id": u["patient_id"],
            "username":   u["username"],
            "full_name":  u.get("full_name"),
            "email":      u.get("email"),
            "registered_at": u.get("registered_at"),
            "profile_complete": u.get("profile_complete", False),
            "logs_count": logs_count,
            "current_glucose": m.get("glucose"),
            "reversal_score":  m.get("reversal_score"),
        })
    return {"total": len(users), "users": users}


@app.post("/admin/promote/{username}", tags=["Admin"])
async def promote_to_admin(username: str, user=Depends(get_current_user)):
    """Admin: promote a user to admin role."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    target = await _db.get_user_by_username(username)
    if not target:
        raise HTTPException(404, "User not found")
    await _db.update_user(username, {"role": "admin"})
    return {"success": True, "message": f"{username} promoted to admin"}


@app.get("/admin/analytics", tags=["Admin"])
async def admin_analytics(user=Depends(get_current_user)):
    """Admin: population-level analytics across all patients."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    all_glucose = await _db.all_glucose_values()
    avg_g = round(sum(all_glucose) / len(all_glucose), 1) if all_glucose else 0
    pop_tir = round(sum(1 for g in all_glucose if 70<=g<=180) / max(len(all_glucose),1) * 100, 1)
    return {
        "total_users":     await _db.count_users(),
        "active_twins":    await _db.count_twins(),
        "total_logs":      await _db.count_logs(),
        "population_mean_glucose": avg_g,
        "population_tir":  pop_tir,
        "total_simulations": await _db.count_predictions(),
    }


# ─────────────────────────────────────────────
# SYSTEM
# ─────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    db_status = {"type": "none", "status": "unavailable"}
    if _DB_AVAILABLE and _db:
        try:
            db_status = await _db.health_check()
        except Exception as e:
            db_status = {"type": "error", "status": str(e)}
    return {
        "status":           "operational",
        "service":          "BioMirror API v3",
        "version":          "3.2.0",
        "active_twins":     await _db.count_twins() if _DB_AVAILABLE else len(TWIN_CACHE),
        "registered_users": await _db.count_users() if _DB_AVAILABLE else 0,
        "total_logs":       await _db.count_logs() if _DB_AVAILABLE else 0,
        "active_ws":        sum(len(v) for v in WS_CONNECTIONS.values()),
        "anthropic_ai":     bool(ANTHROPIC_API_KEY),
        "database":         db_status,
        "ai_modules":       AI_STATUS,
        "timestamp":        datetime.utcnow().isoformat(),
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "project":     "BioMirror — Physics-Informed Digital Twin",
        "version":     "3.2.0",
        "docs":        "/docs",
        "health":      "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000,
                reload=True, workers=1, log_level="info")
