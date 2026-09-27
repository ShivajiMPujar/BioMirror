"""
BioMirror — Database Layer (PostgreSQL + SQLite, real persistence)
====================================================================
Phase 1 rewrite. Replaces every in-memory dict in backend_api.py with
real, restart-safe SQLAlchemy-backed storage.

Environment detection (unchanged behaviour/contract from v3):
  - DATABASE_URL set to a postgresql(+asyncpg) URL → PostgreSQL (production)
  - DATABASE_URL unset/empty                        → SQLite file ./biomirror.db (default — works out of the box)
  - SQLAlchemy/aiosqlite not installed               → in-memory dict fallback (dev-only, data lost on restart,
                                                         clearly reported as such via health_check())

This module is intentionally self-contained (no imports from backend_api.py)
so it can be imported standalone by scripts, tests, or the ai/ pipeline later.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

try:
    from sqlalchemy.ext.asyncio import (
        create_async_engine, AsyncSession, async_sessionmaker
    )
    from sqlalchemy import (
        Column, String, Float, Integer, Boolean,
        DateTime, Text, JSON, Index, select, delete, func
    )
    from sqlalchemy.orm import DeclarativeBase
    SQLA_OK = True
except ImportError:
    SQLA_OK = False

_RAW_URL = os.getenv("DATABASE_URL", "")
if _RAW_URL.startswith("postgresql"):
    DB_URL, DB_TYPE = _RAW_URL, "postgresql"
elif SQLA_OK:
    _sqlite = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "biomirror.db"))
    DB_URL, DB_TYPE = f"sqlite+aiosqlite:///{_sqlite}", "sqlite"
else:
    DB_URL, DB_TYPE = None, "memory"


# ─────────────────────────────────────────────────────────────
# ORM MODELS  (only defined if SQLAlchemy is actually installed)
# ─────────────────────────────────────────────────────────────
if SQLA_OK:

    class Base(DeclarativeBase):
        pass

    def _uid() -> str:
        return str(uuid.uuid4())

    # ---- Users & Auth -------------------------------------------------
    class UserRow(Base):
        __tablename__ = "users"
        id                  = Column(String(36), primary_key=True, default=_uid)
        username            = Column(String(50), unique=True, nullable=False, index=True)
        email               = Column(String(120), unique=True, nullable=False, index=True)
        password_hash       = Column(Text, nullable=False)
        full_name           = Column(String(100), default="")
        patient_id          = Column(String(20), unique=True, nullable=False, index=True)
        role                = Column(String(20), default="patient")
        # profile fields (flattened — matches PatientProfile pydantic model in backend_api.py)
        age                 = Column(Integer, nullable=True)
        gender              = Column(String(20), nullable=True)
        height_cm           = Column(Float, nullable=True)
        weight_kg           = Column(Float, nullable=True)
        bmi                 = Column(Float, default=27.0)
        diabetes_years      = Column(Integer, nullable=True)
        hba1c               = Column(Float, nullable=True)
        hba1c_date          = Column(String(20), nullable=True)
        baseline_glucose    = Column(Float, nullable=True)
        medications         = Column(JSON, default=list)
        activity_level      = Column(String(20), nullable=True)
        goal                = Column(String(30), nullable=True)
        diabetes_risk       = Column(Integer, default=1)
        family_history      = Column(Boolean, nullable=True)
        profile_complete    = Column(Boolean, default=False)
        # auth/lifecycle
        email_verified          = Column(Boolean, default=False)
        email_verification_token= Column(String(64), nullable=True)
        registered_at       = Column(DateTime, default=datetime.utcnow)
        updated_at          = Column(DateTime, nullable=True)
        password_changed_at = Column(DateTime, nullable=True)

    class RefreshTokenRow(Base):
        """Persisted so /auth/logout can actually revoke a refresh token, and
        /auth/refresh can reject revoked ones — the JWT alone can't do this."""
        __tablename__ = "refresh_tokens"
        id          = Column(String(36), primary_key=True, default=_uid)
        username    = Column(String(50), nullable=False, index=True)
        token       = Column(Text, nullable=False, unique=True, index=True)
        issued_at   = Column(DateTime, default=datetime.utcnow)
        expires_at  = Column(DateTime, nullable=False)
        revoked     = Column(Boolean, default=False)

    class PasswordResetTokenRow(Base):
        __tablename__ = "password_reset_tokens"
        id          = Column(String(36), primary_key=True, default=_uid)
        email       = Column(String(120), nullable=False, index=True)
        token       = Column(String(64), nullable=False, unique=True, index=True)
        created_at  = Column(DateTime, default=datetime.utcnow)
        expires_at  = Column(DateTime, nullable=False)
        used        = Column(Boolean, default=False)

    class SettingsRow(Base):
        __tablename__ = "settings"
        patient_id  = Column(String(20), primary_key=True)
        preferences = Column(JSON, default=dict)
        updated_at  = Column(DateTime, default=datetime.utcnow)

    # ---- Digital Twin ---------------------------------------------------
    class TwinStateRow(Base):
        __tablename__ = "twin_state"
        patient_id   = Column(String(20), primary_key=True)
        G            = Column(Float, nullable=False)
        X            = Column(Float, default=0.02)
        I            = Column(Float, default=12.0)
        p3           = Column(Float, default=3.5e-5)
        p3_index     = Column(Float, default=70.0)
        sim_time     = Column(Float, default=0.0)
        meal_queue   = Column(JSON, default=list)
        history      = Column(JSON, default=list)
        last_event   = Column(JSON, nullable=True)
        created_at   = Column(DateTime, default=datetime.utcnow)
        last_updated = Column(DateTime, default=datetime.utcnow)

    # ---- Health logs (one table per log type, per spec) ------------------
    class GlucoseLogRow(Base):
        __tablename__ = "glucose_logs"
        id           = Column(String(36), primary_key=True, default=_uid)
        patient_id   = Column(String(20), nullable=False, index=True)
        timestamp    = Column(DateTime, default=datetime.utcnow, index=True)
        glucose      = Column(Float, nullable=False)
        meal_type    = Column(String(30), nullable=True)
        carbs        = Column(Float, default=0)
        sugar        = Column(Float, default=0)
        food         = Column(String(200), nullable=True)
        steps        = Column(Integer, default=0)
        exercise_min = Column(Integer, default=0)
        heart_rate   = Column(Integer, default=80)
        stress_level = Column(Integer, default=2)
        notes        = Column(Text, nullable=True)
        post_G       = Column(Float, nullable=True)

    class MealLogRow(Base):
        __tablename__ = "meal_logs"
        id         = Column(String(36), primary_key=True, default=_uid)
        patient_id = Column(String(20), nullable=False, index=True)
        timestamp  = Column(DateTime, default=datetime.utcnow, index=True)
        food_name  = Column(String(200), nullable=False)
        carbs_g    = Column(Float, nullable=True)
        protein_g  = Column(Float, nullable=True)
        fat_g      = Column(Float, nullable=True)
        fiber_g    = Column(Float, nullable=True)
        calories   = Column(Float, nullable=True)
        meal_type  = Column(String(30), default="meal")
        gi_score   = Column(Float, nullable=True)
        notes      = Column(Text, nullable=True)
        # Phase 2 — persisted output of ai_bridge.predict_meal() (food_database.py)
        glycemic_load    = Column(Float, nullable=True)
        predicted_peak_glucose = Column(Float, nullable=True)
        spike_risk       = Column(String(10), nullable=True)   # LOW/MEDIUM/HIGH
        time_to_peak_min = Column(Integer, nullable=True)

    class ActivityLogRow(Base):
        __tablename__ = "activity_logs"
        id              = Column(String(36), primary_key=True, default=_uid)
        patient_id      = Column(String(20), nullable=False, index=True)
        timestamp       = Column(DateTime, default=datetime.utcnow, index=True)
        activity        = Column(String(100), nullable=False)
        duration_min    = Column(Integer, default=0)
        steps           = Column(Integer, default=0)
        calories        = Column(Float, default=0)
        heart_rate_avg  = Column(Integer, default=0)
        intensity       = Column(String(20), default="moderate")

    class SleepLogRow(Base):
        __tablename__ = "sleep_logs"
        id         = Column(String(36), primary_key=True, default=_uid)
        patient_id = Column(String(20), nullable=False, index=True)
        timestamp  = Column(DateTime, default=datetime.utcnow, index=True)
        hours      = Column(Float, nullable=False)
        quality    = Column(Integer, default=3)
        bed_time   = Column(String(10), nullable=True)
        wake_time  = Column(String(10), nullable=True)

    class MedicationLogRow(Base):
        __tablename__ = "medication_logs"
        id         = Column(String(36), primary_key=True, default=_uid)
        patient_id = Column(String(20), nullable=False, index=True)
        timestamp  = Column(DateTime, default=datetime.utcnow, index=True)
        medication = Column(String(100), nullable=False)
        dose       = Column(String(50), nullable=True)
        taken      = Column(Boolean, default=True)
        notes      = Column(Text, nullable=True)

    class WeightLogRow(Base):
        __tablename__ = "weight_logs"
        id           = Column(String(36), primary_key=True, default=_uid)
        patient_id   = Column(String(20), nullable=False, index=True)
        timestamp    = Column(DateTime, default=datetime.utcnow, index=True)
        weight_kg    = Column(Float, nullable=False)
        bmi          = Column(Float, nullable=True)
        body_fat_pct = Column(Float, nullable=True)

    class BloodPressureLogRow(Base):
        __tablename__ = "blood_pressure_logs"
        id         = Column(String(36), primary_key=True, default=_uid)
        patient_id = Column(String(20), nullable=False, index=True)
        timestamp  = Column(DateTime, default=datetime.utcnow, index=True)
        systolic   = Column(Integer, nullable=False)
        diastolic  = Column(Integer, nullable=False)
        pulse      = Column(Integer, default=80)
        notes      = Column(Text, nullable=True)

    class WaterLogRow(Base):
        __tablename__ = "water_logs"
        id          = Column(String(36), primary_key=True, default=_uid)
        patient_id  = Column(String(20), nullable=False, index=True)
        timestamp   = Column(DateTime, default=datetime.utcnow, index=True)
        amount_ml   = Column(Integer, nullable=False)
        time_of_day = Column(String(20), nullable=True)

    class HbA1cLogRow(Base):
        __tablename__ = "hba1c_logs"
        id         = Column(String(36), primary_key=True, default=_uid)
        patient_id = Column(String(20), nullable=False, index=True)
        timestamp  = Column(DateTime, default=datetime.utcnow, index=True)
        hba1c      = Column(Float, nullable=False)
        lab_name   = Column(String(100), nullable=True)
        test_date  = Column(String(20), nullable=True)

    class StressLogRow(Base):
        __tablename__ = "stress_logs"
        id           = Column(String(36), primary_key=True, default=_uid)
        patient_id   = Column(String(20), nullable=False, index=True)
        timestamp    = Column(DateTime, default=datetime.utcnow, index=True)
        level        = Column(String(20), nullable=False)
        stress_level = Column(Integer, nullable=False)

    class ChatMessageRow(Base):
        __tablename__ = "chat_messages"
        id         = Column(String(36), primary_key=True, default=_uid)
        patient_id = Column(String(20), nullable=False, index=True)
        timestamp  = Column(DateTime, default=datetime.utcnow, index=True)
        role       = Column(String(10), nullable=False)   # "user" | "assistant"
        message    = Column(Text, nullable=False)

    class GoalRow(Base):
        __tablename__ = "goals"
        patient_id     = Column(String(20), primary_key=True)
        target_hba1c   = Column(Float, nullable=True)
        target_weight  = Column(Float, nullable=True)
        target_steps   = Column(Integer, nullable=True)
        target_tir     = Column(Float, nullable=True)
        target_date    = Column(String(20), nullable=True)
        created_at     = Column(DateTime, default=datetime.utcnow)
        updated_at     = Column(DateTime, default=datetime.utcnow)

    class NotificationRow(Base):
        __tablename__ = "notifications"
        id         = Column(String(36), primary_key=True, default=_uid)
        patient_id = Column(String(20), nullable=False, index=True)
        title      = Column(String(200), nullable=False)
        body       = Column(Text, nullable=False)
        type       = Column(String(20), default="info")
        read       = Column(Boolean, default=False)
        created_at = Column(DateTime, default=datetime.utcnow, index=True)

    class PredictionHistoryRow(Base):
        """Prediction history — covers both What-If simulator runs
        (source='simulate', pre-existing) and live PINN/Bergman forecasts
        (source='forecast', added in Phase 2). Reused rather than split
        into two tables since both represent the same underlying concept:
        'a prediction was made at time T, here's what went into it and
        what came out.'"""
        __tablename__ = "prediction_history"
        id            = Column(String(36), primary_key=True, default=_uid)
        patient_id    = Column(String(20), nullable=False, index=True)
        timestamp     = Column(DateTime, default=datetime.utcnow, index=True)
        source        = Column(String(20), default="simulate")   # 'simulate' | 'forecast'
        inputs        = Column(JSON, default=dict)
        engineered_features = Column(JSON, nullable=True)   # Phase 2: feature_engineering.py output
        predicted_glucose   = Column(JSON, nullable=True)   # Phase 2: list of forecasted mg/dL values
        peak_glucose  = Column(Float, nullable=True)
        spike_pct     = Column(Float, nullable=True)
        tir_pct       = Column(Float, nullable=True)
        recovery_h    = Column(Float, nullable=True)
        confidence    = Column(Float, nullable=True)        # Phase 2
        uncertainty_std = Column(Float, nullable=True)      # Phase 2: MC-dropout std
        inference_time_ms = Column(Float, nullable=True)    # Phase 2
        model_version = Column(String(40), nullable=True)   # Phase 2: 'pinn-v1-...' or 'bergman_fallback'
        model_status  = Column(String(30), nullable=True)   # Phase 2: 'pinn' | 'bergman_fallback' | 'pinn_unavailable'
        shap_explanation = Column(JSON, nullable=True)      # Phase 2
        reversal_score = Column(Float, nullable=True)       # Phase 2

    class ReversalScoreHistoryRow(Base):
        """Historical reversal/lifestyle/recovery score snapshots — Phase 2.
        One row per computation lets /reversal/{id} report week-over-week
        improvement by comparing against a row from ~7 days ago, instead
        of a fabricated number."""
        __tablename__ = "reversal_score_history"
        id            = Column(String(36), primary_key=True, default=_uid)
        patient_id    = Column(String(20), nullable=False, index=True)
        timestamp     = Column(DateTime, default=datetime.utcnow, index=True)
        reversal_score            = Column(Float, nullable=True)
        lifestyle_score           = Column(Float, nullable=True)
        recovery_index            = Column(Float, nullable=True)
        insulin_sensitivity_score = Column(Float, nullable=True)
        metabolic_health_score    = Column(Float, nullable=True)
        recovery_stage            = Column(String(30), nullable=True)
        component_scores          = Column(JSON, nullable=True)

    class RecommendationHistoryRow(Base):
        """Personalized recommendation history — Phase 2."""
        __tablename__ = "recommendation_history"
        id            = Column(String(36), primary_key=True, default=_uid)
        patient_id    = Column(String(20), nullable=False, index=True)
        timestamp     = Column(DateTime, default=datetime.utcnow, index=True)
        recommendations = Column(JSON, default=list)
        basis         = Column(JSON, nullable=True)   # which data signals drove these recommendations

    class EvaluationMetricsRow(Base):
        """Retrospective prediction-accuracy tracking — Phase 2. Computed
        by comparing stored predictions against the actual glucose logged
        afterward (see /metrics/evaluation logic in backend_api.py)."""
        __tablename__ = "evaluation_metrics_history"
        id            = Column(String(36), primary_key=True, default=_uid)
        patient_id    = Column(String(20), nullable=False, index=True)
        timestamp     = Column(DateTime, default=datetime.utcnow, index=True)
        mae           = Column(Float, nullable=True)
        rmse          = Column(Float, nullable=True)
        mard          = Column(Float, nullable=True)
        r2            = Column(Float, nullable=True)
        n_samples     = Column(Integer, nullable=True)

    class PersonalFeaturesRow(Base):
        """Phase 3 Part D — per-user statistical features computed from that
        user's own historical data (not population-wide constants). One
        row per patient, overwritten on every recompute; history of how
        these evolve is implicitly available via prediction_history's
        engineered_features snapshots taken at prediction time."""
        __tablename__ = "personal_features"
        patient_id     = Column(String(20), primary_key=True)
        computed_at    = Column(DateTime, default=datetime.utcnow)
        days_of_data   = Column(Integer, default=0)   # how much history this was computed from — low = low confidence
        features       = Column(JSON, default=dict)   # all Part D features, see compute_personal_features() in backend_api.py

    class TwinSnapshotRow(Base):
        """Phase 3 Part G — Digital Twin memory. Periodic (daily/weekly)
        snapshots of twin state + confidence/calibration, distinct from the
        single live TwinStateRow (which is always just the current state)."""
        __tablename__ = "twin_snapshots"
        id             = Column(String(36), primary_key=True, default=_uid)
        patient_id     = Column(String(20), nullable=False, index=True)
        timestamp      = Column(DateTime, default=datetime.utcnow, index=True)
        period         = Column(String(10), default="daily")   # 'daily' | 'weekly'
        G              = Column(Float, nullable=True)
        p3_index       = Column(Float, nullable=True)
        reversal_score = Column(Float, nullable=True)
        tir_pct        = Column(Float, nullable=True)
        confidence     = Column(Float, nullable=True)   # how much this snapshot should be trusted, grows with days_of_data
        calibration    = Column(Float, nullable=True)   # 1 - normalized recent prediction error; from evaluation_metrics_history
        __table_args__ = (Index("ix_twin_snapshot_patient_period", "patient_id", "period"),)

    # Registry used by the generic log helpers below. Keys match the
    # `log_type` strings already used throughout backend_api.py.
    LOG_MODELS = {
        "glucose":         GlucoseLogRow,
        "meal":            MealLogRow,
        "activity":        ActivityLogRow,
        "sleep":           SleepLogRow,
        "medication":      MedicationLogRow,
        "weight":          WeightLogRow,
        "blood_pressure":  BloodPressureLogRow,
        "water":           WaterLogRow,
        "hba1c":           HbA1cLogRow,
        "stress":          StressLogRow,
    }


# ─────────────────────────────────────────────────────────────
# DATABASE MANAGER
# ─────────────────────────────────────────────────────────────
class DatabaseManager:
    """Async facade used by backend_api.py. Every method here is backed by
    a real SQLAlchemy table when SQLA_OK/engine init succeeds; if SQLAlchemy
    itself isn't installed, falls back to process-local dicts (memory mode)
    so the app still boots, but this is clearly reported via health_check()
    and startup logging — it is NOT the default path."""

    def __init__(self):
        self.engine = None
        self.Session = None
        self.ready = False
        self._mem: Dict[str, Any] = {
            "users": {}, "refresh_tokens": {}, "reset_tokens": {},
            "twin": {}, "logs": {k: [] for k in
                (LOG_MODELS.keys() if SQLA_OK else [])},
            "chat": [], "goals": {}, "notifications": [],
            "predictions": [], "settings": {},
            "reversal_history": [], "recommendations": [], "eval_metrics": [],
            "personal_features": {}, "twin_snapshots": [],
        }

    # ---- lifecycle ----------------------------------------------------
    async def init(self):
        if not SQLA_OK or not DB_URL:
            self.ready = False
            return
        self.engine = create_async_engine(DB_URL, echo=False, future=True)
        self.Session = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.ready = True

    async def close(self):
        if self.engine:
            await self.engine.dispose()

    async def health_check(self) -> dict:
        if not SQLA_OK:
            return {"type": "memory", "status": "degraded",
                    "warning": "SQLAlchemy/aiosqlite not installed — data will NOT survive a restart. "
                               "Run: pip install sqlalchemy aiosqlite --break-system-packages"}
        if not self.ready:
            return {"type": DB_TYPE, "status": "not_initialized"}
        try:
            async with self.Session() as s:
                await s.execute(select(1))
            return {"type": DB_TYPE, "status": "connected", "url": DB_URL.split("@")[-1]}
        except Exception as e:
            return {"type": DB_TYPE, "status": "error", "detail": str(e)}

    def _row_to_dict(self, row) -> dict:
        if row is None:
            return None
        d = {c.name: getattr(row, c.name) for c in row.__table__.columns}
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d

    # ---- USERS ----------------------------------------------------------
    async def create_user(self, profile: dict) -> dict:
        if not self.ready:
            self._mem["users"][profile["username"]] = profile
            return profile
        async with self.Session() as s:
            row = UserRow(
                username=profile["username"], email=profile["email"],
                password_hash=profile["password"], full_name=profile.get("full_name", ""),
                patient_id=profile["patient_id"], role=profile.get("role", "patient"),
                bmi=profile.get("bmi", 27.0), baseline_glucose=profile.get("baseline_glucose"),
                diabetes_risk=profile.get("diabetes_risk", 1),
                profile_complete=profile.get("profile_complete", False),
                email_verified=False,
                email_verification_token=profile.get("email_verification_token"),
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            d = self._row_to_dict(row)
            d["password"] = d.pop("password_hash")   # keep the key name backend_api.py expects
            return d

    async def _get_user_row(self, s, username=None, email=None, patient_id=None):
        if username:
            q = select(UserRow).where(UserRow.username == username)
        elif email:
            q = select(UserRow).where(UserRow.email == email)
        else:
            q = select(UserRow).where(UserRow.patient_id == patient_id)
        return (await s.execute(q)).scalar_one_or_none()

    def _user_dict(self, row) -> Optional[dict]:
        if row is None:
            return None
        d = self._row_to_dict(row)
        d["password"] = d.pop("password_hash")
        return d

    async def get_user_by_username(self, username: str) -> Optional[dict]:
        if not self.ready:
            return self._mem["users"].get(username)
        async with self.Session() as s:
            return self._user_dict(await self._get_user_row(s, username=username))

    async def get_user_by_email(self, email: str) -> Optional[dict]:
        if not self.ready:
            return next((u for u in self._mem["users"].values() if u.get("email") == email), None)
        async with self.Session() as s:
            return self._user_dict(await self._get_user_row(s, email=email))

    async def get_user_by_patient_id(self, patient_id: str) -> Optional[dict]:
        if not self.ready:
            return next((u for u in self._mem["users"].values() if u.get("patient_id") == patient_id), None)
        async with self.Session() as s:
            return self._user_dict(await self._get_user_row(s, patient_id=patient_id))

    async def update_user(self, username: str, updates: dict) -> Optional[dict]:
        if not self.ready:
            if username in self._mem["users"]:
                self._mem["users"][username].update(updates)
                return self._mem["users"][username]
            return None
        async with self.Session() as s:
            row = await self._get_user_row(s, username=username)
            if not row:
                return None
            for k, v in updates.items():
                if k == "password":
                    row.password_hash = v
                elif hasattr(row, k):
                    setattr(row, k, v)
            row.updated_at = datetime.utcnow()
            await s.commit()
            await s.refresh(row)
            return self._user_dict(row)

    async def list_users(self) -> List[dict]:
        if not self.ready:
            return list(self._mem["users"].values())
        async with self.Session() as s:
            rows = (await s.execute(select(UserRow))).scalars().all()
            return [self._user_dict(r) for r in rows]

    async def count_users(self) -> int:
        if not self.ready:
            return len(self._mem["users"])
        async with self.Session() as s:
            return (await s.execute(select(func.count()).select_from(UserRow))).scalar_one()

    # ---- REFRESH TOKENS ---------------------------------------------------
    async def save_refresh_token(self, username: str, token: str, expires_at: datetime):
        if not self.ready:
            self._mem["refresh_tokens"][token] = {"username": username, "revoked": False, "expires_at": expires_at}
            return
        async with self.Session() as s:
            s.add(RefreshTokenRow(username=username, token=token, expires_at=expires_at))
            await s.commit()

    async def is_refresh_token_valid(self, token: str) -> bool:
        if not self.ready:
            rec = self._mem["refresh_tokens"].get(token)
            return bool(rec and not rec["revoked"])
        async with self.Session() as s:
            row = (await s.execute(select(RefreshTokenRow).where(RefreshTokenRow.token == token))).scalar_one_or_none()
            return bool(row and not row.revoked)

    async def revoke_refresh_token(self, token: str):
        if not self.ready:
            if token in self._mem["refresh_tokens"]:
                self._mem["refresh_tokens"][token]["revoked"] = True
            return
        async with self.Session() as s:
            row = (await s.execute(select(RefreshTokenRow).where(RefreshTokenRow.token == token))).scalar_one_or_none()
            if row:
                row.revoked = True
                await s.commit()

    async def revoke_all_user_tokens(self, username: str):
        if not self.ready:
            for rec in self._mem["refresh_tokens"].values():
                if rec["username"] == username:
                    rec["revoked"] = True
            return
        async with self.Session() as s:
            rows = (await s.execute(select(RefreshTokenRow).where(RefreshTokenRow.username == username))).scalars().all()
            for r in rows:
                r.revoked = True
            await s.commit()

    # ---- PASSWORD RESET ----------------------------------------------------
    async def create_reset_token(self, email: str) -> str:
        token = uuid.uuid4().hex
        expires = datetime.utcnow() + timedelta(hours=1)
        if not self.ready:
            self._mem["reset_tokens"][token] = {"email": email, "expires_at": expires, "used": False}
            return token
        async with self.Session() as s:
            s.add(PasswordResetTokenRow(email=email, token=token, expires_at=expires))
            await s.commit()
        return token

    async def consume_reset_token(self, token: str) -> Optional[str]:
        """Returns the email if the token is valid & unused, marks it used, else None."""
        if not self.ready:
            rec = self._mem["reset_tokens"].get(token)
            if not rec or rec["used"] or rec["expires_at"] < datetime.utcnow():
                return None
            rec["used"] = True
            return rec["email"]
        async with self.Session() as s:
            row = (await s.execute(select(PasswordResetTokenRow).where(PasswordResetTokenRow.token == token))).scalar_one_or_none()
            if not row or row.used or row.expires_at < datetime.utcnow():
                return None
            row.used = True
            await s.commit()
            return row.email

    # ---- EMAIL VERIFICATION -------------------------------------------------
    async def verify_email_token(self, token: str) -> bool:
        if not self.ready:
            for u in self._mem["users"].values():
                if u.get("email_verification_token") == token:
                    u["email_verified"] = True
                    return True
            return False
        async with self.Session() as s:
            row = (await s.execute(select(UserRow).where(UserRow.email_verification_token == token))).scalar_one_or_none()
            if not row:
                return False
            row.email_verified = True
            await s.commit()
            return True

    # ---- TWIN STATE ---------------------------------------------------------
    async def get_twin_state(self, patient_id: str) -> Optional[dict]:
        if not self.ready:
            return self._mem["twin"].get(patient_id)
        async with self.Session() as s:
            row = (await s.execute(select(TwinStateRow).where(TwinStateRow.patient_id == patient_id))).scalar_one_or_none()
            if not row:
                return None
            return {
                "patient_id": row.patient_id, "G": row.G, "X": row.X, "I": row.I,
                "p3": row.p3, "p3_index": row.p3_index, "meal_queue": row.meal_queue or [],
                "sim_time": row.sim_time, "history": row.history or [], "last_event": row.last_event,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "last_updated": row.last_updated.isoformat() if row.last_updated else None,
            }

    async def save_twin_state(self, patient_id: str, state: dict):
        if not self.ready:
            self._mem["twin"][patient_id] = state
            return
        async with self.Session() as s:
            row = (await s.execute(select(TwinStateRow).where(TwinStateRow.patient_id == patient_id))).scalar_one_or_none()
            if not row:
                row = TwinStateRow(patient_id=patient_id)
                s.add(row)
            row.G, row.X, row.I = state["G"], state["X"], state["I"]
            row.p3, row.p3_index = state["p3"], state["p3_index"]
            row.sim_time = state["sim_time"]
            row.meal_queue = state.get("meal_queue", [])
            row.history = state.get("history", [])
            row.last_event = state.get("last_event")
            row.last_updated = datetime.utcnow()
            await s.commit()

    async def count_twins(self) -> int:
        if not self.ready:
            return len(self._mem["twin"])
        async with self.Session() as s:
            return (await s.execute(select(func.count()).select_from(TwinStateRow))).scalar_one()

    # ---- GENERIC HEALTH LOGS -------------------------------------------------
    async def add_log(self, patient_id: str, log_type: str, data: dict) -> dict:
        ts = data.pop("timestamp", None) or datetime.utcnow()
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.utcnow()
        entry = {"id": str(uuid.uuid4()), "type": log_type, "patient_id": patient_id,
                 "timestamp": ts.isoformat(), **data}
        if not self.ready:
            self._mem["logs"].setdefault(log_type, []).append({**entry})
            return entry
        model = LOG_MODELS.get(log_type)
        if not model:
            # Unsupported log types stay in the in-memory fallback only.
            self._mem["logs"].setdefault(log_type, []).append({**entry})
            return entry
        cols = {c.name for c in model.__table__.columns}
        clean = {k: v for k, v in data.items() if k in cols}
        async with self.Session() as s:
            row = model(patient_id=patient_id, timestamp=ts, **clean)
            s.add(row)
            await s.commit()
            await s.refresh(row)
            d = self._row_to_dict(row)
            d["type"] = log_type
            return d

    async def get_logs(self, patient_id: str, log_type: Optional[str] = None,
                        limit: int = 500, since: Optional[datetime] = None) -> List[dict]:
        """Mirrors old LOGS_DB.get(patient_id) — returns entries across all
        log types (or a single type) sorted oldest→newest, capped at limit."""
        if not self.ready:
            pool = (self._mem["logs"].get(log_type, []) if log_type
                    else [e for lst in self._mem["logs"].values() for e in lst])
            pool = [e for e in pool if e["patient_id"] == patient_id]
            if since:
                pool = [e for e in pool if e["timestamp"] >= since.isoformat()]
            return sorted(pool, key=lambda e: e["timestamp"])[-limit:]

        results = []
        types = [log_type] if log_type else list(LOG_MODELS.keys())
        async with self.Session() as s:
            for t in types:
                model = LOG_MODELS[t]
                q = select(model).where(model.patient_id == patient_id)
                if since:
                    q = q.where(model.timestamp >= since)
                q = q.order_by(model.timestamp.desc()).limit(limit)
                rows = (await s.execute(q)).scalars().all()
                for r in rows:
                    d = self._row_to_dict(r)
                    d["type"] = t
                    # normalise field name so existing frontend/analytics code that
                    # reads `carbs_g` for meals or `carbs` for glucose keeps working
                    results.append(d)
        results.sort(key=lambda e: e["timestamp"])
        return results[-limit:]

    async def count_logs(self, patient_id: Optional[str] = None) -> int:
        if not self.ready:
            pool = [e for lst in self._mem["logs"].values() for e in lst]
            return len([e for e in pool if patient_id is None or e["patient_id"] == patient_id])
        total = 0
        async with self.Session() as s:
            for model in LOG_MODELS.values():
                q = select(func.count()).select_from(model)
                if patient_id:
                    q = q.where(model.patient_id == patient_id)
                total += (await s.execute(q)).scalar_one()
        return total

    async def all_glucose_values(self) -> List[float]:
        if not self.ready:
            return [e["glucose"] for e in self._mem["logs"].get("glucose", []) if e.get("glucose")]
        async with self.Session() as s:
            rows = (await s.execute(select(GlucoseLogRow.glucose))).scalars().all()
            return [g for g in rows if g is not None]

    # ---- PERSONAL FEATURES (Phase 3) -------------------------------------------
    async def save_personal_features(self, patient_id: str, features: dict, days_of_data: int) -> dict:
        entry = {"patient_id": patient_id, "features": features, "days_of_data": days_of_data,
                  "computed_at": datetime.utcnow().isoformat()}
        if not self.ready:
            self._mem.setdefault("personal_features", {})[patient_id] = entry
            return entry
        async with self.Session() as s:
            row = (await s.execute(select(PersonalFeaturesRow)
                   .where(PersonalFeaturesRow.patient_id == patient_id))).scalar_one_or_none()
            if not row:
                row = PersonalFeaturesRow(patient_id=patient_id)
                s.add(row)
            row.features = features
            row.days_of_data = days_of_data
            row.computed_at = datetime.utcnow()
            await s.commit()
        return entry

    async def get_personal_features(self, patient_id: str) -> Optional[dict]:
        if not self.ready:
            return self._mem.get("personal_features", {}).get(patient_id)
        async with self.Session() as s:
            row = (await s.execute(select(PersonalFeaturesRow)
                   .where(PersonalFeaturesRow.patient_id == patient_id))).scalar_one_or_none()
            return self._row_to_dict(row) if row else None

    # ---- TWIN SNAPSHOTS / MEMORY (Phase 3) -------------------------------------
    async def add_twin_snapshot(self, patient_id: str, period: str, data: dict) -> dict:
        entry = {"id": str(uuid.uuid4()), "timestamp": datetime.utcnow().isoformat(), **data}
        if not self.ready:
            self._mem.setdefault("twin_snapshots", []).append({**entry, "patient_id": patient_id, "period": period})
            return entry
        async with self.Session() as s:
            row = TwinSnapshotRow(
                id=entry["id"], patient_id=patient_id, period=period,
                G=data.get("G"), p3_index=data.get("p3_index"),
                reversal_score=data.get("reversal_score"), tir_pct=data.get("tir_pct"),
                confidence=data.get("confidence"), calibration=data.get("calibration"))
            s.add(row)
            await s.commit()
        return entry

    async def get_twin_snapshots(self, patient_id: str, period: Optional[str] = None,
                                  limit: int = 60) -> List[dict]:
        if not self.ready:
            pool = [t for t in self._mem.get("twin_snapshots", []) if t["patient_id"] == patient_id]
            if period:
                pool = [t for t in pool if t["period"] == period]
            return pool[-limit:]
        async with self.Session() as s:
            q = select(TwinSnapshotRow).where(TwinSnapshotRow.patient_id == patient_id)
            if period:
                q = q.where(TwinSnapshotRow.period == period)
            q = q.order_by(TwinSnapshotRow.timestamp.desc()).limit(limit)
            rows = (await s.execute(q)).scalars().all()
            return [self._row_to_dict(r) for r in reversed(rows)]

    async def has_twin_snapshot_today(self, patient_id: str, period: str = "daily") -> bool:
        if not self.ready:
            today = datetime.utcnow().date().isoformat()
            return any(t["patient_id"] == patient_id and t["period"] == period
                       and t["timestamp"][:10] == today for t in self._mem.get("twin_snapshots", []))
        async with self.Session() as s:
            today_start = datetime.combine(datetime.utcnow().date(), datetime.min.time())
            q = select(func.count()).select_from(TwinSnapshotRow).where(
                TwinSnapshotRow.patient_id == patient_id, TwinSnapshotRow.period == period,
                TwinSnapshotRow.timestamp >= today_start)
            return (await s.execute(q)).scalar_one() > 0

    # ---- AI COACH CHAT HISTORY ----------------------------------------------
    async def add_chat_message(self, patient_id: str, role: str, message: str):
        if not self.ready:
            self._mem["chat"].append({"patient_id": patient_id, "role": role,
                                       "message": message, "timestamp": datetime.utcnow().isoformat()})
            return
        async with self.Session() as s:
            s.add(ChatMessageRow(patient_id=patient_id, role=role, message=message))
            await s.commit()

    async def get_chat_history(self, patient_id: str, limit: int = 40) -> List[dict]:
        if not self.ready:
            return [e for e in self._mem["chat"] if e["patient_id"] == patient_id][-limit:]
        async with self.Session() as s:
            q = (select(ChatMessageRow).where(ChatMessageRow.patient_id == patient_id)
                 .order_by(ChatMessageRow.timestamp.desc()).limit(limit))
            rows = (await s.execute(q)).scalars().all()
            return [self._row_to_dict(r) for r in reversed(rows)]

    async def clear_chat_history(self, patient_id: str):
        if not self.ready:
            self._mem["chat"] = [e for e in self._mem["chat"] if e["patient_id"] != patient_id]
            return
        async with self.Session() as s:
            await s.execute(delete(ChatMessageRow).where(ChatMessageRow.patient_id == patient_id))
            await s.commit()

    # ---- GOALS --------------------------------------------------------------
    async def set_goals(self, patient_id: str, data: dict) -> dict:
        if not self.ready:
            self._mem["goals"][patient_id] = {**data, "patient_id": patient_id}
            return self._mem["goals"][patient_id]
        async with self.Session() as s:
            row = (await s.execute(select(GoalRow).where(GoalRow.patient_id == patient_id))).scalar_one_or_none()
            if not row:
                row = GoalRow(patient_id=patient_id)
                s.add(row)
            for k, v in data.items():
                if hasattr(row, k):
                    setattr(row, k, v)
            row.updated_at = datetime.utcnow()
            await s.commit()
            await s.refresh(row)
            return self._row_to_dict(row)

    async def remove_goal(self, patient_id: str, goal_field: str) -> dict:
        goal_fields = ("target_hba1c", "target_weight", "target_steps", "target_tir", "target_date")
        if goal_field not in goal_fields:
            raise ValueError("Unsupported goal field")
        if not self.ready:
            goals = self._mem["goals"].get(patient_id, {})
            goals.pop(goal_field, None)
            if not any(goals.get(field) is not None for field in goal_fields):
                self._mem["goals"].pop(patient_id, None)
            return await self.get_goals(patient_id)

        async with self.Session() as s:
            row = (await s.execute(select(GoalRow).where(GoalRow.patient_id == patient_id))).scalar_one_or_none()
            if row:
                setattr(row, goal_field, None)
                row.updated_at = datetime.utcnow()
                has_goals = any(getattr(row, field) is not None for field in goal_fields)
                if not has_goals:
                    await s.delete(row)
                await s.commit()
        return await self.get_goals(patient_id)

    async def get_goals(self, patient_id: str) -> dict:
        if not self.ready:
            return self._mem["goals"].get(patient_id, {})
        async with self.Session() as s:
            row = (await s.execute(select(GoalRow).where(GoalRow.patient_id == patient_id))).scalar_one_or_none()
            if not row:
                return {}
            d = self._row_to_dict(row)
            return {k: v for k, v in d.items() if v is not None}

    # ---- NOTIFICATIONS --------------------------------------------------------
    async def push_notification(self, patient_id: str, title: str, body: str, ntype: str = "info") -> dict:
        entry = {"id": str(uuid.uuid4()), "title": title, "body": body, "type": ntype,
                  "read": False, "created_at": datetime.utcnow().isoformat()}
        if not self.ready:
            self._mem["notifications"].append({**entry, "patient_id": patient_id})
            return entry
        async with self.Session() as s:
            row = NotificationRow(id=entry["id"], patient_id=patient_id, title=title,
                                   body=body, type=ntype, read=False)
            s.add(row)
            await s.commit()
        return entry

    async def get_notifications(self, patient_id: str, unread_only: bool = False) -> List[dict]:
        if not self.ready:
            notifs = [n for n in self._mem["notifications"] if n["patient_id"] == patient_id]
            if unread_only:
                notifs = [n for n in notifs if not n["read"]]
            return list(reversed(notifs))
        async with self.Session() as s:
            q = select(NotificationRow).where(NotificationRow.patient_id == patient_id)
            if unread_only:
                q = q.where(NotificationRow.read == False)  # noqa: E712
            q = q.order_by(NotificationRow.created_at.desc()).limit(200)
            rows = (await s.execute(q)).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    async def count_unread_notifications(self, patient_id: str) -> int:
        if not self.ready:
            return sum(1 for n in self._mem["notifications"] if n["patient_id"] == patient_id and not n["read"])
        async with self.Session() as s:
            q = select(func.count()).select_from(NotificationRow).where(
                NotificationRow.patient_id == patient_id, NotificationRow.read == False)  # noqa: E712
            return (await s.execute(q)).scalar_one()

    async def mark_notifications_read(self, patient_id: str, ids: Optional[List[str]] = None) -> int:
        if not self.ready:
            count = 0
            for n in self._mem["notifications"]:
                if n["patient_id"] == patient_id and (ids is None or n["id"] in ids):
                    n["read"] = True
                    count += 1
            return count
        async with self.Session() as s:
            q = select(NotificationRow).where(NotificationRow.patient_id == patient_id)
            if ids:
                q = q.where(NotificationRow.id.in_(ids))
            rows = (await s.execute(q)).scalars().all()
            for r in rows:
                r.read = True
            await s.commit()
            return len(rows)

    # ---- PREDICTION / SIMULATION HISTORY ---------------------------------------
    async def add_prediction(self, patient_id: str, data: dict) -> dict:
        entry = {"id": str(uuid.uuid4()), "timestamp": datetime.utcnow().isoformat(), **data}
        if not self.ready:
            self._mem["predictions"].append({**entry, "patient_id": patient_id})
            return entry
        async with self.Session() as s:
            row = PredictionHistoryRow(
                id=entry["id"], patient_id=patient_id, inputs=data.get("inputs", {}),
                source=data.get("source", "simulate"),
                peak_glucose=data.get("peak_glucose"), spike_pct=data.get("spike_pct"),
                tir_pct=data.get("tir_pct"), recovery_h=data.get("recovery_h"),
                engineered_features=data.get("engineered_features"),
                predicted_glucose=data.get("predicted_glucose"),
                confidence=data.get("confidence"),
                uncertainty_std=data.get("uncertainty_std"),
                inference_time_ms=data.get("inference_time_ms"),
                model_version=data.get("model_version"),
                model_status=data.get("model_status"),
                shap_explanation=data.get("shap_explanation"),
                reversal_score=data.get("reversal_score"),
            )
            s.add(row)
            await s.commit()
        return entry

    async def get_predictions(self, patient_id: str, limit: int = 20,
                               source: Optional[str] = None) -> List[dict]:
        if not self.ready:
            pool = [p for p in self._mem["predictions"] if p["patient_id"] == patient_id]
            if source:
                pool = [p for p in pool if p.get("source", "simulate") == source]
            return pool[-limit:]
        async with self.Session() as s:
            q = select(PredictionHistoryRow).where(PredictionHistoryRow.patient_id == patient_id)
            if source:
                q = q.where(PredictionHistoryRow.source == source)
            q = q.order_by(PredictionHistoryRow.timestamp.desc()).limit(limit)
            rows = (await s.execute(q)).scalars().all()
            return [self._row_to_dict(r) for r in reversed(rows)]

    async def count_predictions(self, patient_id: Optional[str] = None) -> int:
        if not self.ready:
            pool = self._mem["predictions"]
            return len([p for p in pool if patient_id is None or p["patient_id"] == patient_id])
        async with self.Session() as s:
            q = select(func.count()).select_from(PredictionHistoryRow)
            if patient_id:
                q = q.where(PredictionHistoryRow.patient_id == patient_id)
            return (await s.execute(q)).scalar_one()

    # ---- REVERSAL SCORE HISTORY (Phase 2) ------------------------------------
    async def add_reversal_snapshot(self, patient_id: str, data: dict) -> dict:
        entry = {"id": str(uuid.uuid4()), "timestamp": datetime.utcnow().isoformat(), **data}
        if not self.ready:
            self._mem.setdefault("reversal_history", []).append({**entry, "patient_id": patient_id})
            return entry
        async with self.Session() as s:
            row = ReversalScoreHistoryRow(
                id=entry["id"], patient_id=patient_id,
                reversal_score=data.get("reversal_score"),
                lifestyle_score=data.get("lifestyle_score"),
                recovery_index=data.get("recovery_index"),
                insulin_sensitivity_score=data.get("insulin_sensitivity_score"),
                metabolic_health_score=data.get("metabolic_health_score"),
                recovery_stage=data.get("recovery_stage"),
                component_scores=data.get("component_scores"),
            )
            s.add(row)
            await s.commit()
        return entry

    async def get_reversal_history(self, patient_id: str, since: Optional[datetime] = None,
                                    limit: int = 100) -> List[dict]:
        if not self.ready:
            pool = [r for r in self._mem.get("reversal_history", []) if r["patient_id"] == patient_id]
            if since:
                pool = [r for r in pool if r["timestamp"] >= since.isoformat()]
            return pool[-limit:]
        async with self.Session() as s:
            q = select(ReversalScoreHistoryRow).where(ReversalScoreHistoryRow.patient_id == patient_id)
            if since:
                q = q.where(ReversalScoreHistoryRow.timestamp >= since)
            q = q.order_by(ReversalScoreHistoryRow.timestamp.desc()).limit(limit)
            rows = (await s.execute(q)).scalars().all()
            return [self._row_to_dict(r) for r in reversed(rows)]

    # ---- RECOMMENDATIONS (Phase 2) --------------------------------------------
    async def save_recommendations(self, patient_id: str, recommendations: list, basis: dict) -> dict:
        entry = {"id": str(uuid.uuid4()), "timestamp": datetime.utcnow().isoformat(),
                  "recommendations": recommendations, "basis": basis}
        if not self.ready:
            self._mem.setdefault("recommendations", []).append({**entry, "patient_id": patient_id})
            return entry
        async with self.Session() as s:
            row = RecommendationHistoryRow(id=entry["id"], patient_id=patient_id,
                                            recommendations=recommendations, basis=basis)
            s.add(row)
            await s.commit()
        return entry

    async def get_recommendation_history(self, patient_id: str, limit: int = 10) -> List[dict]:
        if not self.ready:
            return [r for r in self._mem.get("recommendations", []) if r["patient_id"] == patient_id][-limit:]
        async with self.Session() as s:
            q = (select(RecommendationHistoryRow).where(RecommendationHistoryRow.patient_id == patient_id)
                 .order_by(RecommendationHistoryRow.timestamp.desc()).limit(limit))
            rows = (await s.execute(q)).scalars().all()
            return [self._row_to_dict(r) for r in reversed(rows)]

    # ---- EVALUATION METRICS (Phase 2) -----------------------------------------
    async def save_evaluation_metrics(self, patient_id: str, metrics: dict) -> dict:
        entry = {"id": str(uuid.uuid4()), "timestamp": datetime.utcnow().isoformat(), **metrics}
        if not self.ready:
            self._mem.setdefault("eval_metrics", []).append({**entry, "patient_id": patient_id})
            return entry
        async with self.Session() as s:
            row = EvaluationMetricsRow(
                id=entry["id"], patient_id=patient_id,
                mae=metrics.get("MAE"), rmse=metrics.get("RMSE"),
                mard=metrics.get("MARD%") or metrics.get("MARD"),
                r2=metrics.get("R2"), n_samples=metrics.get("n_samples"),
            )
            s.add(row)
            await s.commit()
        return entry

    async def get_latest_evaluation_metrics(self, patient_id: str) -> Optional[dict]:
        if not self.ready:
            pool = [m for m in self._mem.get("eval_metrics", []) if m["patient_id"] == patient_id]
            return pool[-1] if pool else None
        async with self.Session() as s:
            q = (select(EvaluationMetricsRow).where(EvaluationMetricsRow.patient_id == patient_id)
                 .order_by(EvaluationMetricsRow.timestamp.desc()).limit(1))
            row = (await s.execute(q)).scalar_one_or_none()
            return self._row_to_dict(row) if row else None

    # ---- SETTINGS (schema present for future use; no route wired yet) ------------

    async def get_settings(self, patient_id: str) -> dict:
        if not self.ready:
            return self._mem["settings"].get(patient_id, {})
        async with self.Session() as s:
            row = (await s.execute(select(SettingsRow).where(SettingsRow.patient_id == patient_id))).scalar_one_or_none()
            return (row.preferences or {}) if row else {}

    async def save_settings(self, patient_id: str, preferences: dict) -> dict:
        if not self.ready:
            self._mem["settings"][patient_id] = preferences
            return preferences
        async with self.Session() as s:
            row = (await s.execute(select(SettingsRow).where(SettingsRow.patient_id == patient_id))).scalar_one_or_none()
            if not row:
                row = SettingsRow(patient_id=patient_id)
                s.add(row)
            row.preferences = preferences
            row.updated_at = datetime.utcnow()
            await s.commit()
            return preferences


# Singleton — imported as `from database import db` by backend_api.py
db = DatabaseManager()
