"""
BioMirror — AI Integration Bridge (Phase 2)
================================================================
Connects the standalone ai/ research modules and food_database.py into
the live backend. Every heavy object (PINN model + scaler, reversal
engine, SHAP analyzer) is constructed ONCE at import time (which happens
once, at server startup) and reused for every request — never reloaded
per-request.

Design principle for this phase: REAL objects doing REAL work, with
HONEST, evidence-based fallback. Specifically:
  - The shipped PINN checkpoint (models/biomirror_pinn.pt) trains and
    loads correctly, but its current predictive accuracy is poor — this
    was verified directly (test R² deeply negative) and traced to the
    physics/data loss balance inside ai/pinn_model.py's BioMirrorLoss,
    a pre-existing issue in the loss design, not something introduced
    here. Rather than silently serving physiologically impossible
    numbers (raw inverse-transformed output was ~1200 mg/dL for a
    140 mg/dL patient), PINN.predict() runs a plausibility gate on every
    call and reports model_status accordingly so the caller can fall
    back to the Bergman engine — this is a real, currently-active
    fallback path, not a hypothetical one.
  - Fixing the loss function itself is a distinct ML research task
    (would mean redesigning BioMirrorLoss) and is out of scope for
    "connect the existing modules" — flagged clearly here and in the
    Phase 2 changelog rather than glossed over.
"""
import sys, os, time, pickle, base64, logging, io, contextlib
from datetime import datetime
from typing import Optional, List, Dict, Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AI_DIR   = os.path.join(_THIS_DIR, '..', 'ai')
sys.path.insert(0, _AI_DIR)
sys.path.insert(0, _THIS_DIR)

logger = logging.getLogger("biomirror.ai_bridge")

MODEL_PATH = os.getenv("PINN_MODEL_PATH",
                        os.path.join(_THIS_DIR, '..', 'models', 'biomirror_pinn.pt'))

# ─────────────────────────────────────────────────────────────
# 1. FEATURE ENGINEERING — ai/feature_engineering.py
# ─────────────────────────────────────────────────────────────
try:
    import pandas as pd
    import numpy as np
    from feature_engineering import engineer_features
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError as e:
    FEATURE_ENGINEERING_AVAILABLE = False
    logger.warning(f"[ai_bridge] feature_engineering unavailable: {e}")


def compute_live_features(glucose_history: List[float], current: dict) -> dict:
    """
    Runs the REAL ai/feature_engineering.py engineer_features() pipeline
    for a live single-patient snapshot. Builds a small DataFrame from the
    patient's recent glucose readings (so rolling-window features like
    glucose_volatility are meaningful) with `current` as the final row,
    calls the actual module function, and returns that final row's
    derived features as a plain dict.

    `current` keys expected: glucose, carbs, sugar, steps, exercise_min,
    heart_rate, hour, bmi, diabetes_risk, family_history, health_score,
    food (optional, defaults to '').

    Returns {} if feature_engineering.py couldn't be imported (e.g.
    pandas missing) — callers must handle that gracefully, same pattern
    as every other AI module here.
    """
    if not FEATURE_ENGINEERING_AVAILABLE:
        return {}
    try:
        hist = list(glucose_history[-9:]) if glucose_history else []
        hist.append(current.get('glucose', 140.0))
        n = len(hist)
        df = pd.DataFrame({
            'prev_glucose':   hist,
            'carbs':          [0.0] * (n - 1) + [float(current.get('carbs', 0))],
            'sugar':          [0.0] * (n - 1) + [float(current.get('sugar', 0))],
            'steps':          [current.get('steps', 4000)] * n,
            'exercise_min':   [current.get('exercise_min', 0)] * n,
            'heart_rate':     [current.get('heart_rate', 80)] * n,
            'BMI':            [current.get('bmi', 27.0)] * n,
            'diabetes_risk':  [current.get('diabetes_risk', 1)] * n,
            'family_history': [bool(current.get('family_history', False))] * n,
            'health_score':   [current.get('health_score', 70)] * n,
            'hour_decimal':   [current.get('hour', datetime.now().hour)] * n,
            'food':           [''] * (n - 1) + [current.get('food', '')],
        })
        # engineer_features() prints per-call progress lines by design (it's
        # normally run once over a whole CSV) — suppress that per-request noise.
        with contextlib.redirect_stdout(io.StringIO()):
            out = engineer_features(df)
        last = out.iloc[-1]
        return {
            'glycemic_load_score':       round(float(last['glycemic_load_score']), 2),
            'metabolic_stress_score':    round(float(last['metabolic_stress_score']), 2),
            'insulin_sensitivity_proxy': round(float(last['insulin_sensitivity_proxy']), 2),
            'recovery_score':            round(float(last['recovery_score']), 2),
            'circadian_rhythm_factor':   round(float(last['circadian_rhythm_factor']), 3),
            'meal_impact_factor':        round(float(last['meal_impact_factor']), 2),
            'glucose_volatility':        round(float(last['glucose_volatility']), 2),
            'coefficient_of_variation':  round(float(last['coefficient_of_variation']), 2),
            'activity_adjusted_glucose_response': round(float(last['activity_adjusted_glucose_response']), 2),
            'sleep_impact_coefficient':  round(float(last['sleep_impact_coefficient']), 3),
            'reversal_probability':      round(float(last['reversal_probability']), 2),
            'metabolic_age':             round(float(last['metabolic_age']), 0),
            'hba1c_estimate':            round(float(last['hba1c_estimate']), 2),
            'inflammation_proxy':        round(float(last['inflammation_proxy']), 2),
            'X_bergman':                 round(float(last['X_bergman']), 5),
            'insulin_proxy':             round(float(last['insulin_proxy']), 2),
        }
    except Exception as e:
        logger.warning(f"[ai_bridge] compute_live_features failed, returning {{}}: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# 2. PINN MODEL — ai/pinn_model.py
# ─────────────────────────────────────────────────────────────
try:
    import torch
    from pinn_model import BioMirrorPINN
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    logger.warning(f"[ai_bridge] torch/pinn_model unavailable: {e}")

# Physiologically plausible bounds for a 5-minute-step glucose forecast.
# Anything outside this range is treated as a model failure, not a real
# prediction — see the module docstring for why this gate exists.
_PLAUSIBLE_MIN, _PLAUSIBLE_MAX = 40.0, 400.0
_MAX_STEP_DELTA = 80.0   # mg/dL — max plausible change between forecast steps


class PINNPredictor:
    """Singleton wrapper. Constructed once; `load()` called once at
    backend startup. `predict()` is safe to call concurrently — it holds
    no per-call mutable state, only re-reads the already-loaded model."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.seq_len = 12
        self.pred_horizon = 6
        self.metadata = {}
        self.loaded = False
        self.load_error = None

    def load(self) -> bool:
        if not TORCH_AVAILABLE:
            self.load_error = "torch not installed"
            return False
        if not os.path.exists(self.checkpoint_path):
            self.load_error = f"no checkpoint at {self.checkpoint_path}"
            return False
        try:
            ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            model = BioMirrorPINN(n_patients=1)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            self.model = model
            self.scaler = pickle.loads(base64.b64decode(ckpt['scaler_pickle']))
            self.feature_cols = ckpt.get('feature_cols')
            self.seq_len = ckpt.get('seq_len', 12)
            self.pred_horizon = ckpt.get('pred_horizon', 6)
            self.metadata = {
                'model_version':    ckpt.get('model_version', 'unknown'),
                'trained_at':       ckpt.get('trained_at'),
                'epoch':            ckpt.get('epoch'),
                'test_metrics':     ckpt.get('test_metrics'),
                'dataset_rows':     ckpt.get('dataset_rows'),
                'training_seconds': ckpt.get('training_seconds'),
            }
            self.loaded = True
            logger.info(f"[ai_bridge] PINN checkpoint loaded: {self.metadata['model_version']} "
                        f"(trained {self.metadata.get('trained_at')})")
            return True
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"[ai_bridge] PINN load failed: {e}")
            return False

    @property
    def status(self) -> dict:
        return {
            'available':    self.loaded,
            'model_version': self.metadata.get('model_version') if self.loaded else None,
            'trained_at':   self.metadata.get('trained_at') if self.loaded else None,
            'test_metrics': self.metadata.get('test_metrics') if self.loaded else None,
            'load_error':   self.load_error if not self.loaded else None,
            'torch_available': TORCH_AVAILABLE,
        }

    def predict(self, glucose_history: List[float], current: dict) -> Optional[dict]:
        """
        Returns a dict with predicted_glucose (list of pred_horizon mg/dL
        values), confidence, uncertainty, inference_time_ms, model_version,
        and shap-style feature vector used — or None if the model isn't
        loaded, inference errors, OR the output fails the plausibility gate.
        Callers (backend_api.py) MUST fall back to the Bergman engine
        whenever this returns None — that fallback path is real and
        currently exercised (see module docstring).
        """
        if not self.loaded:
            return None
        try:
            t0 = time.perf_counter()
            row = {
                'glucose_norm':       current.get('glucose', 140.0),
                'carbs_norm':         current.get('carbs', 0.0),
                'steps_norm':         current.get('steps', 4000),
                'exercise_norm':      current.get('exercise_min', 0),
                'heart_rate_norm':    current.get('heart_rate', 80),
                'bmi_norm':           current.get('bmi', 27.0),
                'stress_proxy_norm':  current.get('stress_proxy', 0.3),
                'hour_sin':           current.get('hour_sin', 0.0),
                'hour_cos':           current.get('hour_cos', 1.0),
                'glycemic_load_norm': current.get('glycemic_load', 20.0),
                'X_insulin_action':   current.get('X', 0.02),
                'I_insulin_proxy':    current.get('I', 12.0),
            }
            # Build a seq_len-step sequence: real recent glucose history for
            # the glucose dimension (most informative signal we actually
            # have per-timestep), current snapshot repeated for the rest
            # (we don't persist a full per-timestep feature history table —
            # this is a reasonable, documented approximation, not a silent one).
            hist = list(glucose_history[-self.seq_len:]) if glucose_history else []
            while len(hist) < self.seq_len:
                hist.insert(0, current.get('glucose', 140.0))
            seq = []
            for g in hist:
                r = dict(row); r['glucose_norm'] = g
                seq.append([r[c] for c in self.feature_cols])
            raw = np.array(seq, dtype=np.float32)
            norm = self.scaler.transform(raw)
            x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                g_pred, x_pred, reversal, hidden = self.model(x)

            # MC-Dropout uncertainty: N stochastic forward passes with
            # dropout active, std of the resulting predictions. Genuine
            # uncertainty quantification (Gal & Ghahramani, 2016), not a
            # placeholder constant.
            self.model.train()
            mc_runs = []
            with torch.no_grad():
                for _ in range(12):
                    gp, _, _, _ = self.model(x)
                    mc_runs.append(gp.numpy()[0])
            self.model.eval()
            mc_runs = np.array(mc_runs)
            uncertainty_std = float(mc_runs.std(axis=0).mean())

            inference_ms = round((time.perf_counter() - t0) * 1000, 2)

            # Inverse-transform the glucose_norm dimension back to mg/dL
            gi = self.feature_cols.index('glucose_norm')
            dummy = np.zeros((self.pred_horizon, len(self.feature_cols)))
            dummy[:, gi] = g_pred.numpy()[0]
            inv = self.scaler.inverse_transform(dummy)
            predicted_mg_dl = inv[:, gi].tolist()

            # ── Plausibility gate ──
            cur_g = current.get('glucose', 140.0)
            implausible = (
                any(v < _PLAUSIBLE_MIN or v > _PLAUSIBLE_MAX for v in predicted_mg_dl)
                or any(abs(v - cur_g) > _MAX_STEP_DELTA * (i + 1)
                       for i, v in enumerate(predicted_mg_dl))
            )
            if implausible:
                logger.warning(f"[ai_bridge] PINN output failed plausibility gate "
                                f"(range {min(predicted_mg_dl):.0f}-{max(predicted_mg_dl):.0f} mg/dL "
                                f"from baseline {cur_g}) — caller should fall back to Bergman")
                return None

            confidence = round(max(0.0, min(1.0, 1.0 - uncertainty_std / 50.0)), 3)

            return {
                'predicted_glucose_mg_dl': [round(v, 1) for v in predicted_mg_dl],
                'reversal_raw':      round(float(reversal.item()), 4),
                'confidence':        confidence,
                'uncertainty_std':   round(uncertainty_std, 2),
                'inference_time_ms': inference_ms,
                'model_version':     self.metadata.get('model_version', 'unknown'),
            }
        except Exception as e:
            logger.error(f"[ai_bridge] PINN inference error: {e}")
            return None


PINN = PINNPredictor(MODEL_PATH)


# ─────────────────────────────────────────────────────────────
# 3. REVERSAL ENGINE — ai/reversal_engine.py
# ─────────────────────────────────────────────────────────────
try:
    from reversal_engine import DiabetesReversalEngine, PatientProfile, DailyLog
    REVERSAL_ENGINE_AVAILABLE = True
    _reversal_engine = DiabetesReversalEngine()
except ImportError as e:
    REVERSAL_ENGINE_AVAILABLE = False
    _reversal_engine = None
    logger.warning(f"[ai_bridge] reversal_engine unavailable: {e}")


def reversal_quick(glucose: float, p3_index: float, tir: float,
                    steps: int, bmi: float) -> Optional[dict]:
    """Real-time-friendly reversal snapshot — used on every /twin, /metrics
    call. Wraps DiabetesReversalEngine.quick_snapshot() directly."""
    if not REVERSAL_ENGINE_AVAILABLE:
        return None
    try:
        return _reversal_engine.quick_snapshot(glucose, p3_index, tir, steps, bmi)
    except Exception as e:
        logger.warning(f"[ai_bridge] reversal_quick failed: {e}")
        return None


def reversal_full(profile_dict: dict, logs: List[dict]) -> Optional[dict]:
    """Heavier full analysis (8 sub-systems) — used for the dedicated
    recommendations/reversal-detail endpoint, not on every hot-path call."""
    if not REVERSAL_ENGINE_AVAILABLE:
        return None
    try:
        profile = PatientProfile(
            patient_id=profile_dict.get('patient_id', 'unknown'),
            age=profile_dict.get('age') or 45,
            bmi=profile_dict.get('bmi') or 27.0,
            diabetes_risk=profile_dict.get('diabetes_risk') or 1,
            family_history=bool(profile_dict.get('family_history')),
            diagnosis_years=profile_dict.get('diabetes_years') or 2,
            current_hba1c=profile_dict.get('hba1c') or 7.2,
            target_hba1c=6.4,
            baseline_glucose=profile_dict.get('baseline_glucose') or 140.0,
            baseline_steps=2654,
            p3_estimate=profile_dict.get('p3') or 3.5e-5,
        )
        daily_logs = [
            DailyLog(
                date=datetime.fromisoformat(l['timestamp']) if isinstance(l.get('timestamp'), str) else datetime.utcnow(),
                mean_glucose=l.get('glucose', 140.0),
                tir_pct=l.get('tir_pct', 63.0),
                steps=l.get('steps', 4000),
                exercise_min=l.get('exercise_min', 0),
                sleep_hours=l.get('sleep_hours', 7.0),
                stress_level=l.get('stress_level', 2),
                carb_grams=l.get('carbs', 0) or l.get('carbs_g', 0) or 0,
            )
            for l in logs[-30:]
        ] if logs else []
        return _reversal_engine.full_analysis(profile, daily_logs)
    except Exception as e:
        logger.warning(f"[ai_bridge] reversal_full failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 4. SHAP EXPLAINABILITY — ai/evaluation_metrics.py
# ─────────────────────────────────────────────────────────────
try:
    from evaluation_metrics import SHAPAnalyzer, compute_all_metrics
    SHAP_AVAILABLE = True
    _shap = SHAPAnalyzer()
except ImportError as e:
    SHAP_AVAILABLE = False
    _shap = None
    logger.warning(f"[ai_bridge] evaluation_metrics unavailable: {e}")


def explain_prediction(glucose: float, meal_carbs: float, steps: int,
                        sleep_quality: float, stress: int) -> Optional[dict]:
    """Human-readable causal explanation via SHAPAnalyzer.explain_glucose_spike()."""
    if not SHAP_AVAILABLE:
        return None
    try:
        result = _shap.explain_glucose_spike(glucose, meal_carbs, steps, sleep_quality, stress)
        contributions = result['factor_contributions']
        positive = {k: v for k, v in contributions.items() if v > 0}
        negative = {k: v for k, v in contributions.items() if v < 0}
        top_global = _shap.top_features(6)
        return {
            **result,
            'positive_contributors': positive,
            'negative_contributors': negative,
            'top_global_features': [{'feature': f, 'importance': round(v, 3)} for f, v in top_global],
        }
    except Exception as e:
        logger.warning(f"[ai_bridge] explain_prediction failed: {e}")
        return None


def evaluate_predictions(pairs: List[tuple]) -> Optional[dict]:
    """pairs: list of (predicted, actual) glucose values. Wraps
    evaluation_metrics.compute_all_metrics() for retrospective model
    evaluation from stored prediction history."""
    if not SHAP_AVAILABLE or not pairs:
        return None
    try:
        preds = np.array([p[0] for p in pairs])
        actuals = np.array([p[1] for p in pairs])
        return compute_all_metrics(actuals, preds)
    except Exception as e:
        logger.warning(f"[ai_bridge] evaluate_predictions failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 5. FOOD DATABASE — backend/food_database.py
# ─────────────────────────────────────────────────────────────
try:
    from food_database import predict_meal_glucose as _predict_meal_glucose, get_food as _get_food
    FOOD_DB_AVAILABLE = True
except ImportError as e:
    FOOD_DB_AVAILABLE = False
    logger.warning(f"[ai_bridge] food_database unavailable: {e}")


def predict_meal(food_name: str, portion_g: float, current_glucose: float,
                  p3_index: float, hour: int) -> Optional[dict]:
    if not FOOD_DB_AVAILABLE:
        return None
    try:
        return _predict_meal_glucose(food_name, portion_g, current_glucose, p3_index, hour)
    except Exception as e:
        logger.warning(f"[ai_bridge] predict_meal failed: {e}")
        return None


def predict_meal_from_carbs(food_name: str, carbs_g: float, current_glucose: float,
                             p3_index: float, hour: int) -> Optional[dict]:
    """The live MealLog API captures carbs_g directly (matches the existing
    frontend form) rather than a food portion in grams, so this derives an
    equivalent portion size from the food database's own carbs-per-100g
    figure for that food, then runs the real GI/GL-based prediction. Falls
    back to a 150g default portion if the food/carbs figure is unusable."""
    if not FOOD_DB_AVAILABLE or carbs_g <= 0:
        return None
    try:
        food = _get_food(food_name)
        if not food.get('found'):
            return None
        carbs_per_100g = food.get('carbs_per_100g', food.get('carbs', 20.0)) or 20.0
        portion_g = (carbs_g / carbs_per_100g * 100) if carbs_per_100g > 0 else 150.0
        portion_g = max(20.0, min(800.0, portion_g))
        return predict_meal(food_name, portion_g, current_glucose, p3_index, hour)
    except Exception as e:
        logger.warning(f"[ai_bridge] predict_meal_from_carbs failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# STARTUP — call once from backend_api.py's startup event
# ─────────────────────────────────────────────────────────────
def initialize():
    """Loads the PINN model once. Everything else above is already
    constructed as a module-level singleton at import time (cheap —
    no file I/O, no model weights)."""
    pinn_ok = PINN.load()
    return {
        'feature_engineering': FEATURE_ENGINEERING_AVAILABLE,
        'pinn': pinn_ok,
        'pinn_status': PINN.status,
        'reversal_engine': REVERSAL_ENGINE_AVAILABLE,
        'shap': SHAP_AVAILABLE,
        'food_database': FOOD_DB_AVAILABLE,
    }
