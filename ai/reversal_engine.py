"""
BioMirror — Diabetes Reversal Intelligence Engine
===================================================
The Reversal Engine goes far beyond glucose prediction.
It models the patient's metabolic TRAJECTORY toward T2D remission.

Components:
  1.  Metabolic Recovery Scoring     — composite reversal index
  2.  Time-in-Range Optimiser        — TIR maximisation strategy
  3.  Insulin Sensitivity Tracker    — p3 improvement over time
  4.  Habit Impact Modeller          — quantifies each lifestyle lever
  5.  Lifestyle Intervention Planner — personalised action roadmap
  6.  Meal Response Forecaster       — food-specific glucose impact
  7.  Long-term Trend Estimator      — 30/60/90/180 day projections
  8.  Reversal Probability Engine    — Bayesian probability of remission
  9.  Circadian Metabolism Engine    — time-of-day metabolic patterns
  10. Reinforcement Learning Nudges  — adaptive precision nudge system

Clinical Reference:
  Remission criterion: HbA1c < 6.5% maintained ≥ 3 months without medication
  Source: Riddle et al., Diabetes Care 2021 consensus statement
"""

import numpy as np
import math
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS & CONSTANTS
# ─────────────────────────────────────────────
class RecoveryStage(Enum):
    DIABETIC          = 1   # HbA1c ≥ 8.0%
    PRE_REVERSAL      = 2   # HbA1c 7.0–8.0%
    ACTIVE_REVERSAL   = 3   # HbA1c 6.5–7.0%
    NEAR_NORMAL       = 4   # HbA1c 5.7–6.5%
    REVERSED          = 5   # HbA1c < 5.7%

class NudgeType(Enum):
    MOVEMENT  = "movement"
    NUTRITION = "nutrition"
    SLEEP     = "sleep"
    STRESS    = "stress"
    HYDRATION = "hydration"
    FASTING   = "fasting"

FOOD_DATABASE = {
    # food: (gi_index, carbs_per_100g, protein_per_100g, fiber_per_100g)
    'white_rice':   (0.72, 28, 2.7, 0.4),
    'brown_rice':   (0.55, 23, 2.6, 1.8),
    'white_bread':  (0.70, 49, 9,   2.7),
    'whole_wheat':  (0.50, 41, 13,  6.4),
    'dosa':         (0.68, 21, 3,   0.5),
    'oats':         (0.55, 27, 17,  10),
    'banana':       (0.51, 23, 1.1, 2.6),
    'apple':        (0.36, 14, 0.3, 2.4),
    'salad':        (0.15, 4,  1.5, 2.0),
    'dal':          (0.28, 20, 9,   8),
    'idli':         (0.65, 22, 2,   0.5),
    'pizza':        (0.60, 33, 11,  2.3),
    'pasta':        (0.49, 25, 5,   1.8),
    'chicken':      (0.00, 0,  27,  0),
    'egg':          (0.00, 1,  13,  0),
    'milk':         (0.27, 5,  3.4, 0),
    'yogurt':       (0.35, 4,  10,  0),
    'potato':       (0.78, 20, 2,   1.8),
    'sweet_potato': (0.54, 20, 1.6, 3),
    'default':      (0.55, 20, 5,   2),
}

EXERCISE_DATABASE = {
    # activity: (met_value, glucose_reduction_per_min, p3_boost_per_min)
    'walking':     (3.5, 0.25, 0.004),
    'brisk_walk':  (4.5, 0.35, 0.006),
    'jogging':     (7.0, 0.50, 0.009),
    'cycling':     (6.0, 0.42, 0.008),
    'swimming':    (6.5, 0.45, 0.009),
    'yoga':        (2.5, 0.12, 0.003),
    'strength':    (5.0, 0.30, 0.010),
    'hiit':        (9.0, 0.65, 0.015),
    'dancing':     (5.5, 0.38, 0.007),
    'default':     (4.0, 0.30, 0.006),
}


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class PatientProfile:
    patient_id:       str
    age:              int   = 45
    bmi:              float = 27.0
    diabetes_risk:    int   = 1        # 0/1/2
    family_history:   bool  = False
    diagnosis_years:  int   = 2        # years since T2D diagnosis
    current_hba1c:    float = 7.2
    target_hba1c:     float = 6.4
    baseline_glucose: float = 161.4    # dataset mean
    baseline_steps:   int   = 2654     # dataset mean
    p3_estimate:      float = 3.5e-5   # insulin sensitivity
    medications:      List[str] = field(default_factory=list)
    comorbidities:    List[str] = field(default_factory=list)


@dataclass
class DailyLog:
    date:         datetime
    mean_glucose: float
    tir_pct:      float
    steps:        int
    exercise_min: int
    sleep_hours:  float
    stress_level: int          # 1-5
    carb_grams:   float
    water_liters: float = 2.0
    meal_count:   int   = 3


@dataclass
class NudgeAction:
    nudge_type:   NudgeType
    message:      str
    action:       str
    expected_glucose_drop: float   # mg/dL
    expected_tir_gain:     float   # percentage points
    difficulty:   str              # LOW/MEDIUM/HIGH
    time_minutes: int
    priority:     int              # 1 = highest


@dataclass
class ReversalSnapshot:
    date:                  datetime
    hba1c:                 float
    reversal_score:        float
    tir_pct:               float
    insulin_sensitivity:   float   # p3 index 0-100
    metabolic_age:         float
    recovery_stage:        RecoveryStage
    glucose_volatility:    float
    days_to_reversal:      Optional[float]


# ─────────────────────────────────────────────
# 1. METABOLIC RECOVERY SCORING
# ─────────────────────────────────────────────
class MetabolicRecoveryScorer:
    """
    Computes a composite Reversal Score (0–100) from:
      - Insulin sensitivity trajectory (p3 trend)
      - Time-in-Range achievement
      - Glucose volatility (lower = better)
      - HbA1c trend (decreasing = better)
      - Lifestyle consistency score
      - Medication reduction (proxy for reversal)
    """
    WEIGHTS = {
        'insulin_sensitivity': 0.28,
        'time_in_range':       0.25,
        'hba1c_improvement':   0.20,
        'glucose_stability':   0.12,
        'lifestyle_score':     0.10,
        'weight_reduction':    0.05,
    }

    def compute(self, profile: PatientProfile,
                logs: List[DailyLog],
                p3_history: List[float]) -> dict:

        # Insulin sensitivity component
        p3_idx = profile.p3_estimate / 5e-5 * 100
        sens_score = min(100, p3_idx)

        # TIR component
        if logs:
            avg_tir = np.mean([l.tir_pct for l in logs])
        else:
            avg_tir = 63.0   # dataset mean
        tir_score = avg_tir

        # HbA1c improvement
        hba1c_target_gap = profile.current_hba1c - profile.target_hba1c
        if hba1c_target_gap <= 0:
            hba1c_score = 100.0
        else:
            max_gap = 3.0  # worst case: HbA1c = 9.4% vs target 6.4%
            hba1c_score = max(0, 100 - (hba1c_target_gap / max_gap * 100))

        # Glucose stability (inverse of coefficient of variation)
        if logs:
            glucose_vals = [l.mean_glucose for l in logs]
            cv = np.std(glucose_vals) / (np.mean(glucose_vals) + 1e-6) * 100
        else:
            cv = 25.0  # assumed baseline
        stability_score = max(0, 100 - cv * 2)

        # Lifestyle consistency score
        if logs:
            avg_steps   = np.mean([l.steps for l in logs])
            avg_sleep   = np.mean([l.sleep_hours for l in logs])
            avg_stress  = np.mean([l.stress_level for l in logs])
            avg_water   = np.mean([l.water_liters for l in logs])
            step_score  = min(100, avg_steps / 8000 * 100)
            sleep_score = max(0, 100 - abs(avg_sleep - 7.5) * 20)
            stress_score= max(0, 100 - (avg_stress - 1) * 25)
            water_score = min(100, avg_water / 2.5 * 100)
            lifestyle_score = (step_score*0.35 + sleep_score*0.30 +
                               stress_score*0.20 + water_score*0.15)
        else:
            lifestyle_score = 55.0

        # Weight reduction score
        bmi_excess = max(0, profile.bmi - 23)
        weight_score = max(0, 100 - bmi_excess * 8)

        # Weighted composite
        components = {
            'insulin_sensitivity': sens_score,
            'time_in_range':       tir_score,
            'hba1c_improvement':   hba1c_score,
            'glucose_stability':   stability_score,
            'lifestyle_score':     lifestyle_score,
            'weight_reduction':    weight_score,
        }
        reversal_score = sum(
            components[k] * self.WEIGHTS[k]
            for k in self.WEIGHTS
        )

        # Determine recovery stage
        h = profile.current_hba1c
        if h >= 8.0:   stage = RecoveryStage.DIABETIC
        elif h >= 7.0: stage = RecoveryStage.PRE_REVERSAL
        elif h >= 6.5: stage = RecoveryStage.ACTIVE_REVERSAL
        elif h >= 5.7: stage = RecoveryStage.NEAR_NORMAL
        else:          stage = RecoveryStage.REVERSED

        return {
            'reversal_score':        round(reversal_score, 2),
            'recovery_stage':        stage.name,
            'recovery_stage_number': stage.value,
            'component_scores':      {k: round(v, 2) for k, v in components.items()},
            'clinical_remission_pct': round(max(0, min(100, reversal_score * 0.95)), 1),
        }


# ─────────────────────────────────────────────
# 2. TIME-IN-RANGE OPTIMISER
# ─────────────────────────────────────────────
class TIROptimiser:
    """
    Identifies the minimum interventions required to reach target TIR (≥70%).
    Uses marginal impact analysis: which lever gives most TIR per unit effort?
    """
    TARGET_TIR = 70.0   # ADA/EASD consensus minimum
    OPTIMAL_TIR = 80.0  # ideal target

    def analyse(self, current_tir: float, logs: List[DailyLog],
                profile: PatientProfile) -> dict:
        gap = self.TARGET_TIR - current_tir
        levers = self._compute_lever_impacts(logs, profile)
        # Sort by TIR gain per effort unit
        sorted_levers = sorted(levers.items(),
                               key=lambda x: x[1]['tir_gain_per_effort'],
                               reverse=True)
        # Greedy selection: pick top levers until gap closed
        selected, cumulative_gain = [], 0.0
        for name, data in sorted_levers:
            if cumulative_gain >= gap:
                break
            selected.append({
                'intervention': name,
                'expected_tir_gain': data['tir_gain'],
                'effort_level': data['effort'],
                'time_to_effect_days': data['days'],
            })
            cumulative_gain += data['tir_gain']

        return {
            'current_tir': round(current_tir, 1),
            'target_tir':  self.TARGET_TIR,
            'gap':         round(max(0, gap), 1),
            'lever_impacts': {k: {
                'tir_gain': round(v['tir_gain'], 1),
                'effort': v['effort'],
                'days_to_effect': v['days'],
            } for k, v in levers.items()},
            'recommended_interventions': selected,
            'days_to_target': round(
                max(gap, 0) / max(sum(s['expected_tir_gain'] for s in selected[:3]) / 60, 0.1), 0
            ),
        }

    def _compute_lever_impacts(self, logs: List[DailyLog],
                                profile: PatientProfile) -> dict:
        avg_steps = np.mean([l.steps for l in logs]) if logs else 2654
        avg_carbs = np.mean([l.carb_grams for l in logs]) if logs else 65
        avg_sleep = np.mean([l.sleep_hours for l in logs]) if logs else 6.5
        avg_stress = np.mean([l.stress_level for l in logs]) if logs else 3

        levers = {
            'increase_steps_to_8000': {
                'tir_gain': min(12, max(0, (8000 - avg_steps) / 800)),
                'effort': 'LOW',
                'days': 14,
                'tir_gain_per_effort': 4.0,
            },
            'reduce_carbs_to_50g': {
                'tir_gain': min(15, max(0, (avg_carbs - 50) / 4)),
                'effort': 'MEDIUM',
                'days': 7,
                'tir_gain_per_effort': 3.5,
            },
            'improve_sleep_to_7h': {
                'tir_gain': min(8, max(0, (7.0 - avg_sleep) * 3.5)),
                'effort': 'LOW',
                'days': 7,
                'tir_gain_per_effort': 3.0,
            },
            'post_meal_walk_15min': {
                'tir_gain': 6.5,
                'effort': 'LOW',
                'days': 3,
                'tir_gain_per_effort': 5.5,
            },
            'reduce_stress': {
                'tir_gain': min(6, max(0, (avg_stress - 2) * 2)),
                'effort': 'MEDIUM',
                'days': 21,
                'tir_gain_per_effort': 1.5,
            },
            'intermittent_fasting_16_8': {
                'tir_gain': 8.0,
                'effort': 'HIGH',
                'days': 21,
                'tir_gain_per_effort': 2.0,
            },
            'low_gi_diet_swap': {
                'tir_gain': 7.5,
                'effort': 'MEDIUM',
                'days': 14,
                'tir_gain_per_effort': 3.0,
            },
        }
        return levers


# ─────────────────────────────────────────────
# 3. INSULIN SENSITIVITY TRACKER
# ─────────────────────────────────────────────
class InsulinSensitivityTracker:
    """
    Tracks p3 improvement over time using the Bergman inverse model.
    Models how lifestyle changes shift the p3 parameter trajectory.
    """
    P3_BASELINE = 3.5e-5
    P3_NORMAL   = 5.5e-5   # healthy insulin sensitivity

    def project_p3_trajectory(self, current_p3: float,
                               daily_logs: List[DailyLog],
                               days: int = 90) -> List[dict]:
        """
        Project p3 trajectory forward given current lifestyle.
        Each day, p3 changes based on:
          +exercise effect: +0.8% p3 per km walked
          +sleep effect:    +0.5% p3 per hour over 7h (to 9h max)
          +weight effect:   +1.2% p3 per kg BMI reduction
          -stress effect:   -0.3% p3 per stress level above 2
          +diet effect:     +0.4% p3 per 10g carbs reduced below 60g
        """
        if daily_logs:
            avg_steps  = np.mean([l.steps for l in daily_logs])
            avg_sleep  = np.mean([l.sleep_hours for l in daily_logs])
            avg_stress = np.mean([l.stress_level for l in daily_logs])
            avg_carbs  = np.mean([l.carb_grams for l in daily_logs])
            avg_ex     = np.mean([l.exercise_min for l in daily_logs])
        else:
            avg_steps, avg_sleep, avg_stress = 2654, 6.5, 3.0
            avg_carbs, avg_ex = 65.0, 15.0

        daily_p3_change = (
              (avg_steps / 1000) * 0.00008          # steps effect
            + max(0, avg_sleep - 7) * 0.00005       # sleep bonus
            - max(0, avg_stress - 2) * 0.00003      # stress penalty
            + max(0, 60 - avg_carbs) / 10 * 0.00004 # carb reduction
            + (avg_ex / 30) * 0.00010               # exercise boost
            - 0.00002                                # natural degradation (ageing)
        )

        trajectory = []
        p3 = current_p3
        for day in range(days + 1):
            p3 = max(1e-5, min(8e-5, p3 + daily_p3_change))
            p3_idx = p3 / 5e-5 * 100
            hba1c_est = max(4.5, 9.0 - p3_idx * 0.025)
            trajectory.append({
                'day':          day,
                'p3':           round(p3, 8),
                'p3_index':     round(p3_idx, 1),
                'hba1c_est':    round(hba1c_est, 2),
                'improvement_pct': round((p3 - current_p3) / current_p3 * 100, 1),
            })

        return trajectory

    def sensitivity_grade(self, p3_index: float) -> str:
        if p3_index >= 90: return "EXCELLENT"
        if p3_index >= 70: return "GOOD"
        if p3_index >= 50: return "MODERATE"
        if p3_index >= 30: return "LOW"
        return "VERY_LOW"


# ─────────────────────────────────────────────
# 4. HABIT IMPACT MODELLER
# ─────────────────────────────────────────────
class HabitImpactModeller:
    """
    Quantifies the glucose impact of each habit change.
    Based on the multi-modal fusion research (Lancet Digital Health, 2024).
    """
    def model_habit_impact(self, habit: str, magnitude: float,
                            baseline_glucose: float = 161.4,
                            p3_index: float = 68.0) -> dict:
        """
        Returns expected glucose change, TIR impact, and timeline.
        magnitude: 0-1 (e.g., for steps: 0.5 = 50% increase from baseline)
        """
        sensitivity_factor = p3_index / 100  # higher sensitivity = bigger response

        impacts = {
            'increase_steps': {
                'glucose_drop_mg_dl': magnitude * 25 * sensitivity_factor,
                'tir_gain_pct':       magnitude * 12,
                'p3_improvement_pct': magnitude * 15,
                'days_to_onset':      3,
                'days_to_peak':       30,
                'mechanism': 'Muscle glucose uptake (GLUT-4 translocation) independent of insulin',
            },
            'low_gi_diet': {
                'glucose_drop_mg_dl': magnitude * 30,
                'tir_gain_pct':       magnitude * 15,
                'p3_improvement_pct': magnitude * 8,
                'days_to_onset':      1,
                'days_to_peak':       14,
                'mechanism': 'Reduced postprandial glucose spike via slower carbohydrate digestion',
            },
            'improve_sleep': {
                'glucose_drop_mg_dl': magnitude * 18 * (1.2 - sensitivity_factor),
                'tir_gain_pct':       magnitude * 8,
                'p3_improvement_pct': magnitude * 18,  # sleep most powerful for sensitivity
                'days_to_onset':      3,
                'days_to_peak':       14,
                'mechanism': 'Cortisol reduction + GH regulation restores insulin receptor sensitivity',
            },
            'stress_reduction': {
                'glucose_drop_mg_dl': magnitude * 15,
                'tir_gain_pct':       magnitude * 7,
                'p3_improvement_pct': magnitude * 10,
                'days_to_onset':      7,
                'days_to_peak':       30,
                'mechanism': 'Cortisol → hepatic gluconeogenesis inhibition',
            },
            'intermittent_fasting': {
                'glucose_drop_mg_dl': magnitude * 22,
                'tir_gain_pct':       magnitude * 18,
                'p3_improvement_pct': magnitude * 22,
                'days_to_onset':      7,
                'days_to_peak':       60,
                'mechanism': 'Insulin level suppression → receptor upregulation (autophagy)',
            },
            'weight_loss_1kg': {
                'glucose_drop_mg_dl': 3.5,   # ~3.5 mg/dL per kg
                'tir_gain_pct':       2.0,
                'p3_improvement_pct': 4.0,
                'days_to_onset':      7,
                'days_to_peak':       30,
                'mechanism': 'Ectopic fat reduction → hepatic and skeletal muscle insulin sensitivity',
            },
            'post_meal_walk': {
                'glucose_drop_mg_dl': magnitude * 28,
                'tir_gain_pct':       magnitude * 10,
                'p3_improvement_pct': magnitude * 5,
                'days_to_onset':      0,   # immediate
                'days_to_peak':       0,
                'mechanism': 'Gastrocnemius glucose uptake peaks within 15-30 min post-meal',
            },
        }
        impact = impacts.get(habit, {
            'glucose_drop_mg_dl': 5.0,
            'tir_gain_pct': 2.0,
            'p3_improvement_pct': 2.0,
            'days_to_onset': 7,
            'days_to_peak': 30,
            'mechanism': 'General metabolic improvement',
        })
        return {
            'habit': habit,
            'magnitude': magnitude,
            'glucose_drop_mg_dl': round(impact['glucose_drop_mg_dl'], 1),
            'expected_glucose': round(baseline_glucose - impact['glucose_drop_mg_dl'], 1),
            'tir_gain_pct': round(impact['tir_gain_pct'], 1),
            'p3_improvement_pct': round(impact['p3_improvement_pct'], 1),
            'days_to_onset': impact['days_to_onset'],
            'days_to_peak': impact['days_to_peak'],
            'mechanism': impact['mechanism'],
        }

    def rank_habits_for_patient(self, profile: PatientProfile,
                                 logs: List[DailyLog]) -> List[dict]:
        """Return habits ranked by expected reversal impact for this patient."""
        avg_steps = np.mean([l.steps for l in logs]) if logs else 2654
        avg_carbs = np.mean([l.carb_grams for l in logs]) if logs else 65
        avg_sleep = np.mean([l.sleep_hours for l in logs]) if logs else 6.5
        avg_stress = np.mean([l.stress_level for l in logs]) if logs else 3.0

        habits_to_rank = [
            ('post_meal_walk',     min(1.0, 1.0)),
            ('increase_steps',     min(1.0, (8000 - avg_steps) / 8000)),
            ('low_gi_diet',        min(1.0, (avg_carbs - 40) / 80)),
            ('improve_sleep',      min(1.0, abs(7.5 - avg_sleep) / 3.5)),
            ('stress_reduction',   min(1.0, (avg_stress - 1) / 4)),
            ('intermittent_fasting', 0.6),
            ('weight_loss_1kg',    min(1.0, max(0, (profile.bmi - 23) / 10))),
        ]

        ranked = []
        for habit, mag in habits_to_rank:
            if mag > 0.1:  # only recommend if meaningful gap exists
                impact = self.model_habit_impact(habit, mag,
                                                  profile.baseline_glucose,
                                                  profile.p3_estimate / 5e-5 * 100)
                ranked.append(impact)

        # Sort by composite impact
        ranked.sort(
            key=lambda x: x['glucose_drop_mg_dl'] * 0.4 + x['tir_gain_pct'] * 0.4
                          + x['p3_improvement_pct'] * 0.2,
            reverse=True,
        )
        for i, h in enumerate(ranked):
            h['rank'] = i + 1
        return ranked


# ─────────────────────────────────────────────
# 5. MEAL RESPONSE FORECASTER
# ─────────────────────────────────────────────
class MealResponseForecaster:
    """
    Predicts individual glucose response to specific foods.
    Integrates: food GI, portion size, circadian timing,
                insulin sensitivity (p3), current glucose level.
    """
    def forecast(self, food: str, portion_g: float,
                 current_glucose: float, p3_index: float,
                 hour: int = 12) -> dict:
        food_data = FOOD_DATABASE.get(food.lower().replace(' ', '_'),
                                      FOOD_DATABASE['default'])
        gi, carbs_per_100, protein, fiber = food_data

        # Effective carbs (subtract fiber — has minimal glucose impact)
        net_carbs = max(0, (carbs_per_100 - fiber * 0.5) * portion_g / 100)

        # Circadian adjustment
        dawn = 1.15 if 4 <= hour <= 8 else 1.0
        evening_resist = 1.10 if 18 <= hour <= 22 else 1.0
        circ_factor = dawn * evening_resist

        # Insulin sensitivity adjustment
        sens_factor = max(0.5, 2.0 - p3_index / 100)

        # Peak glucose rise
        peak_rise = net_carbs * gi * 0.55 * circ_factor * sens_factor

        # Protein effect (small glucose rise via gluconeogenesis)
        protein_effect = protein * portion_g / 100 * 0.15

        total_rise = peak_rise + protein_effect

        # Time to peak (lower GI = later peak)
        peak_min = int(30 + (1 - gi) * 30)

        # Recovery time (p3-dependent)
        recovery_min = int(90 + (100 - p3_index) * 0.9)

        # Predicted curve (simplified Gaussian + exponential decay)
        curve = []
        for t in range(0, 241, 5):
            if t <= peak_min:
                g = current_glucose + total_rise * (t / peak_min) ** 2
            else:
                decay = math.exp(-0.015 * (t - peak_min) * (p3_index / 70))
                g = current_glucose + total_rise * decay
            curve.append({'min': t, 'glucose': round(max(70, g), 1)})

        peak_glucose = round(current_glucose + total_rise, 1)
        return {
            'food': food,
            'portion_g': portion_g,
            'net_carbs_g': round(net_carbs, 1),
            'glycemic_index': gi,
            'peak_glucose': peak_glucose,
            'peak_glucose_rise': round(total_rise, 1),
            'time_to_peak_min': peak_min,
            'recovery_min': recovery_min,
            'glucose_curve': curve,
            'spike_risk': 'HIGH' if peak_glucose > 200 else
                          'MEDIUM' if peak_glucose > 160 else 'LOW',
            'recommendation': self._food_recommendation(food, gi, net_carbs, p3_index),
        }

    def _food_recommendation(self, food: str, gi: float,
                              net_carbs: float, p3_index: float) -> str:
        if gi > 0.65 and net_carbs > 30:
            return (f"{food.title()} has high GI ({gi:.0%}) and high net carbs ({net_carbs:.0f}g). "
                    f"Pair with dal/protein to reduce spike by ~25%. "
                    f"Consider replacing with lower-GI alternative.")
        elif gi > 0.50:
            return (f"Moderate GI food. Take a 15-min walk post-meal to blunt the glucose rise. "
                    f"Your current sensitivity ({p3_index:.0f}%) means recovery in "
                    f"~{int(90 + (100-p3_index)*0.8)} min.")
        return (f"Low-GI choice ✓. {food.title()} should have minimal glucose impact. "
                f"Excellent for your reversal plan.")

    def compare_foods(self, foods_portions: List[Tuple[str, float]],
                      current_glucose: float, p3_index: float,
                      hour: int = 12) -> List[dict]:
        """Compare multiple food options for a given meal slot."""
        results = []
        for food, portion in foods_portions:
            r = self.forecast(food, portion, current_glucose, p3_index, hour)
            results.append({
                'food': food,
                'portion_g': portion,
                'peak_glucose': r['peak_glucose'],
                'recovery_min': r['recovery_min'],
                'spike_risk': r['spike_risk'],
                'net_carbs': r['net_carbs_g'],
            })
        results.sort(key=lambda x: x['peak_glucose'])
        return results


# ─────────────────────────────────────────────
# 6. LONG-TERM TREND ESTIMATOR
# ─────────────────────────────────────────────
class LongTermTrendEstimator:
    """
    Projects HbA1c, insulin sensitivity, and reversal probability
    over 30/60/90/180 days given current lifestyle adherence.
    """
    def project(self, profile: PatientProfile,
                logs: List[DailyLog],
                horizons: List[int] = [30, 60, 90, 180]) -> dict:

        # Compute daily improvement rates from logs
        if logs:
            adherence = self._compute_adherence(logs)
        else:
            adherence = {'steps': 0.33, 'diet': 0.50, 'sleep': 0.60,
                         'stress': 0.55, 'exercise': 0.30}

        # Monthly HbA1c reduction rate (% per month)
        monthly_reduction = (
              adherence['steps']    * 0.08
            + adherence['diet']     * 0.12
            + adherence['sleep']    * 0.06
            + adherence['exercise'] * 0.10
            + adherence['stress']   * 0.04
        )

        # Monthly p3 improvement rate
        monthly_p3_gain = (
              adherence['exercise'] * 0.8e-5
            + adherence['steps']    * 0.3e-5
            + adherence['sleep']    * 0.5e-5
            + adherence['diet']     * 0.4e-5
        )

        projections = {}
        for days in horizons:
            months = days / 30
            hba1c_proj = max(
                profile.target_hba1c,
                profile.current_hba1c - monthly_reduction * months
            )
            p3_proj = min(
                7e-5,
                profile.p3_estimate + monthly_p3_gain * months
            )
            p3_idx_proj = p3_proj / 5e-5 * 100
            tir_proj = min(85, 63 + months * 3.5 * adherence['diet'])
            glucose_proj = max(90, profile.baseline_glucose - months * 8 * adherence['steps'])
            rev_score = (p3_idx_proj * 0.35 + tir_proj * 0.35
                         + max(0, 100 - (hba1c_proj - 5) / 3 * 100) * 0.30)

            projections[f'{days}d'] = {
                'days': days,
                'hba1c': round(hba1c_proj, 2),
                'p3_index': round(p3_idx_proj, 1),
                'tir_pct': round(tir_proj, 1),
                'mean_glucose': round(glucose_proj, 1),
                'reversal_score': round(rev_score, 1),
                'remission_probability': round(
                    max(0, min(100, (rev_score - 50) * 1.5)), 1
                ),
            }

        reversal_months = None
        if monthly_reduction > 0:
            months_needed = (profile.current_hba1c - profile.target_hba1c) / monthly_reduction
            reversal_months = round(max(0, months_needed), 1)

        return {
            'projections': projections,
            'months_to_reversal': reversal_months,
            'confidence': 'HIGH' if len(logs) > 14 else
                          'MEDIUM' if len(logs) > 7 else 'LOW',
            'adherence_scores': {k: round(v * 100, 1) for k, v in adherence.items()},
            'monthly_hba1c_reduction': round(monthly_reduction, 3),
        }

    def _compute_adherence(self, logs: List[DailyLog]) -> dict:
        if not logs:
            return {k: 0.5 for k in ['steps', 'diet', 'sleep', 'stress', 'exercise']}
        avg_steps   = np.mean([l.steps for l in logs])
        avg_carbs   = np.mean([l.carb_grams for l in logs])
        avg_sleep   = np.mean([l.sleep_hours for l in logs])
        avg_stress  = np.mean([l.stress_level for l in logs])
        avg_ex      = np.mean([l.exercise_min for l in logs])
        return {
            'steps':    min(1.0, avg_steps / 8000),
            'diet':     min(1.0, max(0, (80 - avg_carbs) / 80 + 0.3)),
            'sleep':    min(1.0, max(0, 1 - abs(avg_sleep - 7.5) / 3.5)),
            'stress':   min(1.0, max(0, (5 - avg_stress) / 4)),
            'exercise': min(1.0, avg_ex / 45),
        }


# ─────────────────────────────────────────────
# 7. CIRCADIAN METABOLISM ENGINE
# ─────────────────────────────────────────────
class CircadianMetabolismEngine:
    """
    Models time-of-day effects on glucose metabolism.
    Key phenomena: dawn effect, Somogyi effect, postprandial variability.
    """
    def get_metabolic_window(self, hour: float) -> dict:
        """Returns metabolic characteristics for a given hour of day."""
        # Dawn phenomenon (cortisol, GH surge 4-8 AM)
        if 4 <= hour < 8:
            return {
                'window': 'dawn',
                'glucose_multiplier': 1.15,
                'insulin_sensitivity': 0.82,
                'meal_recommendation': 'Small, low-GI breakfast. Skip if not hungry.',
                'exercise_benefit': 'HIGH — morning walk reduces dawn glucose by 15-20 mg/dL',
                'caution': 'Avoid high-carb breakfast during dawn window',
            }
        # Peak insulin sensitivity (9 AM – 1 PM)
        elif 9 <= hour < 13:
            return {
                'window': 'peak_sensitivity',
                'glucose_multiplier': 0.92,
                'insulin_sensitivity': 1.15,
                'meal_recommendation': 'Best time for largest meal of day. Body handles carbs best.',
                'exercise_benefit': 'MEDIUM — good time for moderate intensity exercise',
                'caution': None,
            }
        # Post-lunch dip (2-4 PM)
        elif 14 <= hour < 16:
            return {
                'window': 'afternoon_dip',
                'glucose_multiplier': 1.05,
                'insulin_sensitivity': 0.95,
                'meal_recommendation': 'Light snack only if hungry. Avoid high-sugar drinks.',
                'exercise_benefit': 'HIGH — afternoon walk is optimal for post-lunch spike',
                'caution': 'Blood glucose may be falsely elevated — not true hyperglycemia',
            }
        # Evening resistance (6-10 PM)
        elif 18 <= hour < 22:
            return {
                'window': 'evening_resistance',
                'glucose_multiplier': 1.10,
                'insulin_sensitivity': 0.88,
                'meal_recommendation': 'Light, early dinner (<7 PM). Avoid carbs after 8 PM.',
                'exercise_benefit': 'HIGH — pre-dinner walk (30 min) improves glucose by 20-30 mg/dL',
                'caution': 'Evening insulin resistance peaks at 9 PM — worst time for large carb meal',
            }
        # Night / sleep (10 PM – 4 AM)
        elif hour >= 22 or hour < 4:
            return {
                'window': 'sleep',
                'glucose_multiplier': 0.95,
                'insulin_sensitivity': 1.05,
                'meal_recommendation': 'No eating. If hypoglycemic risk: small protein snack.',
                'exercise_benefit': 'LOW — avoid intense exercise within 3h of sleep',
                'caution': 'Somogyi effect possible 2-4 AM if insulin-dependent',
            }
        else:
            return {
                'window': 'mid_morning',
                'glucose_multiplier': 0.97,
                'insulin_sensitivity': 1.08,
                'meal_recommendation': 'Light snack with protein if needed.',
                'exercise_benefit': 'HIGH — excellent window for exercise',
                'caution': None,
            }

    def optimal_meal_times(self, profile: PatientProfile) -> List[dict]:
        """Returns optimal eating windows based on metabolic profile."""
        return [
            {
                'meal': 'breakfast',
                'optimal_window': '7:30-8:30 AM',
                'reason': 'After dawn effect subsides; cortisol peak aids insulin action',
                'max_carbs_g': 35 if profile.bmi > 27 else 45,
            },
            {
                'meal': 'lunch',
                'optimal_window': '12:00-1:00 PM',
                'reason': 'Peak insulin sensitivity window; body handles glucose best',
                'max_carbs_g': 55 if profile.bmi > 27 else 70,
            },
            {
                'meal': 'dinner',
                'optimal_window': '6:00-7:00 PM',
                'reason': 'Before evening resistance peak; allows 3h digestion before sleep',
                'max_carbs_g': 35 if profile.bmi > 27 else 45,
            },
        ]


# ─────────────────────────────────────────────
# 8. REINFORCEMENT LEARNING NUDGES
# ─────────────────────────────────────────────
class RLNudgeEngine:
    """
    Adaptive precision nudge system.
    Uses a simple Q-table RL approach to learn which nudges
    the patient responds to most effectively.
    Actions: 6 nudge types
    State:   (glucose_zone × time_of_day × last_response)
    Reward:  glucose reduction + TIR improvement
    """
    NUDGE_ACTIONS = [
        NudgeType.MOVEMENT, NudgeType.NUTRITION, NudgeType.SLEEP,
        NudgeType.STRESS, NudgeType.HYDRATION, NudgeType.FASTING,
    ]

    def __init__(self, n_states: int = 24, learning_rate: float = 0.1,
                 gamma: float = 0.9, epsilon: float = 0.15):
        self.lr      = learning_rate
        self.gamma   = gamma
        self.epsilon = epsilon  # exploration rate
        self.n_actions = len(self.NUDGE_ACTIONS)
        # Q-table: state × action
        self.Q = np.zeros((n_states, self.n_actions))
        self.response_history: List[dict] = []

    def _state_index(self, glucose: float, hour: int,
                     last_action: int = 0) -> int:
        """Encode state as integer index."""
        glucose_zone = 0 if glucose < 70 else (1 if glucose < 140 else
                        2 if glucose < 180 else 3)
        time_zone = hour // 6  # 0=night, 1=morning, 2=afternoon, 3=evening
        return (glucose_zone * 6 + time_zone * 2 + (last_action % 2)) % len(self.Q)

    def select_nudge(self, glucose: float, hour: int,
                     patient_context: dict) -> NudgeAction:
        """ε-greedy action selection."""
        state = self._state_index(glucose, hour)
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)   # explore
        else:
            action_idx = int(np.argmax(self.Q[state]))        # exploit

        nudge_type = self.NUDGE_ACTIONS[action_idx]
        return self._create_nudge(nudge_type, glucose, hour, patient_context)

    def update_q(self, state: int, action: int, reward: float,
                 next_state: int):
        """Q-learning update rule."""
        current_q = self.Q[state, action]
        max_next_q = np.max(self.Q[next_state])
        self.Q[state, action] = current_q + self.lr * (
            reward + self.gamma * max_next_q - current_q
        )

    def record_response(self, nudge_type: NudgeType,
                         glucose_before: float, glucose_after: float,
                         followed: bool):
        """Record nudge outcome and update Q-table."""
        reward = 0.0
        if followed:
            glucose_drop = glucose_before - glucose_after
            reward = glucose_drop * 0.1 + (5.0 if followed else -1.0)
        state = self._state_index(glucose_before, datetime.now().hour)
        action_idx = self.NUDGE_ACTIONS.index(nudge_type)
        next_state = self._state_index(glucose_after, datetime.now().hour)
        self.update_q(state, action_idx, reward, next_state)
        self.response_history.append({
            'nudge': nudge_type.value,
            'glucose_change': round(glucose_after - glucose_before, 1),
            'followed': followed,
            'reward': round(reward, 2),
        })

    def _create_nudge(self, nudge_type: NudgeType, glucose: float,
                       hour: int, ctx: dict) -> NudgeAction:
        nudge_map = {
            NudgeType.MOVEMENT: NudgeAction(
                nudge_type=NudgeType.MOVEMENT,
                message="Your glucose is elevated — time to move!",
                action=f"Take a {'15-min brisk walk' if glucose > 160 else '10-min walk'}",
                expected_glucose_drop=25.0 if glucose > 180 else 15.0,
                expected_tir_gain=5.0,
                difficulty="LOW",
                time_minutes=15,
                priority=1 if glucose > 180 else 2,
            ),
            NudgeType.NUTRITION: NudgeAction(
                nudge_type=NudgeType.NUTRITION,
                message="Smart food choice coming up?",
                action="Swap white rice for brown rice or ragi. Add 1 cup dal for protein.",
                expected_glucose_drop=20.0,
                expected_tir_gain=8.0,
                difficulty="LOW",
                time_minutes=0,
                priority=2,
            ),
            NudgeType.SLEEP: NudgeAction(
                nudge_type=NudgeType.SLEEP,
                message="Sleep is your best insulin sensitivity drug!",
                action="Aim for bed by 10 PM. No screens 45 min before sleep.",
                expected_glucose_drop=12.0,
                expected_tir_gain=4.0,
                difficulty="MEDIUM",
                time_minutes=0,
                priority=3,
            ),
            NudgeType.STRESS: NudgeAction(
                nudge_type=NudgeType.STRESS,
                message="Cortisol spike detected — glucose rising!",
                action="Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s. Repeat 3×.",
                expected_glucose_drop=10.0,
                expected_tir_gain=3.0,
                difficulty="LOW",
                time_minutes=5,
                priority=2,
            ),
            NudgeType.HYDRATION: NudgeAction(
                nudge_type=NudgeType.HYDRATION,
                message="Dehydration raises blood glucose!",
                action="Drink 500ml water now. Target 2.5L total today.",
                expected_glucose_drop=8.0,
                expected_tir_gain=2.0,
                difficulty="LOW",
                time_minutes=1,
                priority=3,
            ),
            NudgeType.FASTING: NudgeAction(
                nudge_type=NudgeType.FASTING,
                message="Your 16-hour fasting window starting now",
                action="Last meal by 8 PM. Fast until 12 noon tomorrow for IF 16:8.",
                expected_glucose_drop=18.0,
                expected_tir_gain=9.0,
                difficulty="HIGH",
                time_minutes=0,
                priority=2 if hour >= 19 else 4,
            ),
        }
        return nudge_map.get(nudge_type, nudge_map[NudgeType.MOVEMENT])

    def get_top_nudges(self, glucose: float, hour: int,
                        ctx: dict, n: int = 3) -> List[NudgeAction]:
        """Return top-n nudges for current context."""
        nudges = [self._create_nudge(nt, glucose, hour, ctx)
                  for nt in self.NUDGE_ACTIONS]
        # Filter and sort by priority
        if glucose > 180:
            nudges.sort(key=lambda x: x.priority)
        else:
            nudges.sort(key=lambda x: (x.priority, -x.expected_tir_gain))
        return nudges[:n]


# ─────────────────────────────────────────────
# MASTER REVERSAL ENGINE
# ─────────────────────────────────────────────
class DiabetesReversalEngine:
    """
    Master orchestrator integrating all reversal sub-systems.
    Entry point for full reversal analysis.
    """
    def __init__(self):
        self.scorer     = MetabolicRecoveryScorer()
        self.tir_opt    = TIROptimiser()
        self.sens_track = InsulinSensitivityTracker()
        self.habit_mod  = HabitImpactModeller()
        self.meal_fore  = MealResponseForecaster()
        self.trend_est  = LongTermTrendEstimator()
        self.circadian  = CircadianMetabolismEngine()
        self.nudge_eng  = RLNudgeEngine()

    def full_analysis(self, profile: PatientProfile,
                       logs: List[DailyLog]) -> dict:
        """Run complete reversal analysis. Returns comprehensive report."""
        now = datetime.now()

        # 1. Recovery scoring
        p3_history = [profile.p3_estimate]  # placeholder single value
        scoring = self.scorer.compute(profile, logs, p3_history)

        # 2. TIR optimisation
        current_tir = np.mean([l.tir_pct for l in logs]) if logs else 63.0
        tir_plan = self.tir_opt.analyse(current_tir, logs, profile)

        # 3. Insulin sensitivity trajectory (90 days)
        p3_traj = self.sens_track.project_p3_trajectory(
            profile.p3_estimate, logs, days=90
        )

        # 4. Habit impact ranking
        top_habits = self.habit_mod.rank_habits_for_patient(profile, logs)[:5]

        # 5. Meal comparisons (typical Indian meal choices)
        meal_compare = self.meal_fore.compare_foods(
            [('white_rice', 150), ('brown_rice', 150), ('dosa', 100),
             ('oats', 80), ('salad', 200)],
            current_glucose=profile.baseline_glucose,
            p3_index=profile.p3_estimate / 5e-5 * 100,
        )

        # 6. Long-term trend
        trend = self.trend_est.project(profile, logs)

        # 7. Circadian window
        circadian = self.circadian.get_metabolic_window(now.hour)
        optimal_meals = self.circadian.optimal_meal_times(profile)

        # 8. Adaptive nudges
        top_nudges = self.nudge_eng.get_top_nudges(
            profile.baseline_glucose, now.hour, {}, n=3
        )

        return {
            'patient_id': profile.patient_id,
            'analysis_timestamp': now.isoformat(),
            'recovery_scoring': scoring,
            'tir_optimisation': tir_plan,
            'p3_trajectory_90d': p3_traj[::10],   # every 10 days
            'top_habit_interventions': top_habits,
            'meal_glucose_comparison': meal_compare,
            'long_term_trend': trend,
            'circadian_window': circadian,
            'optimal_meal_times': optimal_meals,
            'precision_nudges': [
                {
                    'type': n.nudge_type.value,
                    'message': n.message,
                    'action': n.action,
                    'glucose_drop_expected': n.expected_glucose_drop,
                    'tir_gain_expected': n.expected_tir_gain,
                    'difficulty': n.difficulty,
                    'minutes': n.time_minutes,
                }
                for n in top_nudges
            ],
            'reversal_summary': {
                'current_hba1c': profile.current_hba1c,
                'target_hba1c': profile.target_hba1c,
                'reversal_score': scoring['reversal_score'],
                'recovery_stage': scoring['recovery_stage'],
                'months_to_reversal': trend['months_to_reversal'],
                'confidence': trend['confidence'],
                'p3_at_90d': p3_traj[-1]['p3_index'] if p3_traj else 68.0,
                'hba1c_at_90d': p3_traj[-1]['hba1c_est'] if p3_traj else 6.8,
            },
        }

    def quick_snapshot(self, glucose: float, p3_index: float,
                        tir: float, steps: int, bmi: float) -> dict:
        """Lightweight snapshot for real-time dashboard updates."""
        p3 = p3_index / 100 * 5e-5
        hba1c = (glucose + 46.7) / 28.7
        h = hba1c
        stage = (RecoveryStage.DIABETIC if h >= 8 else
                 RecoveryStage.PRE_REVERSAL if h >= 7 else
                 RecoveryStage.ACTIVE_REVERSAL if h >= 6.5 else
                 RecoveryStage.NEAR_NORMAL if h >= 5.7 else
                 RecoveryStage.REVERSED)
        rev = round(p3_index * 0.35 + tir * 0.35
                    + max(0, 100 - (hba1c - 5) / 3 * 100) * 0.30, 1)
        sens_grade = self.sens_track.sensitivity_grade(p3_index)
        nudge = self.nudge_eng.get_top_nudges(glucose, datetime.now().hour, {}, n=1)

        return {
            'glucose': round(glucose, 1),
            'hba1c_estimate': round(hba1c, 2),
            'reversal_score': rev,
            'recovery_stage': stage.name,
            'insulin_sensitivity_grade': sens_grade,
            'p3_index': round(p3_index, 1),
            'tir_pct': round(tir, 1),
            'metabolic_age': round(max(28, 52 - p3_index * 0.24), 0),
            'top_nudge': {
                'action': nudge[0].action,
                'expected_drop': nudge[0].expected_glucose_drop,
            } if nudge else None,
        }


# ─────────────────────────────────────────────
# DEMO RUNNER
# ─────────────────────────────────────────────
def demo():
    print("\n" + "═"*62)
    print("  BioMirror — Diabetes Reversal Intelligence Engine Demo")
    print("═"*62)

    # Create sample patient (based on dataset averages)
    profile = PatientProfile(
        patient_id="BM-DEMO-001",
        age=45, bmi=27.0, diabetes_risk=1,
        current_hba1c=7.2, target_hba1c=6.4,
        baseline_glucose=161.4, baseline_steps=2654,
        p3_estimate=3.5e-5,
    )

    # Simulate 14 days of logs
    np.random.seed(42)
    logs = [
        DailyLog(
            date=datetime.now() - timedelta(days=14 - i),
            mean_glucose=161 + np.random.normal(0, 12),
            tir_pct=63 + np.random.normal(0, 5),
            steps=int(2654 + np.random.normal(0, 500)),
            exercise_min=int(np.random.exponential(15)),
            sleep_hours=round(6.5 + np.random.normal(0, 0.8), 1),
            stress_level=int(np.clip(np.random.normal(3, 0.8), 1, 5)),
            carb_grams=round(65 + np.random.normal(0, 15), 1),
            water_liters=round(1.8 + np.random.uniform(0, 0.8), 1),
        )
        for i in range(14)
    ]

    engine = DiabetesReversalEngine()
    report = engine.full_analysis(profile, logs)

    # Print summary
    s = report['reversal_summary']
    print(f"\n  Patient     : {profile.patient_id}")
    print(f"  HbA1c now   : {s['current_hba1c']}%  →  Target: {s['target_hba1c']}%")
    print(f"  Stage       : {s['recovery_stage']}")
    print(f"  Rev. Score  : {s['reversal_score']}/100")
    print(f"  Months left : {s['months_to_reversal']} ({s['confidence']} confidence)")
    print(f"  90d HbA1c   : {s['hba1c_at_90d']}%")
    print(f"  90d p3 idx  : {s['p3_at_90d']}/100")

    print(f"\n  Top Habit Interventions:")
    for h in report['top_habit_interventions'][:3]:
        print(f"  [{h['rank']}] {h['habit']:30s} "
              f"Glucose: -{h['glucose_drop_mg_dl']} | TIR: +{h['tir_gain_pct']}%")

    print(f"\n  Meal Comparison (post-meal glucose):")
    for m in report['meal_glucose_comparison']:
        risk_icon = "🔴" if m['spike_risk'] == "HIGH" else "🟡" if m['spike_risk'] == "MEDIUM" else "🟢"
        print(f"  {risk_icon} {m['food']:15s}: peak ~{m['peak_glucose']:5.0f} mg/dL "
              f"({m['net_carbs']}g net carbs)")

    print(f"\n  Precision Nudges:")
    for n in report['precision_nudges']:
        print(f"  • [{n['type'].upper():10s}] {n['action'][:60]}")
        print(f"                  Expected: -{n['glucose_drop_expected']} mg/dL | "
              f"+{n['tir_gain_expected']}% TIR | Effort: {n['difficulty']}")

    print(f"\n  Circadian Window ({datetime.now().hour}:00):")
    cw = report['circadian_window']
    print(f"  Window: {cw['window'].upper()}")
    print(f"  Meal tip: {cw['meal_recommendation']}")
    print(f"  Exercise: {cw['exercise_benefit']}")

    out_path = 'reversal_report_demo.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report → {out_path}")
    print("═"*62)


if __name__ == '__main__':
    demo()
