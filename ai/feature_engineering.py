"""
BioMirror — Feature Engineering & Dataset Preprocessing Pipeline
================================================================
Dataset: diabetes_lifestyle_dataset_500.csv
  Columns: time, meal_type, food, carbs, sugar, steps, exercise_min,
           activity_type, heart_rate, prev_glucose, predicted_glucose,
           BMI, family_history, diabetes_risk, health_score

Dataset Stats (N=500):
  Avg glucose : 161.4 mg/dL   High glucose rate : 37%
  TIR (70-180): 63%           Avg BMI           : 27.0
  Avg steps   : 2,654/day     Avg carbs         : 30.7 g/meal

Pipeline:
  1.  Load & audit
  2.  Missing value imputation
  3.  Outlier detection (IQR + Z-score)
  4.  Normalization (MinMax + StandardScaler)
  5.  Feature engineering (9 advanced derived features)
  6.  Temporal sequence generation
  7.  Synthetic sparse glucose timeline creation
  8.  Metabolic event simulation
  9.  FHIR-compatible export
  10. Train / val / test split with stratification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import json, os, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
GLUCOSE_LOW   = 70    # mg/dL
GLUCOSE_NORM  = 140   # mg/dL
GLUCOSE_HIGH  = 180   # mg/dL
GLUCOSE_CRIT  = 250   # mg/dL
TARGET_STEPS  = 8000  # recommended daily steps
TARGET_SLEEP  = 8.0   # hours
MEAL_PEAK_MIN = 45    # minutes post-meal glucose peak
RECOVERY_MIN  = 120   # minutes for glucose recovery

GI_TABLE = {  # Glycemic Index reference values
    'rice': 0.72, 'bread': 0.70, 'dosa': 0.68, 'roti': 0.60,
    'oats': 0.55, 'apple': 0.36, 'banana': 0.51, 'salad': 0.15,
    'pasta': 0.49, 'pizza': 0.60, 'idli': 0.65, 'poha': 0.55,
    'default': 0.55,
}


# ─────────────────────────────────────────────
# 1. LOAD & AUDIT
# ─────────────────────────────────────────────
def load_and_audit(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n{'='*55}")
    print(f"  BioMirror Feature Engineering Pipeline")
    print(f"{'='*55}")
    print(f"[Load] Records: {len(df)} | Columns: {df.shape[1]}")
    print(f"[Missing]\n{df.isnull().sum().to_string()}")
    print(f"\n[Dtypes]\n{df.dtypes.to_string()}")
    print(f"\n[Glucose stats]\n"
          f"  Mean : {df['prev_glucose'].mean():.1f} mg/dL\n"
          f"  Std  : {df['prev_glucose'].std():.1f}\n"
          f"  Min  : {df['prev_glucose'].min()}\n"
          f"  Max  : {df['prev_glucose'].max()}\n"
          f"  TIR  : {df['prev_glucose'].between(70,180).mean()*100:.1f}%\n"
          f"  High : {(df['prev_glucose']>180).mean()*100:.1f}%\n"
          f"  Low  : {(df['prev_glucose']<70).mean()*100:.1f}%")
    return df


# ─────────────────────────────────────────────
# 2. MISSING VALUE IMPUTATION
# ─────────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    # Numeric: forward fill → median fill
    for col in numeric_cols:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            df[col] = df[col].ffill().bfill()
            df[col] = df[col].fillna(df[col].median())
            print(f"[Impute] {col}: {n_miss} filled with ffill+median")

    # Categorical: mode fill
    for col in categorical_cols:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"[Impute] {col}: {n_miss} filled with mode")

    return df


# ─────────────────────────────────────────────
# 3. OUTLIER DETECTION & CAPPING
# ─────────────────────────────────────────────
def detect_and_cap_outliers(df: pd.DataFrame) -> tuple:
    df = df.copy()
    outlier_report = {}

    # Physiological bounds for glucose
    BOUNDS = {
        'prev_glucose':      (40, 400),
        'predicted_glucose': (40, 400),
        'heart_rate':        (40, 200),
        'BMI':               (15, 55),
        'steps':             (0, 50000),
        'exercise_min':      (0, 300),
        'carbs':             (0, 300),
        'sugar':             (0, 150),
    }

    # Z-score method for general outliers
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        z = np.abs(stats.zscore(df[col].dropna()))
        n_outliers = (z > 3.5).sum()
        if n_outliers > 0:
            outlier_report[col] = int(n_outliers)

    # Physiological capping
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            n_before = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lo, hi)
            if n_before > 0:
                print(f"[Outlier] {col}: clipped {n_before} values to [{lo},{hi}]")

    # IQR capping for remaining numerics
    for col in numeric_cols:
        if col not in BOUNDS:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 3 * IQR, Q3 + 3 * IQR
            n_clip = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_clip > 0:
                df[col] = df[col].clip(lo, hi)

    print(f"[Outlier] Z-score outliers detected: {outlier_report}")
    return df, outlier_report


# ─────────────────────────────────────────────
# 4. TIME PARSING & CIRCADIAN ENCODING
# ─────────────────────────────────────────────
def parse_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Parse HH:MM to float hours
    df['hour'] = df['time'].str.extract(r'(\d+):', expand=False).astype(float)
    df['minute'] = df['time'].str.extract(r':(\d+)', expand=False).fillna(0).astype(float)
    df['hour_decimal'] = df['hour'] + df['minute'] / 60

    # Circadian rhythm encoding (Fourier basis)
    df['hour_sin']    = np.sin(2 * np.pi * df['hour_decimal'] / 24)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour_decimal'] / 24)
    df['hour_sin_2x'] = np.sin(4 * np.pi * df['hour_decimal'] / 24)
    df['hour_cos_2x'] = np.cos(4 * np.pi * df['hour_decimal'] / 24)

    # Meal period flags
    df['is_breakfast'] = df['hour_decimal'].between(6, 10).astype(int)
    df['is_lunch']     = df['hour_decimal'].between(11, 14).astype(int)
    df['is_dinner']    = df['hour_decimal'].between(18, 22).astype(int)
    df['is_night']     = ((df['hour_decimal'] >= 22) | (df['hour_decimal'] < 6)).astype(int)
    df['is_postmeal']  = (df['meal_type'].notna()).astype(int)

    return df


# ─────────────────────────────────────────────
# 5. ADVANCED FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 5.1 GLYCEMIC LOAD SCORE ──────────────────────────
    # GL = carbs × GI / 100  (reflects glucose impact of meal)
    def get_gi(food: str) -> float:
        if pd.isna(food):
            return 0.0
        food_lower = str(food).lower()
        for key, gi in GI_TABLE.items():
            if key in food_lower:
                return gi
        return GI_TABLE['default']

    df['gi_factor'] = df['food'].apply(get_gi)
    sugar_ratio = (df['sugar'] / (df['carbs'].replace(0, 1))).clip(0, 1)
    df['gi_factor_adjusted'] = df['gi_factor'] * (1 + sugar_ratio * 0.3)
    df['glycemic_load_score'] = (df['carbs'] * df['gi_factor_adjusted']).clip(0, 150)
    print(f"[Feature] Glycemic Load Score — mean: {df['glycemic_load_score'].mean():.2f}")

    # ── 5.2 METABOLIC STRESS SCORE ───────────────────────
    # Composite of glucose deviation + activity deficit + risk
    glucose_deviation = np.abs(df['prev_glucose'] - 100) / 100  # normalized
    activity_deficit  = np.maximum(0, (TARGET_STEPS - df['steps']) / TARGET_STEPS)
    risk_factor       = df['diabetes_risk'] / 2.0
    health_inv        = 1 - df['health_score'] / 100
    df['metabolic_stress_score'] = (
        glucose_deviation * 0.35 +
        activity_deficit  * 0.25 +
        risk_factor       * 0.25 +
        health_inv        * 0.15
    ).clip(0, 1) * 100
    print(f"[Feature] Metabolic Stress Score — mean: {df['metabolic_stress_score'].mean():.2f}")

    # ── 5.3 INSULIN SENSITIVITY PROXY ────────────────────
    # Higher BMI + higher glucose → lower sensitivity
    # p3_proxy ∝ 1 / (BMI × glucose_ratio)
    glucose_ratio = df['prev_glucose'] / 90.0  # normalized to basal
    df['insulin_sensitivity_proxy'] = (
        100 / (df['BMI'] / 22 * glucose_ratio + 0.5)
    ).clip(0, 100)
    print(f"[Feature] Insulin Sensitivity — mean: {df['insulin_sensitivity_proxy'].mean():.2f}/100")

    # ── 5.4 RECOVERY SCORE ───────────────────────────────
    # How well body is recovering: TIR tendency, low stress, good activity
    tir_bonus = df['prev_glucose'].between(70, 180).astype(float) * 20
    activity_bonus = (df['steps'] / TARGET_STEPS).clip(0, 1) * 15
    df['recovery_score'] = (
        100
        - df['metabolic_stress_score'] * 0.5
        + tir_bonus
        + activity_bonus
        - df['diabetes_risk'] * 8
        + df['family_history'] * (-5)
    ).clip(0, 100)
    print(f"[Feature] Recovery Score — mean: {df['recovery_score'].mean():.2f}/100")

    # ── 5.5 CIRCADIAN RHYTHM FACTOR ──────────────────────
    # Dawn phenomenon: glucose naturally higher in early morning (4-8 AM)
    # Postprandial windows: 1-2h post meal glucose naturally peaks
    dawn_effect = np.where(df['hour_decimal'].between(4, 8), 1.15, 1.0)
    somogyi     = np.where(df['hour_decimal'].between(2, 4), 0.90, 1.0)  # rebound hypoglycemia
    df['circadian_rhythm_factor'] = dawn_effect * somogyi
    print(f"[Feature] Circadian Factor — range: {df['circadian_rhythm_factor'].min():.2f}–"
          f"{df['circadian_rhythm_factor'].max():.2f}")

    # ── 5.6 MEAL IMPACT FACTOR ───────────────────────────
    # Expected glucose rise from meal = GL × circadian × (1/sensitivity)
    df['meal_impact_factor'] = (
        df['glycemic_load_score'] * df['circadian_rhythm_factor']
        / (df['insulin_sensitivity_proxy'] / 100 + 0.1)
    ).clip(0, 100)
    print(f"[Feature] Meal Impact Factor — mean: {df['meal_impact_factor'].mean():.2f}")

    # ── 5.7 GLUCOSE VOLATILITY ───────────────────────────
    # Rolling std over last 3 records (proxy for CGM variability)
    df['glucose_volatility'] = df['prev_glucose'].rolling(
        window=3, min_periods=1
    ).std().fillna(0)
    df['coefficient_of_variation'] = (
        df['glucose_volatility'] / (df['prev_glucose'] + 1) * 100
    ).clip(0, 50)
    print(f"[Feature] Glucose Volatility — mean: {df['glucose_volatility'].mean():.2f} mg/dL")

    # ── 5.8 ACTIVITY-ADJUSTED GLUCOSE RESPONSE ───────────
    # Each additional 1000 steps ≈ -2 mg/dL post-exercise glucose reduction
    step_effect   = -(df['steps'] / 1000) * 2.0
    exercise_effect = -(df['exercise_min'] * 0.3)  # ~0.3 mg/dL per min
    hr_factor     = np.where(df['heart_rate'] > 120, -5, 0)  # intense exercise bonus
    df['activity_adjusted_glucose_response'] = (
        step_effect + exercise_effect + hr_factor
    ).clip(-50, 0)
    print(f"[Feature] Activity-Adj Response — mean: "
          f"{df['activity_adjusted_glucose_response'].mean():.2f} mg/dL")

    # ── 5.9 SLEEP IMPACT COEFFICIENT ─────────────────────
    # Poor sleep (<6h) → insulin resistance increases 18-22%
    # Using health_score as proxy for sleep quality (dataset limitation)
    sleep_quality = df['health_score'] / 100  # 0-1, 1=good
    sleep_hours_proxy = 4 + sleep_quality * 5  # 4–9h mapping
    sleep_deficit = np.maximum(0, TARGET_SLEEP - sleep_hours_proxy)
    df['sleep_impact_coefficient'] = (
        1 + sleep_deficit * 0.18
    ).clip(1.0, 1.8)  # up to 80% insulin resistance increase
    print(f"[Feature] Sleep Impact Coeff — range: "
          f"{df['sleep_impact_coefficient'].min():.2f}–"
          f"{df['sleep_impact_coefficient'].max():.2f}")

    # ── COMPOSITE REVERSAL SCORE ──────────────────────────
    df['reversal_probability'] = (
        df['insulin_sensitivity_proxy'] * 0.30 +
        df['recovery_score'] * 0.30 +
        df['health_score'] * 0.25 +
        (100 - df['metabolic_stress_score']) * 0.15
    ).clip(0, 100)

    # ── METABOLIC AGE ESTIMATE ────────────────────────────
    # Younger metabolic age = better insulin sensitivity
    df['metabolic_age'] = (
        50
        - df['insulin_sensitivity_proxy'] * 0.25
        + df['BMI'] * 0.5
        + df['glucose_volatility'] * 0.1
        + df['diabetes_risk'] * 3
    ).clip(25, 75).round(0)

    # ── HbA1c ESTIMATE ───────────────────────────────────
    # Nathan formula: HbA1c% = (glucose + 46.7) / 28.7
    df['hba1c_estimate'] = (
        (df['prev_glucose'].rolling(30, min_periods=1).mean() + 46.7) / 28.7
    ).clip(4.5, 14.0).round(2)

    # ── INFLAMMATION PROXY ────────────────────────────────
    df['inflammation_proxy'] = (
        df['metabolic_stress_score'] * 0.5
        + df['glucose_volatility'] * 0.3
        + df['BMI'] * 0.8
    ).clip(0, 100)

    # ── BERGMAN INSULIN ACTION X(t) ESTIMATE ─────────────
    df['X_bergman'] = ((df['prev_glucose'] / 90 - 1) * 0.042).clip(0, 0.2)

    # ── INSULIN PROXY I(t) ────────────────────────────────
    df['insulin_proxy'] = (7 + (df['prev_glucose'] - 90) * 0.1).clip(0, 100)

    return df


# ─────────────────────────────────────────────
# 6. TEMPORAL SEQUENCE GENERATION
# ─────────────────────────────────────────────
def generate_temporal_sequences(df: pd.DataFrame,
                                seq_len: int = 12,
                                pred_horizon: int = 6,
                                stride: int = 1) -> dict:
    """
    Creates overlapping temporal windows for sequence modeling.

    Returns dict with:
      X     : (N, seq_len, n_features) — input sequences
      y_G   : (N, pred_horizon)        — target glucose
      y_risk: (N,)                     — diabetes risk label
      meta  : (N,)                     — window metadata
    """
    feature_cols = [
        'prev_glucose', 'glycemic_load_score', 'metabolic_stress_score',
        'insulin_sensitivity_proxy', 'recovery_score', 'circadian_rhythm_factor',
        'meal_impact_factor', 'glucose_volatility', 'activity_adjusted_glucose_response',
        'sleep_impact_coefficient', 'X_bergman', 'insulin_proxy',
        'hour_sin', 'hour_cos', 'steps', 'exercise_min', 'heart_rate',
        'BMI', 'carbs', 'sugar', 'is_postmeal',
    ]

    available_cols = [c for c in feature_cols if c in df.columns]
    X_all, y_G_all, y_risk_all, meta_all = [], [], [], []
    n = len(df)

    for i in range(0, n - seq_len - pred_horizon + 1, stride):
        window = df.iloc[i: i + seq_len][available_cols].values
        future_G = df.iloc[i + seq_len: i + seq_len + pred_horizon]['prev_glucose'].values

        if len(window) == seq_len and len(future_G) == pred_horizon:
            X_all.append(window)
            y_G_all.append(future_G)
            y_risk_all.append(df.iloc[i + seq_len]['diabetes_risk'])
            meta_all.append({'start': i, 'time': df.iloc[i]['time']})

    X = np.array(X_all, dtype=np.float32)
    y_G = np.array(y_G_all, dtype=np.float32)
    y_risk = np.array(y_risk_all, dtype=np.int32)

    print(f"\n[Sequences] Generated {len(X)} windows "
          f"(seq_len={seq_len}, horizon={pred_horizon})")
    print(f"[Sequences] X shape: {X.shape} | y shape: {y_G.shape}")
    return {'X': X, 'y_glucose': y_G, 'y_risk': y_risk, 'meta': meta_all,
            'feature_names': available_cols}


# ─────────────────────────────────────────────
# 7. SYNTHETIC SPARSE GLUCOSE TIMELINE
# ─────────────────────────────────────────────
def create_sparse_glucose_timeline(df: pd.DataFrame,
                                   sparsity: float = 0.15,
                                   n_days: int = 7) -> pd.DataFrame:
    """
    Simulates what a finger-prick patient would have:
    Only 15% of datapoints observed (≈3-4 readings per day).
    The PINN must interpolate the full continuous timeline.
    """
    n = min(len(df), n_days * 24)
    timeline = df.head(n).copy()
    timeline['timestamp_min'] = np.arange(n) * 60  # 1 reading per hour
    timeline['is_observed']   = np.random.rand(n) < sparsity
    # Always observe first and last point
    timeline.iloc[0, timeline.columns.get_loc('is_observed')] = True
    timeline.iloc[-1, timeline.columns.get_loc('is_observed')] = True
    timeline['glucose_observed'] = np.where(
        timeline['is_observed'],
        timeline['prev_glucose'],
        np.nan
    )
    # Linear interpolation as baseline for comparison
    timeline['glucose_interp_linear'] = timeline['glucose_observed'].interpolate('linear')
    n_observed = timeline['is_observed'].sum()
    print(f"\n[Sparse Timeline] {n_days} days | {n} readings → "
          f"{n_observed} observed ({n_observed/n*100:.1f}%) | "
          f"{n - n_observed} to infer via PINN")
    return timeline


# ─────────────────────────────────────────────
# 8. METABOLIC EVENT SIMULATION (Bergman)
# ─────────────────────────────────────────────
def simulate_metabolic_events(df: pd.DataFrame, p3: float = 3.5e-5) -> pd.DataFrame:
    """
    Runs Bergman Minimal Model forward simulation for entire dataset.
    Generates: simulated_glucose, insulin_action_X, metabolic_state.
    """
    p1, p2 = 0.028, 0.025
    Gb, Ib = 90.0, 7.0
    dt = 5.0  # minutes

    G, X = float(df['prev_glucose'].iloc[0]), 0.0
    sim_G, sim_X = [], []

    for idx, row in df.iterrows():
        I = float(row['insulin_proxy']) if 'insulin_proxy' in df.columns else Ib
        # Meal input: Gaussian absorption curve, peak at 45 min
        carbs = float(row['carbs'])
        meal_input = carbs * 0.5 * np.exp(-0.5 * ((0 - MEAL_PEAK_MIN / dt) / 10) ** 2) / 10

        dG = -(p1 + X) * G + p1 * Gb + meal_input
        dX = -p2 * X + p3 * (I - Ib)
        G = max(40, min(400, G + dG * dt))
        X = max(0, X + dX * dt)
        sim_G.append(round(G, 2))
        sim_X.append(round(X, 5))

    df['simulated_glucose'] = sim_G
    df['insulin_action_X']  = sim_X

    # Metabolic state classification
    def classify(g):
        if g < 70:   return 'hypoglycemia'
        if g <= 140: return 'normal'
        if g <= 180: return 'elevated'
        if g <= 250: return 'high'
        return 'critical'

    df['metabolic_state'] = df['simulated_glucose'].apply(classify)
    state_dist = df['metabolic_state'].value_counts(normalize=True) * 100
    print(f"\n[Bergman Sim] p3={p3:.2e} | State distribution:")
    for state, pct in state_dist.items():
        print(f"  {state:12s}: {pct:.1f}%")
    return df


# ─────────────────────────────────────────────
# 9. NORMALIZATION
# ─────────────────────────────────────────────
def normalize_features(df: pd.DataFrame) -> tuple:
    """
    Three-tier normalization:
      - Glucose features: MinMaxScaler [0,1]
      - Derived scores:   Already [0,100], divide by 100
      - Raw numerics:     RobustScaler (outlier-resistant)
    """
    df = df.copy()

    glucose_cols  = ['prev_glucose', 'predicted_glucose', 'simulated_glucose',
                     'glucose_volatility']
    score_cols    = ['glycemic_load_score', 'metabolic_stress_score',
                     'insulin_sensitivity_proxy', 'recovery_score',
                     'reversal_probability', 'inflammation_proxy']
    robust_cols   = ['steps', 'exercise_min', 'heart_rate', 'BMI', 'carbs', 'sugar']

    scalers = {}

    # MinMax for glucose (physiological range 40-400)
    mm = MinMaxScaler(feature_range=(0, 1))
    available_g = [c for c in glucose_cols if c in df.columns]
    df[available_g] = mm.fit_transform(df[available_g])
    scalers['glucose_minmax'] = mm

    # Score normalization [0-100] → [0-1]
    available_s = [c for c in score_cols if c in df.columns]
    df[available_s] = df[available_s] / 100.0
    scalers['score_div100'] = 'divide_by_100'

    # RobustScaler for raw inputs
    rs = RobustScaler()
    available_r = [c for c in robust_cols if c in df.columns]
    df[available_r] = rs.fit_transform(df[available_r])
    scalers['robust'] = rs

    print(f"\n[Normalize] Applied MinMax({len(available_g)} cols) | "
          f"Div100({len(available_s)} cols) | Robust({len(available_r)} cols)")
    return df, scalers


# ─────────────────────────────────────────────
# 10. FHIR R4 EXPORT
# ─────────────────────────────────────────────
def export_fhir(df: pd.DataFrame, patient_id: str = "BioMirror-P001",
                output_path: str = None) -> list:
    """
    Converts dataset rows to HL7 FHIR R4 Observation resources.
    Compatible with any FHIR-capable EMR / health platform.
    """
    observations = []
    for idx, row in df.head(5).iterrows():  # sample 5 for demo
        obs = {
            "resourceType": "Observation",
            "id": f"glucose-obs-{idx}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "laboratory",
                    "display": "Laboratory"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "2339-0",
                    "display": "Glucose [Mass/volume] in Blood"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": f"2025-01-01T{row['time']}:00+05:30",
            "valueQuantity": {
                "value": float(row['prev_glucose']) if 'prev_glucose' in row else 0,
                "unit": "mg/dL",
                "system": "http://unitsofmeasure.org",
                "code": "mg/dL"
            },
            "component": [
                {
                    "code": {"coding": [{"system": "http://loinc.org",
                                         "code": "55284-4",
                                         "display": "Blood pressure"}]},
                    "valueQuantity": {"value": float(row.get('heart_rate', 80)),
                                      "unit": "bpm"}
                }
            ],
            "extension": [{
                "url": "http://biomirror.ai/fhir/extension/metabolic-twin",
                "valueDecimal": float(row.get('reversal_probability', 0))
                    if 'reversal_probability' in row else 0.0
            }]
        }
        observations.append(obs)

    fhir_bundle = {
        "resourceType": "Bundle",
        "id": f"biomirror-bundle-{patient_id}",
        "type": "collection",
        "entry": [{"resource": obs} for obs in observations]
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(fhir_bundle, f, indent=2)
        print(f"\n[FHIR] Exported {len(observations)} Observation resources → {output_path}")
    return observations


# ─────────────────────────────────────────────
# 11. STRATIFIED SPLIT
# ─────────────────────────────────────────────
def stratified_split(sequences: dict, test_size: float = 0.15,
                     val_size: float = 0.15) -> dict:
    """
    Stratified split on diabetes_risk label to maintain class distribution.
    Returns train / val / test dictionaries.
    """
    from sklearn.model_selection import train_test_split
    X, y_G, y_risk = sequences['X'], sequences['y_glucose'], sequences['y_risk']

    X_temp, X_test, yG_temp, yG_test, yr_temp, yr_test = train_test_split(
        X, y_G, y_risk, test_size=test_size, stratify=y_risk, random_state=42
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, yG_train, yG_val = train_test_split(
        X_temp, yG_temp, test_size=val_frac, random_state=42
    )

    print(f"\n[Split] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    for label, name in [(yr_temp[:len(X_train)], 'Train'),
                        (yr_temp[len(X_train):], 'Val'),
                        (yr_test, 'Test')]:
        unique, counts = np.unique(label, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  {name} risk dist: {dist}")

    return {
        'train': {'X': X_train, 'y': yG_train},
        'val':   {'X': X_val,   'y': yG_val},
        'test':  {'X': X_test,  'y': yG_test},
        'feature_names': sequences['feature_names'],
    }


# ─────────────────────────────────────────────
# FULL PIPELINE RUNNER
# ─────────────────────────────────────────────
def run_feature_pipeline(csv_path: str = 'diabetes_lifestyle_dataset_500.csv',
                          output_dir: str = '.') -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # Steps 1–3
    df = load_and_audit(csv_path)
    df = impute_missing(df)
    df, outlier_report = detect_and_cap_outliers(df)

    # Step 4: Time parsing
    df = parse_time_features(df)

    # Step 5: Feature engineering
    df = engineer_features(df)

    # Step 6: Metabolic simulation
    df = simulate_metabolic_events(df, p3=3.5e-5)

    # Step 7: Sparse timeline
    sparse_timeline = create_sparse_glucose_timeline(df, sparsity=0.15)

    # Print feature summary
    new_cols = ['glycemic_load_score', 'metabolic_stress_score',
                'insulin_sensitivity_proxy', 'recovery_score',
                'circadian_rhythm_factor', 'meal_impact_factor',
                'glucose_volatility', 'activity_adjusted_glucose_response',
                'sleep_impact_coefficient', 'reversal_probability',
                'metabolic_age', 'hba1c_estimate', 'inflammation_proxy']
    print(f"\n{'─'*55}")
    print(f"  DERIVED FEATURE STATISTICS")
    print(f"{'─'*55}")
    for col in new_cols:
        if col in df.columns:
            print(f"  {col:40s}: {df[col].mean():.2f} ± {df[col].std():.2f}")

    # Step 8: Sequences
    sequences = generate_temporal_sequences(df, seq_len=12, pred_horizon=6)

    # Step 9: FHIR export
    export_fhir(df, output_path=os.path.join(output_dir, 'fhir_bundle.json'))

    # Step 10: Split
    splits = stratified_split(sequences)

    # Step 11: Save processed dataset
    out_csv = os.path.join(output_dir, 'biomirror_processed.csv')
    df.to_csv(out_csv, index=False)
    print(f"\n[Save] Processed dataset → {out_csv} ({len(df)} rows, {len(df.columns)} cols)")

    print(f"\n{'='*55}")
    print(f"  Feature Engineering Complete ✓")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Sequences: {len(sequences['X'])}")
    print(f"{'='*55}")

    return {'df': df, 'sequences': sequences, 'splits': splits,
            'sparse_timeline': sparse_timeline, 'outlier_report': outlier_report}


if __name__ == '__main__':
    result = run_feature_pipeline(
        csv_path='diabetes_lifestyle_dataset_500.csv',
        output_dir='biomirror_outputs'
    )
