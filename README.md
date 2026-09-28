# BioMirror
Physics-Informed Digital Twin for Type-2 Diabetes Management

## 1. Project Overview

BioMirror is a software framework for glucose forecasting and metabolic-state simulation. It combines a global Physics-Informed Neural Network (PINN), a Bergman glucose-insulin model, per-user health logs and residual correction, and a What-If scenario simulator. The backend exposes metabolic metrics and the static browser dashboard visualizes the returned data.

The PINN is implemented and the checkpoint loads in the audited local runtime. Its live output is checked on each forecast request; requests whose predictions fail the code's plausibility gate use the Bergman-based short-term fallback instead. The current runtime behavior is described in [Prediction and Fallback](#prediction-and-fallback).

## 2. Key Features

- User registration, login, token refresh/revocation, and password reset routes.
- Manual logging for glucose, meals, activity, sleep, stress, medication, weight, blood pressure, water, and HbA1c.
- Derived feature engineering from glucose and lifestyle inputs.
- A global `BioMirrorPINN` multi-step glucose model with a physics-informed training loss.
- A Bergman-based Digital Twin with persisted state and glucose history.
- Per-request PINN plausibility checks and Bergman fallback for rejected or unavailable model outputs.
- User-specific additive prediction correction computed from recent matched forecast errors.
- 1-72 hour forecast responses, including a long-horizon Bergman trajectory and a separate 30-minute prediction.
- What-If meal, exercise, sleep, and stress scenario simulation.
- Metabolic metrics, retrospective forecast evaluation, history, and recommendation endpoints.
- A static HTML/CSS/JavaScript dashboard with Chart.js visualizations.
- An AI Coach endpoint. It requires a responding Groq or Anthropic provider; without one it returns HTTP 503. No offline rule-based fallback is implemented in this route.

## 3. System Architecture

```text
User-entered profile, glucose, and lifestyle logs
                         |
                         v
Live feature engineering and persisted Bergman Digital Twin state
                         |
                         v
Forecast request: global PINN attempted for the next 30 minutes
                         |
                         v
Per-request plausibility gate -- rejected/unavailable? --> Bergman short-term fallback
                         |
                         v
Long-horizon Bergman forecast + accepted PINN or fallback near-term values
                         |
                         v
Per-user recent residual correction
                         |
                         v
Forecast/metrics API --> static dashboard; What-If requests run a separate scenario
```

**Global model:** the checkpoint-backed `BioMirrorPINN` is shared; it is not retrained from user logs during requests.

**Digital Twin:** `backend/backend_api.py` stores each user's current glucose `G`, insulin-action state `X`, insulin proxy `I`, parameter `p3`, simulation time, and state history. Its updates use the Bergman equations below.

**Personalization:** recent matched actual-minus-predicted residuals are averaged per patient and added to subsequent forecast values. This is separate from the global checkpoint.

**Forecasting:** the route always constructs a Bergman long-horizon trajectory. It separately attempts PINN inference for the six 5-minute steps; the fallback for that output is the first six values of the Bergman forecast.

**What-If simulation:** `/simulate` runs a separate 48-step scenario and baseline from the supplied scenario inputs and current twin state. It does not alter the global PINN.

## 4. Methodology / Technical Implementation

### Data Processing and Feature Engineering

`ai/feature_engineering.py:engineer_features()` computes the derived features used by the live feature bridge and offline pipeline. `ai/pinn_model.py:GlucoseDataset._engineer_features()` separately prepares the model's 12 input columns; the two sets should not be treated as identical.

Implemented examples (clipping omitted unless shown):

```text
sugar_ratio = clip(sugar / (carbs with zero replaced by 1), 0, 1)
GI_adjusted = GI(food) * (1 + 0.3 * sugar_ratio)
glycemic_load_score = clip(carbs * GI_adjusted, 0, 150)

activity_deficit = max(0, (8000 - steps) / 8000)
metabolic_stress_score = 100 * clip(
    0.35 * abs(glucose - 100) / 100
  + 0.25 * activity_deficit
  + 0.25 * diabetes_risk / 2
  + 0.15 * (1 - health_score / 100), 0, 1)

insulin_sensitivity_proxy = clip(
    100 / ((BMI / 22) * (glucose / 90) + 0.5), 0, 100)

recovery_score = clip(
    100 - 0.5 * metabolic_stress_score + TIR_bonus
  + 15 * clip(steps / 8000, 0, 1)
  - 8 * diabetes_risk - 5 * family_history, 0, 100)

meal_impact_factor = clip(
    glycemic_load_score * circadian_rhythm_factor
    / (insulin_sensitivity_proxy / 100 + 0.1), 0, 100)

activity_adjusted_glucose_response = clip(
    -2 * steps / 1000 - 0.3 * exercise_min
    - 5 * indicator(heart_rate > 120), -50, 0)

X_bergman = clip((glucose / 90 - 1) * 0.042, 0, 0.2)
```

`TIR_bonus` is 20 when glucose is in [70, 180] mg/dL and 0 otherwise. The circadian features include daily sine/cosine encodings and second-harmonic encodings in `parse_time_features`; the live feature function also applies its coded dawn/Somogyi multipliers. Sleep impact uses `health_score` as a sleep-quality proxy in this dataset pipeline; it is not a measurement from a sleep sensor. The pipeline also calculates rolling glucose volatility, coefficient of variation, metabolic-age and HbA1c estimates, reversal probability, and an insulin proxy.

### Physics-Informed Neural Network

`ai/pinn_model.py` implements `BioMirrorPINN`. Its checkpoint's feature columns per time step are: `glucose_norm`, `carbs_norm`, `steps_norm`, `exercise_norm`, `heart_rate_norm`, `bmi_norm`, `stress_proxy_norm`, `hour_sin`, `hour_cos`, `glycemic_load_norm`, `X_insulin_action`, and `I_insulin_proxy`. The configured sequence is 12 steps and the output horizon is six 5-minute steps (30 minutes). The architecture embeds features, applies two temporal-attention blocks, uses a bidirectional GRU and residual MLP decoder, and has glucose, insulin-action, and reversal output heads.

At live inference, `backend/ai_bridge.py` builds the 12-step input from recent glucose values and repeats the current feature snapshot for the other input columns. It does not rebuild a complete per-timestep history of lifestyle features.

`BioMirrorLoss` implements:

```text
L_total = lambda_1 * L_data
        + lambda_2 * L_physics
        + lambda_3 * L_bc
        + lambda_4 * L_temporal
```

The configured weights are 1.0, 0.5, 0.3, and 0.2, respectively.

- `L_data`: Huber loss (delta 10) between predicted and target glucose.
- `L_physics`: mean-squared Bergman ODE residuals for glucose and insulin action, using finite differences and midpoint states.
- `L_bc`: squared penalties for glucose predictions below 40 or above 400 mg/dL.
- `L_temporal`: squared penalty for glucose changes exceeding 3 mg/dL/min, calculated over 5-minute prediction steps.

The differentiable Bergman ODE solver clamps glucose to [40, 400] and its insulin-action state to [0, 1]. These are software constraints, not evidence of clinical validity. The live route does not use the PINN as the unconditional source of every prediction.

### Bergman Digital Twin

`BergmanEngine.step()` in `backend/backend_api.py` uses the following implemented dynamics, where `meal` is the meal input passed to the step:

```text
dG/dt = -(p1 + X) * G + p1 * Gb + meal
dX/dt = -p2 * X + p3 * (I - Ib)
G_next = clip(G + dG/dt * dt, 40, 400)
X_next = max(0, X + dX/dt * dt)
```

The live engine defaults to `p1=0.028`, `p2=0.025`, `Gb=90 mg/dL`, `Ib=7`, and a 5-minute step. Each twin's `p3` is initialized from the user's BMI and can be adjusted by logged exercise; it is not estimated from a clinical insulin-sensitivity test. Meal input in the simulator uses the implemented Gaussian-shaped `meal_bolus()` function.

### Prediction and Fallback

`GET /forecast/{patient_id}` simulates `hours * 12` five-minute steps with the Bergman engine, using a coded insulin proxy and built-in forecast meal assumptions at offsets 7, 13, and 19 hours from the forecast start (50 g, 65 g, and 55 g). These are fixed simulation inputs, not future meals inferred from the user's actual schedule. The route then attempts the PINN for a separate 30-minute output.

`backend/ai_bridge.py:PINNPredictor.predict()` rejects a PINN result if any value is outside 40-400 mg/dL or if its absolute difference from current twin glucose exceeds 80 mg/dL times its one-based step index. When inference is unavailable or rejected, the route sets the near-term values to the first six values of its Bergman forecast and labels the model status accordingly.

**Current audited runtime:** the API health response confirms the checkpoint is loaded. Forecast requests observed in this runtime generated plausibility-gate rejection warnings, so the active prediction path for those requests was the Bergman forecast/fallback, not accepted PINN output. The gate is evaluated per request; this statement does not assert every possible input or future checkpoint will be rejected.

### Personalization

`get_prediction_personalization()` averages up to the five most recent matched feedback errors for the patient. `record_prediction_error()` matches an unused forecast to an actual glucose log near the five-minute target horizon and stores:

```text
e_t = actual_glucose_t - predicted_glucose_t
C_u = mean(the user's recent matched e_t values)
personalized_prediction = base_prediction + C_u
```

If there are no matched errors, `C_u` is zero. The same user correction is applied additively to forecast arrays; the shared PINN checkpoint is not modified or retrained by this correction.

### What-If Simulation

`POST /simulate` runs 48 Bergman steps at 5-minute intervals for a submitted scenario and a baseline. It adjusts scenario `p3` using exercise duration and sleep hours, applies a stress multiplier to the simulated glucose trajectory, and supplies a meal-bolus trace from the requested carbohydrate amount. The response includes both trajectories and derived peak glucose, percentage of simulated points above 180 mg/dL, time-in-range percentage, a recovery-time estimate, adjusted `p3` index, and heuristic advice. These are outputs of the coded scenario assumptions, not guaranteed physiological outcomes.

### Metrics

`ai/evaluation_metrics.py:compute_all_metrics()` implements MAE, RMSE, R-squared, MARD, TIR classification accuracy, glucose-direction accuracy, weighted precision/recall/F1, hypoglycemia/hyperglycemia threshold detection, and simplified relative-error bands labeled Clarke A/B. The bands are relative-error calculations in this implementation, not a full clinical Clarke Error Grid analysis.

The backend can match recent forecast records against subsequent glucose logs and persist retrospective evaluation metrics when enough matched samples are available. The `/metrics/{patient_id}` response also reports twin-derived measures including TIR, volatility, stress, recovery, and reversal score. No accuracy values are asserted here.

## 5. Technology Stack

- Python, FastAPI, and Uvicorn for the API.
- PyTorch for `BioMirrorPINN`; the live API can run without it but cannot perform PINN inference.
- NumPy, pandas, SciPy, and scikit-learn for feature processing, scaling, and evaluation.
- SQLAlchemy's asynchronous ORM and aiosqlite for the active local database configuration.
- HTML, CSS, and JavaScript in a static single-page frontend; Chart.js is loaded from a CDN.
- JWT (`python-jose`) and bcrypt/passlib for authentication.
- Bergman glucose-insulin equations implemented in Python for the Digital Twin and scenario simulation.

The current local runtime uses SQLite. Although deployment/configuration files exist, this README does not describe Docker, PostgreSQL, or another deployment as the active runtime.

## 6. Project Structure

```text
BioMirror/
├── README.md
├── .gitignore
├── biomirror.db                    # local SQLite database (runtime data)
├── backend/
│   ├── backend_api.py               # FastAPI routes, twin, forecast, simulation
│   ├── database.py                  # SQLAlchemy models and persistence
│   ├── ai_bridge.py                 # live features, PINN loading/inference, gate
│   ├── food_database.py             # nutrition/food calculations
│   └── test_log_health_e2e.py       # backend end-to-end test
├── ai/
│   ├── pinn_model.py                # BioMirrorPINN, Bergman ODE, loss, training
│   ├── feature_engineering.py       # preprocessing and derived features
│   ├── evaluation_metrics.py        # evaluation metrics and attribution utilities
│   ├── reversal_engine.py           # reversal/recommendation analysis
│   ├── federated_learning.py        # separate research module
│   ├── optuna_hypertuning.py        # separate tuning module
│   ├── train.py                     # research training pipeline
│   ├── train_checkpoint.py          # checkpoint training/export pipeline
│   ├── demo.py                      # research/demo entry point
│   ├── tests.py                     # AI module tests
│   └── biomirror_outputs/
│       ├── training_log.json
│       ├── eval/evaluation_report.json
│       └── tuning/optuna_results.json
├── data/
│   └── diabetes_lifestyle_dataset_500.csv
├── demo_eval/
│   └── evaluation_report.json
├── deployment/
│   ├── requirements.txt
│   ├── .env.example
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── nginx.conf
│   └── setup_windows.bat
├── frontend/
│   ├── biomirror_app.html          # static HTML/CSS/JavaScript application
│   └── package-lock.json           # present; not needed to run the static page
└── models/
    └── biomirror_pinn.pt           # checkpoint loaded by the live AI bridge
```

## 7. API / Backend

Authenticated routes require a bearer access token unless otherwise handled by the route. The full generated API reference is available at `/docs` while the backend is running.

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/auth/register` | Register a user and create access/refresh tokens. |
| `POST` | `/auth/login` | Authenticate and issue tokens. |
| `POST` | `/auth/refresh` | Exchange a valid refresh token. |
| `POST` | `/auth/logout` | Revoke the user's refresh tokens. |
| `GET`, `PUT` | `/profile/{patient_id}` | Read or update the authenticated user's profile. |
| `POST` | `/logs/{patient_id}/glucose` | Record observed glucose and update the twin. |
| `POST` | `/logs/{patient_id}/meal` | Record a meal. |
| `POST` | `/logs/{patient_id}/activity` | Record activity. |
| `POST` | `/logs/{patient_id}/sleep` | Record sleep. |
| `POST` | `/logs/{patient_id}/stress` | Record stress. |
| `POST` | `/logs/{patient_id}/medication` | Record medication. |
| `POST` | `/logs/{patient_id}/weight` | Record weight. |
| `POST` | `/logs/{patient_id}/bp` | Record blood pressure. |
| `POST` | `/logs/{patient_id}/water` | Record water intake. |
| `POST` | `/logs/{patient_id}/hba1c` | Record HbA1c. |
| `GET` | `/twin/{patient_id}` | Read twin state and computed metrics. |
| `GET` | `/twin/{patient_id}/memory` | Read twin snapshots and prediction/simulation history. |
| `GET` | `/forecast/{patient_id}` | Return Bergman forecast and PINN-or-fallback near-term prediction. |
| `GET` | `/metrics/{patient_id}` | Return current and retrospective metrics where available. |
| `POST` | `/simulate` | Run a What-If scenario and return baseline/simulated trajectories. |
| `GET` | `/simulate/history/{patient_id}` | Read stored simulation/prediction history. |
| `GET` | `/history/{patient_id}` | Read the user's health-log history. |
| `GET` | `/history/{patient_id}/glucose` | Read glucose history. |
| `POST` | `/coach/ask` | Submit a coach question; requires a responding configured external provider. |
| `GET`, `DELETE` | `/coach/history/{patient_id}` | Read or clear coach conversation history. |
| `GET` | `/analytics/{patient_id}` | Read analytics summary. |
| `GET` | `/health` | Report service, database, and AI-module status. |
| `GET` | `/` | Return service name, version, and docs path. |
| `WS` | `/ws/{patient_id}` | Receive Digital Twin updates. |

## 8. Database

The audited local runtime is connected to SQLite at `biomirror.db` in the project root. `backend/database.py` selects this file when `DATABASE_URL` is unset. SQLAlchemy models include users and tokens, settings, twin state and snapshots, glucose/meal/activity/sleep/medication/weight/blood-pressure/water/HbA1c/stress logs, chat messages, goals, notifications, prediction history, reversal-score history, recommendation history, evaluation metrics, and per-user personal features.

## 9. Running the Project

Run these commands from the repository root in Windows PowerShell. Python 3.10-3.13 is listed as supported in `deployment/requirements.txt`.

1. Create and activate the virtual environment:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the project dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r deployment/requirements.txt
   ```

   PyTorch is optional for the API, but required to load and run the PINN checkpoint. For a CPU-only installation:

   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. Start the FastAPI backend from the project root:

   ```powershell
   python -m uvicorn backend.backend_api:app --host 127.0.0.1 --port 8000 --reload
   ```

4. Open `frontend/biomirror_app.html` directly in a browser. The page is static and calls the API at `http://127.0.0.1:8000`; no frontend build step is required.

5. Open the API documentation at `http://127.0.0.1:8000/docs`. The health endpoint is `http://127.0.0.1:8000/health`.

When `DATABASE_URL` is not configured, the backend uses the project-root SQLite database. Set `GROQ_API_KEY` or `ANTHROPIC_API_KEY` in the backend process environment to enable the corresponding AI Coach provider. The optional `PINN_MODEL_PATH` setting selects a checkpoint path; by default the bridge looks for `models/biomirror_pinn.pt`.

## 10. Model and Prediction Limitations

- The PINN is implemented and its checkpoint loads in the audited runtime, but live outputs are subject to a plausibility gate. Observed forecast requests in the current runtime were rejected by that gate and used the Bergman fallback.
- The Bergman fallback is a coded mathematical simulation, not a glucose measurement. Long-horizon forecasts include fixed meal assumptions at specified hours rather than a personalized future meal schedule.
- PINN and Digital Twin outputs are computational estimates and are not clinical measurements, diagnoses, or treatment advice.
- What-If outputs reflect the scenario equations and heuristic adjustments in the implementation; they are not guaranteed physiological outcomes.
- Logged actual glucose readings provide observations for state updates, residual matching, and retrospective evaluation. They are useful for checking forecasts and enabling the implemented per-user correction, but do not establish clinical validity.
- The model checkpoint's reported evaluation metadata is not reproduced here as an accuracy claim. This README makes no claim of clinical validation or predictive performance.
- The AI Coach depends on an external provider and API key. The endpoint returns an unavailable response if no provider returns a response; it does not have an offline rule-based fallback.

## 11. Development Status

**Implemented:** FastAPI backend and authentication; health logging; SQLite persistence; Bergman Digital Twin; feature engineering; PINN architecture, training pipeline, and checkpoint loading; PINN output plausibility gate; Bergman forecast fallback; per-user residual correction; What-If simulation; analytics and evaluation utilities; static dashboard; and a provider-backed AI Coach endpoint.

**Active in the audited runtime:** SQLite persistence is connected; the PINN checkpoint is loaded; observed forecast calls fail the implemented plausibility gate, so their near-term prediction path uses Bergman values; the static frontend calls the local API.

**Improvement areas, not completed capabilities:** improve and independently evaluate PINN predictions so that outputs can pass the implemented gate when appropriate; assess forecasts against additional observed data; and validate simulation assumptions. These are development needs, not claims of current functionality.

## 12. Academic/Project Description

BioMirror is a Physics-Informed Digital Twin framework for personalized Type-2 diabetes glucose modeling, combining a global PINN, Bergman physiological modeling, user-specific residual personalization, and What-If metabolic simulation.
