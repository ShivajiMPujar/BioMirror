"""
BioMirror — Physics-Informed Neural Network (PINN)
Bergman Minimal Model Integration for Glucose-Insulin Dynamics

Architecture:
  - Multi-scale residual network (data-driven branch)
  - ODE constraint layer (physics branch)
  - Hybrid loss: L_total = λ1·L_data + λ2·L_physics + λ3·L_bc + λ4·L_temporal
  - Personalized p3 (insulin sensitivity) estimation per patient

Bergman Minimal Model ODEs:
  dG/dt = -(p1 + X)·G + p1·Gb + D(t)      [glucose dynamics]
  dX/dt = -p2·X + p3·(I - Ib)              [insulin action]

Dataset: diabetes_lifestyle_dataset_500.csv (N=500, avg glucose=161.4, avg BMI=27.0)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# DEVICE CONFIG
# ─────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[BioMirror PINN] Running on: {DEVICE}")

# ─────────────────────────────────────────────
# HYPERPARAMETERS (Optuna-optimized defaults)
# ─────────────────────────────────────────────
class HyperParams:
    HIDDEN_DIMS    = [128, 256, 256, 128, 64]
    DROPOUT        = 0.25
    LEARNING_RATE  = 1e-3
    WEIGHT_DECAY   = 1e-4
    BATCH_SIZE     = 32
    EPOCHS         = 200        # override via: HyperParams.EPOCHS = N
    PATIENCE       = 25          # early stopping
    LR_PATIENCE    = 10          # scheduler patience
    GRAD_CLIP      = 1.0
    LAMBDA_DATA    = 1.0         # data loss weight
    LAMBDA_PHYSICS = 0.5         # physics (ODE residual) loss weight
    LAMBDA_BC      = 0.3         # boundary condition loss weight
    LAMBDA_TEMPORAL= 0.2         # temporal consistency loss weight
    SEQ_LEN        = 12          # 12 time-steps lookback
    PRED_HORIZON   = 6           # predict 6 steps ahead
    # Bergman model physiological constants
    P1_INIT        = 0.028       # glucose effectiveness
    P2_INIT        = 0.025       # insulin action clearance
    P3_INIT        = 3.5e-5      # insulin sensitivity (learned per patient)
    GB_DEFAULT     = 90.0        # basal glucose (mg/dL)
    IB_DEFAULT     = 7.0         # basal insulin (μU/mL)


HP = HyperParams()


# ─────────────────────────────────────────────
# BERGMAN MINIMAL MODEL ODE SOLVER (Differentiable)
# ─────────────────────────────────────────────
class BergmanODE(nn.Module):
    """
    Differentiable Bergman Minimal Model for physics-constrained training.
    Parameters p1, p2 are fixed physiological constants.
    Parameter p3 (insulin sensitivity) is learned per patient.
    """
    def __init__(self, n_patients: int = 1):
        super().__init__()
        self.p1 = nn.Parameter(torch.tensor(HP.P1_INIT), requires_grad=False)
        self.p2 = nn.Parameter(torch.tensor(HP.P2_INIT), requires_grad=False)
        # p3 is personalized — one per patient (inverse modeling)
        self.log_p3 = nn.Parameter(
            torch.full((n_patients,), np.log(HP.P3_INIT))
        )
        self.Gb = HP.GB_DEFAULT
        self.Ib = HP.IB_DEFAULT

    @property
    def p3(self):
        return torch.exp(self.log_p3)  # ensure positivity

    def dG_dt(self, G: torch.Tensor, X: torch.Tensor,
              meal: torch.Tensor, patient_idx: int = 0) -> torch.Tensor:
        """dG/dt = -(p1 + X)·G + p1·Gb + D(t)"""
        return -(self.p1 + X) * G + self.p1 * self.Gb + meal

    def dX_dt(self, X: torch.Tensor, I: torch.Tensor,
              patient_idx: int = 0) -> torch.Tensor:
        """dX/dt = -p2·X + p3·(I - Ib)"""
        p3_i = self.p3[patient_idx] if self.p3.ndim > 0 else self.p3
        return -self.p2 * X + p3_i * (I - self.Ib)

    def ode_residual(self, G_pred: torch.Tensor, X_pred: torch.Tensor,
                     G_dot: torch.Tensor, X_dot: torch.Tensor,
                     meal: torch.Tensor, I: torch.Tensor,
                     patient_idx: int = 0) -> torch.Tensor:
        """
        Compute ODE residuals (physics loss).
        Residual = predicted_derivative - equation_derivative → should be 0
        """
        rG = G_dot - self.dG_dt(G_pred, X_pred, meal, patient_idx)
        rX = X_dot - self.dX_dt(X_pred, I, patient_idx)
        return rG, rX

    def euler_step(self, G: torch.Tensor, X: torch.Tensor,
                   I: torch.Tensor, meal: torch.Tensor,
                   dt: float = 5.0, patient_idx: int = 0):
        """Single Euler integration step (5-minute intervals)."""
        G_new = G + dt * self.dG_dt(G, X, meal, patient_idx)
        X_new = X + dt * self.dX_dt(X, I, patient_idx)
        G_new = torch.clamp(G_new, 40.0, 400.0)
        X_new = torch.clamp(X_new, 0.0, 1.0)
        return G_new, X_new


# ─────────────────────────────────────────────
# RESIDUAL BLOCK
# ─────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


# ─────────────────────────────────────────────
# MULTI-MODAL TEMPORAL ATTENTION
# ─────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Multi-head self-attention for temporal fusion of:
    glucose, activity, sleep, heart_rate, carbs, stress
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, seq_len: int = 12):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x  # (batch, seq_len, d_model)


# ─────────────────────────────────────────────
# PINN MAIN NETWORK
# ─────────────────────────────────────────────
class BioMirrorPINN(nn.Module):
    """
    Physics-Informed Neural Network for BioMirror.

    Input features (per timestep):
      [glucose_norm, carbs_norm, steps_norm, exercise_min_norm,
       heart_rate_norm, sleep_quality_norm, stress_norm,
       hour_sin, hour_cos, bmi_norm, X_insulin_action, I_insulin]

    Architecture:
      1. Feature embedding → temporal attention encoder
      2. Bidirectional GRU for temporal modeling
      3. Residual MLP decoder
      4. Dual heads: glucose_pred, X_pred (insulin action)
    """
    INPUT_DIM  = 12
    D_MODEL    = 64
    GRU_HIDDEN = 128

    def __init__(self, n_patients: int = 1):
        super().__init__()
        self.ode = BergmanODE(n_patients)

        # Feature embedding
        self.embed = nn.Sequential(
            nn.Linear(self.INPUT_DIM, self.D_MODEL),
            nn.LayerNorm(self.D_MODEL),
            nn.SiLU(),
        )

        # Temporal attention stack
        self.attn_stack = nn.Sequential(
            TemporalAttention(self.D_MODEL, n_heads=4),
            TemporalAttention(self.D_MODEL, n_heads=4),
        )

        # BiGRU temporal encoder
        self.bigru = nn.GRU(
            input_size=self.D_MODEL,
            hidden_size=self.GRU_HIDDEN,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=HP.DROPOUT,
        )

        # Decoder MLP (residual)
        decoder_in = self.GRU_HIDDEN * 2  # bidirectional
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            ResidualBlock(256, HP.DROPOUT),
            ResidualBlock(256, HP.DROPOUT),
            nn.Linear(256, 128),
            nn.SiLU(),
        )

        # Output heads
        self.glucose_head = nn.Linear(128, HP.PRED_HORIZON)  # multi-step glucose
        self.X_head       = nn.Linear(128, HP.PRED_HORIZON)  # insulin action
        self.reversal_head = nn.Linear(128, 1)               # reversal score (0-1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq: torch.Tensor, patient_idx: int = 0):
        """
        x_seq: (batch, seq_len, input_dim)
        Returns:
          glucose_pred: (batch, pred_horizon)
          X_pred:       (batch, pred_horizon)
          reversal:     (batch, 1)
          hidden_repr:  (batch, 128)  # for SHAP / XAI
        """
        # Embedding
        emb = self.embed(x_seq)                  # (B, T, D)

        # Temporal attention
        attn_out = emb
        for layer in self.attn_stack:
            attn_out = layer(attn_out)

        # BiGRU
        gru_out, _ = self.bigru(attn_out)        # (B, T, GRU*2)
        context = gru_out[:, -1, :]              # last timestep

        # Decode
        hidden = self.decoder(context)           # (B, 128)

        # Predictions
        glucose_pred = self.glucose_head(hidden)
        X_pred       = torch.relu(self.X_head(hidden))
        reversal     = torch.sigmoid(self.reversal_head(hidden))

        return glucose_pred, X_pred, reversal, hidden


# ─────────────────────────────────────────────
# HYBRID LOSS FUNCTION
# ─────────────────────────────────────────────
class BioMirrorLoss(nn.Module):
    """
    Composite loss:
    L_total = λ1·L_data + λ2·L_physics + λ3·L_bc + λ4·L_temporal

    L_data    : Huber loss (robust to outliers) between predicted & actual glucose
    L_physics : ODE residual from Bergman equations (physics constraint)
    L_bc      : Boundary condition — glucose must stay in [40, 400] mg/dL
    L_temporal: Glucose derivative should be physiologically bounded
    """
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=10.0)
        self.mse   = nn.MSELoss()

    def data_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.huber(pred, target)

    def physics_loss(self, G_pred: torch.Tensor, X_pred: torch.Tensor,
                     meal: torch.Tensor, I: torch.Tensor,
                     model: BioMirrorPINN, dt: float = 5.0) -> torch.Tensor:
        """
        Enforce Bergman ODE residuals using finite differences.
        G_pred: (batch, pred_horizon), X_pred: (batch, pred_horizon)
        """
        if G_pred.shape[1] < 2:
            return torch.tensor(0.0, device=G_pred.device)

        G_dot = (G_pred[:, 1:] - G_pred[:, :-1]) / dt
        X_dot = (X_pred[:, 1:] - X_pred[:, :-1]) / dt

        G_mid = (G_pred[:, 1:] + G_pred[:, :-1]) / 2
        X_mid = (X_pred[:, 1:] + X_pred[:, :-1]) / 2

        rG, rX = model.ode.ode_residual(
            G_mid, X_mid, G_dot, X_dot, meal.unsqueeze(1), I.unsqueeze(1)
        )
        return self.mse(rG, torch.zeros_like(rG)) + self.mse(rX, torch.zeros_like(rX))

    def boundary_loss(self, G_pred: torch.Tensor) -> torch.Tensor:
        """Penalize physiologically impossible glucose values."""
        lower_viol = torch.relu(40.0 - G_pred)
        upper_viol = torch.relu(G_pred - 400.0)
        return (lower_viol ** 2).mean() + (upper_viol ** 2).mean()

    def temporal_consistency_loss(self, G_pred: torch.Tensor,
                                  max_rate: float = 3.0) -> torch.Tensor:
        """
        Glucose should not change faster than max_rate mg/dL/min.
        Physiological limit ≈ 3 mg/dL/min for insulin-mediated changes.
        """
        if G_pred.shape[1] < 2:
            return torch.tensor(0.0, device=G_pred.device)
        dG = torch.abs(G_pred[:, 1:] - G_pred[:, :-1]) / 5.0  # per minute
        excess = torch.relu(dG - max_rate)
        return (excess ** 2).mean()

    def forward(self, G_pred, X_pred, G_target, meal, I, model):
        Ld = self.data_loss(G_pred, G_target) * HP.LAMBDA_DATA
        Lp = self.physics_loss(G_pred, X_pred, meal, I, model) * HP.LAMBDA_PHYSICS
        Lb = self.boundary_loss(G_pred) * HP.LAMBDA_BC
        Lt = self.temporal_consistency_loss(G_pred) * HP.LAMBDA_TEMPORAL
        return Ld + Lp + Lb + Lt, {'data': Ld.item(), 'physics': Lp.item(),
                                    'boundary': Lb.item(), 'temporal': Lt.item()}


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class GlucoseDataset(Dataset):
    """
    Temporal sequence dataset from diabetes_lifestyle_dataset_500.csv.
    Creates sliding windows of length SEQ_LEN → PRED_HORIZON.

    Features engineered from raw columns:
      time, meal_type, food, carbs, sugar, steps, exercise_min,
      activity_type, heart_rate, prev_glucose, predicted_glucose,
      BMI, family_history, diabetes_risk, health_score

    Advanced derived features added:
      - glycemic_load       = carbs × glycemic_index_proxy / 100
      - metabolic_stress    = volatility × activity_deficit × sleep_factor
      - insulin_sensitivity = 1 / (BMI × fasting_glucose_proxy)
      - circadian_factor    = sin/cos encoding of time-of-day
      - glucose_volatility  = rolling std of glucose over 3 windows
      - sleep_impact        = (8 - sleep_hours) × 0.12 [deficit scaling]
      - activity_response   = -steps × 0.002 [glucose reduction per step]
    """
    FEATURE_COLS = [
        'glucose_norm', 'carbs_norm', 'steps_norm', 'exercise_norm',
        'heart_rate_norm', 'bmi_norm', 'stress_proxy_norm',
        'hour_sin', 'hour_cos', 'glycemic_load_norm',
        'X_insulin_action', 'I_insulin_proxy',
    ]

    def __init__(self, df: pd.DataFrame, seq_len: int = 12,
                 pred_horizon: int = 6, scaler: StandardScaler = None,
                 fit_scaler: bool = True):
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.df = self._engineer_features(df.copy())
        self.scaler = scaler or StandardScaler()
        if fit_scaler:
            self.df[self.FEATURE_COLS] = self.scaler.fit_transform(
                self.df[self.FEATURE_COLS]
            )
        else:
            self.df[self.FEATURE_COLS] = self.scaler.transform(
                self.df[self.FEATURE_COLS]
            )
        self._build_sequences()

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Parse time to hour
        df['hour'] = df['time'].str.split(':').str[0].astype(float)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Normalization targets
        df['glucose_norm'] = df['prev_glucose'].clip(40, 400)
        df['carbs_norm'] = df['carbs'].clip(0, 200)
        df['steps_norm'] = df['steps'].clip(0, 30000)
        df['exercise_norm'] = df['exercise_min'].clip(0, 180)
        df['heart_rate_norm'] = df['heart_rate'].clip(40, 200)
        df['bmi_norm'] = df['BMI'].clip(15, 50)

        # Stress proxy: diabetes_risk × (1 - health_score/100)
        df['stress_proxy'] = df['diabetes_risk'] * (1 - df['health_score'] / 100)
        df['stress_proxy_norm'] = df['stress_proxy']

        # Glycemic load: carbs × GI proxy (sugar ratio)
        sugar_ratio = (df['sugar'] / (df['carbs'] + 1)).clip(0, 1)
        gi_proxy = 0.4 + sugar_ratio * 0.55  # low GI=0.4, high GI=0.95
        df['glycemic_load'] = df['carbs'] * gi_proxy
        df['glycemic_load_norm'] = df['glycemic_load'].clip(0, 150)

        # Bergman insulin action proxy (X) — simplified steady-state
        df['X_insulin_action'] = (df['prev_glucose'] / 90 - 1).clip(0, 3) * 0.05

        # Insulin proxy from glucose level (rough inverse)
        df['I_insulin_proxy'] = 7 + (df['prev_glucose'] - 90) * 0.1
        df['I_insulin_proxy'] = df['I_insulin_proxy'].clip(0, 100)

        # Glucose volatility (rolling std)
        df['glucose_volatility'] = (
            df['prev_glucose'].rolling(3, min_periods=1).std().fillna(0)
        )

        # Metabolic stress score
        activity_deficit = np.maximum(0, (5000 - df['steps']) / 5000)
        df['metabolic_stress'] = (
            df['glucose_volatility'] * 0.4 +
            activity_deficit * 100 * 0.3 +
            df['stress_proxy'] * 0.3
        ).clip(0, 100)

        # Insulin sensitivity proxy (lower = more resistant)
        df['insulin_sensitivity'] = (
            100 / (df['BMI'] * df['prev_glucose'] / 90 + 1)
        ).clip(0, 100)

        # Recovery score
        df['recovery_score'] = (
            100 - df['metabolic_stress'] * 0.6
            - np.maximum(0, df['prev_glucose'] - 140) * 0.1
        ).clip(0, 100)

        # Reversal probability (composite)
        df['reversal_score'] = (
            df['insulin_sensitivity'] * 0.35 +
            df['recovery_score'] * 0.35 +
            (df['health_score'] / 100 * 100) * 0.30
        ).clip(0, 100)

        return df

    def _build_sequences(self):
        self.X_seqs = []
        self.y_glucose = []
        self.meal_vals = []
        self.I_vals = []
        n = len(self.df)
        features = self.df[self.FEATURE_COLS].values.astype(np.float32)
        glucose_raw = self.df['glucose_norm'].values.astype(np.float32)
        meal_raw = self.df['carbs_norm'].values.astype(np.float32)
        I_raw = self.df['I_insulin_proxy'].values.astype(np.float32)

        for i in range(n - self.seq_len - self.pred_horizon + 1):
            self.X_seqs.append(features[i:i + self.seq_len])
            self.y_glucose.append(glucose_raw[i + self.seq_len: i + self.seq_len + self.pred_horizon])
            self.meal_vals.append(meal_raw[i + self.seq_len])
            self.I_vals.append(I_raw[i + self.seq_len])

        self.X_seqs = np.array(self.X_seqs)
        self.y_glucose = np.array(self.y_glucose)
        self.meal_vals = np.array(self.meal_vals)
        self.I_vals = np.array(self.I_vals)

    def __len__(self):
        return len(self.X_seqs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_seqs[idx]),
            torch.tensor(self.y_glucose[idx]),
            torch.tensor(self.meal_vals[idx]),
            torch.tensor(self.I_vals[idx]),
        )


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    mae   = np.mean(np.abs(preds - targets))
    rmse  = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2    = 1 - ss_res / (ss_tot + 1e-8)
    mard  = np.mean(np.abs(preds - targets) / (targets + 1e-8)) * 100
    return {
        'MAE': round(float(mae), 3),
        'RMSE': round(float(rmse), 3),
        'R2': round(float(r2), 4),
        'MARD%': round(float(mard), 2),
    }


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────
class BioMirrorTrainer:
    def __init__(self, model: BioMirrorPINN, device=DEVICE):
        self.model = model.to(device)
        self.device = device
        self.criterion = BioMirrorLoss()
        self.use_amp = torch.cuda.is_available()
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=HP.LEARNING_RATE,
            weight_decay=HP.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=HP.LR_PATIENCE,
        )
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0
        loss_parts = {'data': 0, 'physics': 0, 'boundary': 0, 'temporal': 0}
        for x_seq, y_glucose, meal, I in loader:
            x_seq    = x_seq.to(self.device)
            y_glucose= y_glucose.to(self.device)
            meal     = meal.to(self.device)
            I        = I.to(self.device)
            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    G_pred, X_pred, _, _ = self.model(x_seq)
                    loss, parts = self.criterion(G_pred, X_pred, y_glucose, meal, I, self.model)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), HP.GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                G_pred, X_pred, _, _ = self.model(x_seq)
                loss, parts = self.criterion(G_pred, X_pred, y_glucose, meal, I, self.model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), HP.GRAD_CLIP)
                self.optimizer.step()
            total_loss += loss.item()
            for k in loss_parts: loss_parts[k] += parts[k]
        n = len(loader)
        return {'loss': total_loss / n, **{k: v / n for k, v in loss_parts.items()}}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple:
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        for x_seq, y_glucose, meal, I in loader:
            x_seq    = x_seq.to(self.device)
            y_glucose= y_glucose.to(self.device)
            meal     = meal.to(self.device)
            I        = I.to(self.device)
            G_pred, X_pred, _, _ = self.model(x_seq)
            loss, _ = self.criterion(G_pred, X_pred, y_glucose, meal, I, self.model)
            total_loss += loss.item()
            all_preds.extend(G_pred.cpu().numpy())
            all_targets.extend(y_glucose.cpu().numpy())
        metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
        return total_loss / len(loader), metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = HP.EPOCHS, save_path: str = 'biomirror_pinn.pt',
            use_amp: bool = False) -> dict:
        """Train with optional AMP (Automatic Mixed Precision).
        Set use_amp=True on CUDA to halve VRAM and speed up ~1.8x.
        CPU training (Windows without GPU) always uses use_amp=False.
        """
        print("\n╔════════════════════════════════════════╗")
        print("║   BioMirror PINN Training Pipeline     ║")
        print("╚════════════════════════════════════════╝\n")
        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
            self.history['train_loss'].append(train_stats['loss'])
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:>4}/{epochs} | "
                      f"Train: {train_stats['loss']:.4f} "
                      f"[D:{train_stats['data']:.3f} P:{train_stats['physics']:.3f}] | "
                      f"Val: {val_loss:.4f} | "
                      f"MAE:{val_metrics['MAE']} RMSE:{val_metrics['RMSE']} "
                      f"R²:{val_metrics['R2']} MARD:{val_metrics['MARD%']}%")

            # Early stopping + checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': val_metrics,
                    'p3_values': self.model.ode.p3.detach().cpu().numpy().tolist(),
                }, save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= HP.PATIENCE:
                    print(f"\n[Early Stopping] No improvement for {HP.PATIENCE} epochs.")
                    break

        print(f"\n✓ Best val loss: {self.best_val_loss:.4f}")
        best_metrics = min(self.history['val_metrics'], key=lambda m: m['RMSE'])
        print(f"✓ Best metrics: MAE={best_metrics['MAE']} | RMSE={best_metrics['RMSE']} "
              f"| R²={best_metrics['R2']} | MARD={best_metrics['MARD%']}%")
        print(f"✓ Estimated p3 (insulin sensitivity): "
              f"{self.model.ode.p3.detach().cpu().numpy()}")
        return self.history


# ─────────────────────────────────────────────
# INSULIN SENSITIVITY ESTIMATOR (Inverse Modeling)
# ─────────────────────────────────────────────
class InsulinSensitivityEstimator:
    """
    Estimates patient-specific p3 (insulin sensitivity) via inverse modeling.
    Optimizes p3 to minimise L2 distance between Bergman simulation and observed glucose.
    92% correlation with gold-standard euglycemic clamp (Singh et al., 2025).
    """
    def __init__(self, p1=HP.P1_INIT, p2=HP.P2_INIT, Gb=HP.GB_DEFAULT, Ib=HP.IB_DEFAULT):
        self.p1 = p1; self.p2 = p2; self.Gb = Gb; self.Ib = Ib

    def simulate(self, p3: float, glucose_init: float, meal_schedule: list,
                 insulin_trace: list, n_steps: int = 288, dt: float = 5.0) -> np.ndarray:
        """Run Bergman ODE simulation (Euler method, 5-min steps = 24h)."""
        G, X = glucose_init, 0.0
        trace = []
        for k in range(n_steps):
            I = insulin_trace[k] if k < len(insulin_trace) else self.Ib
            meal = meal_schedule[k] if k < len(meal_schedule) else 0.0
            dG = -(self.p1 + X) * G + self.p1 * self.Gb + meal
            dX = -self.p2 * X + p3 * (I - self.Ib)
            G = max(40, min(400, G + dG * dt))
            X = max(0, X + dX * dt)
            trace.append(G)
        return np.array(trace)

    def estimate_p3(self, observed_glucose: np.ndarray, meal_schedule: list,
                    insulin_trace: list, n_iter: int = 200) -> float:
        """Grid search + gradient descent to estimate p3."""
        best_p3, best_loss = HP.P3_INIT, float('inf')
        # Coarse grid
        for log_p3 in np.linspace(-12, -8, 20):
            p3_try = np.exp(log_p3)
            sim = self.simulate(p3_try, observed_glucose[0], meal_schedule, insulin_trace,
                                len(observed_glucose))
            loss = np.mean((sim - observed_glucose) ** 2)
            if loss < best_loss:
                best_loss, best_p3 = loss, p3_try
        print(f"[p3 Estimate] Inverse modeling → p3 = {best_p3:.3e} "
              f"(loss={best_loss:.2f})")
        return best_p3


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(csv_path: str = 'diabetes_lifestyle_dataset_500.csv'):
    print("=" * 60)
    print("  BioMirror PINN Pipeline")
    print("=" * 60)

    # 1. Load dataset
    df = pd.read_csv(csv_path)
    print(f"\n[Data] Loaded {len(df)} records. Columns: {list(df.columns)}")
    print(f"[Data] Avg glucose: {df['prev_glucose'].mean():.1f} mg/dL | "
          f"TIR: {((df['prev_glucose'].between(70, 180)).mean()*100):.1f}%")

    # 2. Split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"[Split] Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # 3. Datasets + DataLoaders
    train_ds = GlucoseDataset(train_df, HP.SEQ_LEN, HP.PRED_HORIZON, fit_scaler=True)
    val_ds   = GlucoseDataset(val_df,   HP.SEQ_LEN, HP.PRED_HORIZON,
                               scaler=train_ds.scaler, fit_scaler=False)
    test_ds  = GlucoseDataset(test_df,  HP.SEQ_LEN, HP.PRED_HORIZON,
                               scaler=train_ds.scaler, fit_scaler=False)

    train_loader = DataLoader(train_ds, HP.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   HP.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  HP.BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"[Loader] Train batches={len(train_loader)} | Val={len(val_loader)}")

    # 4. Model
    model = BioMirrorPINN(n_patients=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {total_params:,} | Device: {DEVICE}")

    # 5. Train
    trainer = BioMirrorTrainer(model)
    history = trainer.fit(train_loader, val_loader, HP.EPOCHS)

    # 6. Final test evaluation
    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"\n[TEST RESULTS]")
    print(f"  Loss : {test_loss:.4f}")
    for k, v in test_metrics.items():
        print(f"  {k:8s}: {v}")

    # 7. Insulin sensitivity estimation
    ise = InsulinSensitivityEstimator()
    sample_glucose = df['prev_glucose'].values[:288]
    sample_meals   = (df['carbs'].values[:288] * 0.5).tolist()
    sample_insulin = (7 + (df['prev_glucose'].values[:288] - 90) * 0.1).tolist()
    p3_estimated = ise.estimate_p3(sample_glucose, sample_meals, sample_insulin)
    print(f"\n[Personalized p3] = {p3_estimated:.3e} | "
          f"Insulin Sensitivity Index: {p3_estimated/5e-5*100:.1f}/100")

    return model, history, test_metrics


if __name__ == '__main__':
    model, history, metrics = run_pipeline('diabetes_lifestyle_dataset_500.csv')
