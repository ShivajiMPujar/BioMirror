"""
BioMirror — Production Checkpoint Exporter
============================================
Trains BioMirrorPINN on the shipped dataset and saves a checkpoint that
backend/ai_bridge.py can load directly for live inference.

This is distinct from ai/train.py (the full research/evaluation CLI with
plots, SHAP, federated learning, etc.) — this script has one job: produce
`models/biomirror_pinn.pt` with everything the live API needs to run
inference correctly, most importantly the FITTED StandardScaler, which
BioMirrorTrainer.fit() does NOT persist on its own (it only saves
model_state/optimizer_state/metrics) — without the exact scaler used at
training time, live inputs could not be normalized consistently and
predictions would be meaningless.

Usage:
    python ai/train_checkpoint.py                  # default 40 epochs
    python ai/train_checkpoint.py --epochs 100      # longer training
"""
import argparse, os, sys, time, pickle, base64
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_model import (
    BioMirrorPINN, BioMirrorTrainer, GlucoseDataset, HP, DEVICE
)
from torch.utils.data import DataLoader
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data',
        'diabetes_lifestyle_dataset_500.csv'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'models',
        'biomirror_pinn.pt'))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[1/5] Loading dataset: {args.csv}")
    df = pd.read_csv(args.csv)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"      Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    print("[2/5] Building sequences + fitting scaler")
    train_ds = GlucoseDataset(train_df, HP.SEQ_LEN, HP.PRED_HORIZON, fit_scaler=True)
    val_ds   = GlucoseDataset(val_df, HP.SEQ_LEN, HP.PRED_HORIZON,
                               scaler=train_ds.scaler, fit_scaler=False)
    test_ds  = GlucoseDataset(test_df, HP.SEQ_LEN, HP.PRED_HORIZON,
                               scaler=train_ds.scaler, fit_scaler=False)
    train_loader = DataLoader(train_ds, HP.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, HP.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, HP.BATCH_SIZE, shuffle=False)

    print(f"[3/5] Training BioMirrorPINN for up to {args.epochs} epochs on {DEVICE}")
    model = BioMirrorPINN(n_patients=1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Parameters: {n_params:,}")
    trainer = BioMirrorTrainer(model)
    t0 = time.time()
    trainer.fit(train_loader, val_loader, epochs=args.epochs, save_path=args.out)
    train_secs = time.time() - t0

    print("[4/5] Final test-set evaluation")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"      Test loss={test_loss:.4f}  {test_metrics}")

    print("[5/5] Re-saving checkpoint WITH scaler + metadata")
    ckpt = torch.load(args.out, map_location='cpu', weights_only=False)
    ckpt['scaler_pickle'] = base64.b64encode(pickle.dumps(train_ds.scaler)).decode('ascii')
    ckpt['feature_cols'] = GlucoseDataset.FEATURE_COLS
    ckpt['seq_len'] = HP.SEQ_LEN
    ckpt['pred_horizon'] = HP.PRED_HORIZON
    ckpt['test_metrics'] = test_metrics
    ckpt['trained_at'] = pd.Timestamp.utcnow().isoformat()
    ckpt['training_seconds'] = round(train_secs, 1)
    ckpt['dataset_rows'] = len(df)
    ckpt['model_version'] = f"pinn-v1-{pd.Timestamp.utcnow().strftime('%Y%m%d')}"
    torch.save(ckpt, args.out)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"      Saved {args.out} ({size_kb:.0f} KB)")
    print("\nDone. backend/ai_bridge.py will auto-load this checkpoint on next server start.")


if __name__ == '__main__':
    main()
