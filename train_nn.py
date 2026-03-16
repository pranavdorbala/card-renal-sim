#!/usr/bin/env python3
"""
Residual Neural Network: ARIC Visit 5 → Visit 7 Prediction
============================================================
Small residual MLP that learns V7 = V5 + f(V5) (disease progression delta).

Usage:
    python train_nn.py                          # train on cohort_data.npz
    python train_nn.py --data cohort_data.npz --epochs 200
    python train_nn.py --evaluate models/v5_to_v7_best.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ARIC_VARIABLES, NUMERIC_VAR_NAMES, N_FEATURES, NN_DEFAULTS


# ═══════════════════════════════════════════════════════════════════════════
# Model Architecture
# ═══════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Two FC layers with BatchNorm, ReLU, Dropout, and skip connection."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class V5toV7Net(nn.Module):
    """
    Residual MLP: V7 = skip(V5) + f(V5).
    The skip connection lets the network learn only the disease-progression delta.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(n_features)
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, n_features)
        self.skip = nn.Linear(n_features, n_features)  # learnable skip V5 → V7

    def forward(self, v5):
        h = torch.relu(self.input_proj(self.input_norm(v5)))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h) + self.skip(v5)


# ═══════════════════════════════════════════════════════════════════════════
# Loss Function
# ═══════════════════════════════════════════════════════════════════════════

class CompositeLoss(nn.Module):
    """
    Weighted MSE with direction-of-change penalty.

    Components:
    1. Per-variable MSE / training_std, weighted by clinical importance
    2. Direction penalty: extra cost when predicted delta sign differs from true delta
    """

    def __init__(self, var_weights: torch.Tensor, training_std: torch.Tensor,
                 direction_weight: float = 0.1):
        super().__init__()
        self.register_buffer('var_weights', var_weights)
        self.register_buffer('training_std', training_std.clamp(min=1e-6))
        self.direction_weight = direction_weight

    def forward(self, pred_v7, true_v7, v5):
        # Normalized MSE
        diff = (pred_v7 - true_v7) / self.training_std
        weighted_mse = (diff ** 2 * self.var_weights).mean()

        # Direction-of-change penalty
        true_delta = true_v7 - v5
        pred_delta = pred_v7 - v5
        direction_mismatch = (true_delta * pred_delta < 0).float()  # sign disagrees
        direction_penalty = (direction_mismatch * self.var_weights).mean()

        return weighted_mse + self.direction_weight * direction_penalty


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def load_data(data_path: str):
    """Load cohort_data.npz and split into train/val/test."""
    data = np.load(data_path, allow_pickle=True)
    v5 = data['v5'].astype(np.float32)
    v7 = data['v7'].astype(np.float32)
    var_names = list(data['var_names'])

    # Replace NaN/Inf with 0
    v5 = np.nan_to_num(v5, nan=0.0, posinf=0.0, neginf=0.0)
    v7 = np.nan_to_num(v7, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(v5)
    n_train = int(n * NN_DEFAULTS['train_frac'])
    n_val = int(n * NN_DEFAULTS['val_frac'])

    # Shuffle deterministically
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    v5, v7 = v5[idx], v7[idx]

    train_v5, train_v7 = v5[:n_train], v7[:n_train]
    val_v5, val_v7 = v5[n_train:n_train+n_val], v7[n_train:n_train+n_val]
    test_v5, test_v7 = v5[n_train+n_val:], v7[n_train+n_val:]

    # Compute training statistics for normalization in loss
    training_std = train_v7.std(axis=0)

    # Variable importance weights from config
    weights = np.array([
        ARIC_VARIABLES.get(vn, {}).get('weight', 0.5) for vn in var_names
    ], dtype=np.float32)

    return {
        'train': (train_v5, train_v7),
        'val': (val_v5, val_v7),
        'test': (test_v5, test_v7),
        'var_names': var_names,
        'training_std': training_std,
        'weights': weights,
    }


def make_loaders(data_dict, batch_size: int = 256):
    """Create DataLoaders from numpy arrays."""
    loaders = {}
    for split in ['train', 'val', 'test']:
        v5, v7 = data_dict[split]
        ds = TensorDataset(torch.from_numpy(v5), torch.from_numpy(v7))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == 'train'), drop_last=(split == 'train'),
        )
    return loaders


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def train(
    data_path: str = 'cohort_data.npz',
    hidden_dim: int = NN_DEFAULTS['hidden_dim'],
    n_blocks: int = NN_DEFAULTS['n_blocks'],
    dropout: float = NN_DEFAULTS['dropout'],
    lr: float = NN_DEFAULTS['lr'],
    weight_decay: float = NN_DEFAULTS['weight_decay'],
    epochs: int = NN_DEFAULTS['epochs'],
    batch_size: int = NN_DEFAULTS['batch_size'],
    patience: int = NN_DEFAULTS['patience'],
    save_dir: str = 'models',
):
    """Train V5→V7 residual network."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, data_path) if not os.path.isabs(data_path) else data_path
    data = load_data(data_path)
    loaders = make_loaders(data, batch_size)

    n_features = len(data['var_names'])
    print(f"Features: {n_features}, Train: {len(data['train'][0])}, "
          f"Val: {len(data['val'][0])}, Test: {len(data['test'][0])}")

    # Model
    model = V5toV7Net(n_features, hidden_dim, n_blocks, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Loss
    var_weights = torch.from_numpy(data['weights']).to(device)
    training_std = torch.from_numpy(data['training_std'].astype(np.float32)).to(device)
    criterion = CompositeLoss(var_weights, training_std).to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_loss = float('inf')
    epochs_no_improve = 0
    save_path = os.path.join(base, save_dir, 'v5_to_v7_best.pt')
    os.makedirs(os.path.join(base, save_dir), exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for v5_batch, v7_batch in loaders['train']:
            v5_batch, v7_batch = v5_batch.to(device), v7_batch.to(device)
            pred = model(v5_batch)
            loss = criterion(pred, v7_batch, v5_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(v5_batch)
        train_loss /= len(data['train'][0])

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v5_batch, v7_batch in loaders['val']:
                v5_batch, v7_batch = v5_batch.to(device), v7_batch.to(device)
                pred = model(v5_batch)
                loss = criterion(pred, v7_batch, v5_batch)
                val_loss += loss.item() * len(v5_batch)
        val_loss /= len(data['val'][0])

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'n_features': n_features,
                'hidden_dim': hidden_dim,
                'n_blocks': n_blocks,
                'dropout': dropout,
                'var_names': data['var_names'],
                'training_std': data['training_std'],
                'weights': data['weights'],
                'best_val_loss': best_val_loss,
                'epoch': epoch,
            }, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}, saved to {save_path}")

    # Evaluate on test set
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    evaluate_model(model, data, device)

    return save_path


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def load_trained_model(model_path: str, device=None):
    """Load a trained V5toV7Net from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = V5toV7Net(
        n_features=checkpoint['n_features'],
        hidden_dim=checkpoint['hidden_dim'],
        n_blocks=checkpoint['n_blocks'],
        dropout=checkpoint['dropout'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint


def predict(model, v5_array: np.ndarray, device=None) -> np.ndarray:
    """Run V5→V7 prediction. v5_array: (N, n_features) or (n_features,)."""
    if device is None:
        device = next(model.parameters()).device
    single = v5_array.ndim == 1
    if single:
        v5_array = v5_array[np.newaxis, :]
    with torch.no_grad():
        v5_t = torch.from_numpy(v5_array.astype(np.float32)).to(device)
        pred = model(v5_t).cpu().numpy()
    return pred[0] if single else pred


def evaluate_model(model, data_dict, device):
    """Compute per-variable metrics on test set."""
    model.eval()
    test_v5, test_v7 = data_dict['test']
    var_names = data_dict['var_names']

    pred_v7 = predict(model, test_v5, device)

    print(f"\n{'='*70}")
    print(f"  Test Set Evaluation ({len(test_v5)} patients, {len(var_names)} variables)")
    print(f"{'='*70}")

    # Per-variable R^2 and MAE
    r2_list, mae_list = [], []
    key_vars = ['LVEF_pct', 'MAP_mmHg', 'GFR_mL_min', 'eGFR_mL_min_173m2',
                'E_e_prime_avg', 'CO_Lmin', 'SBP_mmHg', 'NTproBNP_pg_mL',
                'serum_creatinine_mg_dL', 'GLS_pct', 'LVEDV_mL', 'LVESV_mL']

    for i, vn in enumerate(var_names):
        true = test_v7[:, i]
        pred = pred_v7[:, i]
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        mae = np.mean(np.abs(true - pred))
        r2_list.append(r2)
        mae_list.append(mae)
        if vn in key_vars:
            print(f"  {vn:35s}  R²={r2:.3f}  MAE={mae:.2f}")

    # Direction-of-change accuracy
    true_delta = test_v7 - test_v5
    pred_delta = pred_v7 - test_v5
    # Only count where true delta is non-trivial
    mask = np.abs(true_delta) > 1e-6
    if mask.any():
        correct_dir = ((true_delta * pred_delta) > 0) | (~mask)
        dir_acc = correct_dir[mask].mean()
        print(f"\n  Direction-of-change accuracy: {dir_acc:.1%}")

    # Summary
    r2_arr = np.array(r2_list)
    print(f"\n  Overall R² — mean: {r2_arr.mean():.3f}, median: {np.median(r2_arr):.3f}")
    print(f"  Variables with R² > 0.8: {(r2_arr > 0.8).sum()}/{len(r2_arr)}")
    print(f"  Variables with R² > 0.5: {(r2_arr > 0.5).sum()}/{len(r2_arr)}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train V5→V7 residual NN')
    parser.add_argument('--data', type=str, default='cohort_data.npz')
    parser.add_argument('--epochs', type=int, default=NN_DEFAULTS['epochs'])
    parser.add_argument('--hidden_dim', type=int, default=NN_DEFAULTS['hidden_dim'])
    parser.add_argument('--n_blocks', type=int, default=NN_DEFAULTS['n_blocks'])
    parser.add_argument('--dropout', type=float, default=NN_DEFAULTS['dropout'])
    parser.add_argument('--lr', type=float, default=NN_DEFAULTS['lr'])
    parser.add_argument('--batch_size', type=int, default=NN_DEFAULTS['batch_size'])
    parser.add_argument('--patience', type=int, default=NN_DEFAULTS['patience'])
    parser.add_argument('--evaluate', type=str, default=None,
                        help='Path to trained model for evaluation only')
    args = parser.parse_args()

    if args.evaluate:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, ckpt = load_trained_model(args.evaluate, device)
        data = load_data(args.data)
        evaluate_model(model, data, device)
    else:
        train(
            data_path=args.data,
            hidden_dim=args.hidden_dim,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )


if __name__ == '__main__':
    main()
