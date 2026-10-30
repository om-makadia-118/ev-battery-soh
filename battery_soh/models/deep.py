"""Small reproducible PyTorch regressors; PyTorch is optional for CPU-only ML runs."""
from __future__ import annotations
import copy
import random
import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:  # Classical ML should work without installing PyTorch locally.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

DEEP_NAMES = ('lstm', 'cnn', 'transformer')

if TORCH_AVAILABLE:
    class LSTMNet(nn.Module):
        def __init__(self, feature_count, hidden=64):
            super().__init__()
            self.encoder = nn.LSTM(feature_count, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            output, _ = self.encoder(x)
            return self.head(output[:, -1]).squeeze(-1)

    class CNNNet(nn.Module):
        def __init__(self, feature_count, hidden=64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(feature_count, hidden // 2, 3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden // 2, hidden, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            return self.head(self.encoder(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)

    class TransformerNet(nn.Module):
        def __init__(self, feature_count, window, hidden=64):
            super().__init__()
            self.project = nn.Linear(feature_count, hidden)
            self.position = nn.Parameter(torch.zeros(1, window, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
                dropout=.1, batch_first=True, norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            z = self.project(x) + self.position[:, :x.shape[1], :]
            return self.head(self.encoder(z).mean(dim=1)).squeeze(-1)
else:
    class _TorchRequired:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for LSTM/CNN/Transformer. "
                "Use the Google Colab notebook or install a CPU-only PyTorch wheel."
            )

    LSTMNet = CNNNet = TransformerNet = _TorchRequired


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class TorchRegressor:
    """fit/finetune/predict. Per-source input/output scaling retained on transfer."""
    def __init__(self, name: str, seed=42, epochs=30, patience=5, batch_size=64,
                 learning_rate=.001, hidden_dim=64, device='cpu'):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Run deep-learning experiments in "
                "notebooks/colab_deep_learning.ipynb, or install CPU PyTorch."
            )
        if name not in DEEP_NAMES:
            raise ValueError(name)
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Select CPU or use a Colab GPU runtime.")
        self.name, self.seed = name, seed
        self.epochs, self.patience, self.batch_size = epochs, patience, batch_size
        self.learning_rate, self.hidden_dim, self.device = learning_rate, hidden_dim, device
        self.net = None

    def _build(self, features, window):
        if self.name == 'lstm':
            return LSTMNet(features, self.hidden_dim)
        if self.name == 'cnn':
            return CNNNet(features, self.hidden_dim)
        return TransformerNet(features, window, self.hidden_dim)

    def _normalize_x(self, X):
        return ((np.asarray(X, dtype=np.float32) - self.xmean) / self.xstd).astype(np.float32)

    def fit(self, X, y, X_val, y_val):
        X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)
        X_val, y_val = np.asarray(X_val, np.float32), np.asarray(y_val, np.float32)
        if len(X) == 0 or len(X_val) == 0:
            raise ValueError('Not enough observations for sequence training')
        seed_all(self.seed)
        self.xmean = X.mean(axis=(0, 1), keepdims=True)
        self.xstd = X.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
        self.ymean = float(y.mean())
        self.ystd = max(float(y.std()), 1.0)
        self.net = self._build(X.shape[-1], X.shape[1]).to(self.device)
        self._train(X, y, X_val, y_val, self.learning_rate, self.epochs, self.patience)
        return self

    def _train(self, X, y, X_val, y_val, lr, epochs, patience):
        x = torch.from_numpy(self._normalize_x(X))
        targets = torch.tensor((np.asarray(y, dtype=np.float32) - self.ymean) / self.ystd)
        xv = torch.from_numpy(self._normalize_x(X_val)).to(self.device)
        yv = torch.tensor((np.asarray(y_val, dtype=np.float32) - self.ymean) / self.ystd).to(self.device)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(TensorDataset(x, targets), batch_size=self.batch_size,
                            shuffle=True, generator=generator)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        best, bad, best_state = float('inf'), 0, None
        for _epoch in range(epochs):
            self.net.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
            self.net.eval()
            with torch.no_grad():
                vloss = loss_fn(self.net(xv), yv).item()
            if vloss < best - 1e-6:
                best, bad = vloss, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def finetune(self, X, y, *, epochs=15, learning_rate=1e-4):
        if self.net is None:
            raise ValueError('Must pretrain before fine-tuning')
        if not len(X):
            raise ValueError('No target support samples')
        model = copy.deepcopy(self)
        model._train(X, y, X, y, learning_rate, epochs, patience=max(epochs, 1))
        return model

    def predict(self, X):
        if self.net is None:
            raise ValueError('Model is not fitted')
        if len(X) == 0:
            return np.empty(0, dtype=float)
        self.net.eval()
        vals = []
        with torch.no_grad():
            for a in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(self._normalize_x(X[a:a + self.batch_size])).to(self.device)
                vals.append(self.net(xb).cpu().numpy())
        return np.concatenate(vals) * self.ystd + self.ymean
