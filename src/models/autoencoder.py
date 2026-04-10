"""
Stage 4: Simple Fully-Connected AutoEncoder
============================================
Convention: each *patient* is one sample.
  Input vector  : gene counts for one patient  (shape: n_genes)
  Latent vector : compressed representation     (shape: latent_dim)

Normalisation: log1p(counts) before encoding; exp(output) - 1 when decoding
back to count-like space.  The AE operates entirely in log1p space.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Softplus()]   # non-negative output
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

    def forward(self, x):
        z    = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

    def encode(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return self.encoder(t).numpy()

    def decode(self, z: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(z, dtype=torch.float32)
            return self.decoder(t).numpy()


# ---------------------------------------------------------------------------
def _log1p_normalize(counts: np.ndarray) -> np.ndarray:
    return np.log1p(counts.astype(float))


def train_autoencoder(
    counts: np.ndarray,          # (n_genes, n_patients)
    latent_dim: int = 4,
    hidden_dims: list[int] | None = None,
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 20,
    seed: int = 0,
    loss_type: str = "mse",      # "mse" | "poisson"
) -> tuple[AutoEncoder, dict]:
    """
    Train AE on patient-level count vectors.

    counts shape: (n_genes, n_patients) → transposed internally to (n_patients, n_genes)

    Returns
    -------
    (trained_model, history)
    history has keys: train_loss  (list of float per epoch)
    """
    if hidden_dims is None:
        hidden_dims = [32, 16]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Transpose: samples = patients, features = genes
    X = _log1p_normalize(counts).T          # (n_patients, n_genes)
    n_patients, n_genes = X.shape

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader  = DataLoader(dataset, batch_size=min(batch_size, n_patients), shuffle=True)

    model     = AutoEncoder(n_genes, hidden_dims, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # For Poisson loss we also need the raw counts as targets
    counts_tensor = torch.tensor(
        counts.T.astype(float), dtype=torch.float32
    )  # (n_patients, n_genes)
    count_dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        counts_tensor,
    )
    count_loader = DataLoader(count_dataset,
                              batch_size=min(batch_size, n_patients), shuffle=True)

    poisson_nll = nn.PoissonNLLLoss(log_input=False, full=False, reduction="mean")
    mse_loss    = nn.MSELoss()

    history = {"train_loss": [], "loss_type": loss_type}
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_log, batch_counts in count_loader:
            xhat_log, _ = model(batch_log)      # output in log1p scale

            if loss_type == "mse":
                loss = mse_loss(xhat_log, batch_log)
            elif loss_type == "poisson":
                # Convert log1p output to count-scale predicted rates
                lambda_hat = torch.expm1(xhat_log).clamp(min=1e-6)
                loss = poisson_nll(lambda_hat, batch_counts)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type!r}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_log)
        history["train_loss"].append(epoch_loss / n_patients)

    return model, history


# ---------------------------------------------------------------------------
def evaluate_reconstruction(
    model: AutoEncoder,
    counts: np.ndarray,           # (n_genes, n_patients)
) -> dict:
    """
    Measure how well the AE reconstructs the data.

    Returns decoded counts (continuous, in original count scale) plus metrics.
    """
    X = _log1p_normalize(counts).T                      # (n_patients, n_genes)
    X_hat_log = model.encode(X)                         # latent
    X_hat_log = model.decode(X_hat_log)                 # back in log1p scale

    # Convert back to count-like scale
    X_hat = np.expm1(X_hat_log)                         # (n_patients, n_genes)

    # Correlations
    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    orig_flat = counts.T.ravel()
    rec_flat  = X_hat.ravel()
    corr      = safe_corr(orig_flat, rec_flat)
    mse       = float(np.mean((orig_flat - rec_flat) ** 2))

    # Per-patient correlation
    per_patient_corr = [
        safe_corr(counts[:, n], X_hat[n]) for n in range(counts.shape[1])
    ]

    return {
        "reconstruction_corr":   corr,
        "reconstruction_mse":    mse,
        "mean_per_patient_corr": float(np.mean(per_patient_corr)),
        "decoded_counts":        X_hat.T,   # back to (n_genes, n_patients)
    }


# ---------------------------------------------------------------------------
def pca_baseline(counts: np.ndarray, latent_dim: int) -> dict:
    """
    PCA with same latent dimension as AE.  Returns reconstruction metrics.
    """
    from sklearn.decomposition import PCA

    X = _log1p_normalize(counts).T                      # (n_patients, n_genes)

    n_components = min(latent_dim, X.shape[0] - 1, X.shape[1])
    pca   = PCA(n_components=n_components)
    Z     = pca.fit_transform(X)
    X_hat = pca.inverse_transform(Z)

    X_hat_counts = np.expm1(X_hat)

    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    corr = safe_corr(counts.T.ravel(), X_hat_counts.ravel())
    mse  = float(np.mean((counts.T.ravel() - X_hat_counts.ravel()) ** 2))

    return {
        "pca_reconstruction_corr": corr,
        "pca_reconstruction_mse":  mse,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
