"""
Variational AutoEncoder (VAE / beta-VAE)
=========================================
Same patient-vector convention as the plain AE.
Encoder → (μ, log σ²) → reparameterisation → z → decoder → x̂

Loss = reconstruction_loss  +  β · KL[ N(μ, σ²) || N(0, I) ]

  KL per sample = -0.5 · Σ_j (1 + log σ²_j  − μ²_j − σ²_j)

β = 1 is standard VAE; β > 1 (beta-VAE) pushes towards disentanglement.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net    = nn.Sequential(*layers)
        self.mu     = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

    def forward(self, x):
        h      = self.net(x)
        mu     = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        self.encoder    = VAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder    = VAEDecoder(latent_dim, hidden_dims, input_dim)
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval time

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterise(mu, logvar)
        xhat       = self.decoder(z)
        return xhat, z, mu, logvar

    # ------------------------------------------------------------------
    def encode(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            mu, _ = self.encoder(t)
            return mu.numpy()

    def decode(self, z: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(z, dtype=torch.float32)
            return self.decoder(t).numpy()


# ---------------------------------------------------------------------------
def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL divergence per sample."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def _log1p_normalize(counts: np.ndarray) -> np.ndarray:
    return np.log1p(counts.astype(float))


def train_vae(
    counts: np.ndarray,
    latent_dim: int = 4,
    hidden_dims: list[int] | None = None,
    n_epochs: int = 600,
    lr: float = 1e-3,
    batch_size: int = 20,
    beta: float = 1.0,
    loss_type: str = "mse",      # "mse" | "poisson"
    seed: int = 0,
) -> tuple[VAE, dict]:
    """
    Train VAE.  beta > 1 encourages disentangled representations.

    Returns (model, history) where history has:
      train_loss, recon_loss, kl_loss
    """
    if hidden_dims is None:
        hidden_dims = [32, 16]

    torch.manual_seed(seed)
    np.random.seed(seed)

    X = _log1p_normalize(counts).T              # (n_patients, n_genes)
    n_patients, n_genes = X.shape

    counts_tensor = torch.tensor(counts.T.astype(float), dtype=torch.float32)
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        counts_tensor,
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, n_patients), shuffle=True)

    model     = VAE(n_genes, hidden_dims, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    poisson_nll = nn.PoissonNLLLoss(log_input=False, full=False, reduction="mean")
    mse_loss    = nn.MSELoss()

    history: dict[str, list] = {"train_loss": [], "recon_loss": [], "kl_loss": []}
    model.train()

    for epoch in range(n_epochs):
        ep_total = ep_recon = ep_kl = 0.0
        for batch_log, batch_counts in loader:
            xhat, _, mu, logvar = model(batch_log)

            if loss_type == "mse":
                recon = mse_loss(xhat, batch_log)
            else:  # poisson
                lambda_hat = torch.expm1(xhat).clamp(min=1e-6)
                recon = poisson_nll(lambda_hat, batch_counts)

            kl   = _kl_divergence(mu, logvar)
            loss = recon + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(batch_log)
            ep_total += loss.item() * n
            ep_recon += recon.item() * n
            ep_kl    += kl.item() * n

        history["train_loss"].append(ep_total / n_patients)
        history["recon_loss"].append(ep_recon / n_patients)
        history["kl_loss"].append(ep_kl / n_patients)

    return model, history


# ---------------------------------------------------------------------------
def evaluate_reconstruction(
    model: VAE,
    counts: np.ndarray,
) -> dict:
    """Same interface as autoencoder.evaluate_reconstruction."""
    X = _log1p_normalize(counts).T
    Z = model.encode(X)
    X_hat_log   = model.decode(Z)
    X_hat       = np.expm1(X_hat_log)

    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    orig_flat = counts.T.ravel()
    rec_flat  = X_hat.ravel()
    return {
        "reconstruction_corr":   safe_corr(orig_flat, rec_flat),
        "reconstruction_mse":    float(np.mean((orig_flat - rec_flat) ** 2)),
        "mean_per_patient_corr": float(np.mean([
            safe_corr(counts[:, n], X_hat[n]) for n in range(counts.shape[1])
        ])),
        "decoded_counts": X_hat.T,
    }
