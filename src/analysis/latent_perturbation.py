"""
Stage 6: Single-Latent Perturbation Experiments
=================================================
For each latent dimension l:
  1. Encode all patient vectors to get Z  (n_patients, latent_dim)
  2. For each perturbation magnitude delta (in units of std_l):
       Z_perturbed[:, l] = Z[:, l] + delta * std_l
  3. Decode Z_perturbed → count matrix
  4. Fit PGM to decoded matrix
  5. Record lambda_hat, p_hat

Sensitivity of PGM parameter θ to latent dim l:
  slope of linear regression of mean(θ_hat) across patients on delta
"""

from __future__ import annotations
import numpy as np
from scipy import stats

from src.models.autoencoder import AutoEncoder, _log1p_normalize
from src.inference.fitting import fit_pgm_mom


def screen_latent_dimensions(
    model: AutoEncoder,
    counts: np.ndarray,           # (n_genes, n_patients)
    magnitudes: list[float] | None = None,
) -> dict:
    """
    Perturb each latent dimension across multiple magnitudes and record PGM fits.

    Returns
    -------
    dict with keys:
      latent_dim, n_magnitudes
      results[l][delta] = {"lambda_hat": ..., "p_hat": ...}
      sensitivity        = array (latent_dim, 2)  columns: [lambda_sens, p_sens]
      baseline_fit       = fit on unperturbed reconstruction
    """
    if magnitudes is None:
        magnitudes = [-2.0, -1.0, 0.0, 1.0, 2.0]

    X = _log1p_normalize(counts).T                    # (n_patients, n_genes)

    # Encode once
    Z = model.encode(X)                               # (n_patients, latent_dim)
    latent_dim = Z.shape[1]
    z_std = Z.std(axis=0) + 1e-8                     # per-dimension std

    # Baseline: decode without perturbation
    X_hat_log = model.decode(Z)
    baseline_counts = np.expm1(X_hat_log).T          # (n_genes, n_patients)
    baseline_fit = fit_pgm_mom(baseline_counts)

    # Full screening
    results: dict[int, dict[float, dict]] = {}
    for l in range(latent_dim):
        results[l] = {}
        for delta in magnitudes:
            Z_pert = Z.copy()
            Z_pert[:, l] += delta * z_std[l]
            X_pert_log = model.decode(Z_pert)
            pert_counts = np.expm1(X_pert_log).T     # (n_genes, n_patients)
            fit = fit_pgm_mom(pert_counts)
            results[l][delta] = {
                "lambda_hat":   fit["lambda_hat"],
                "p_hat":        fit["p_hat"],
                "mean_lambda":  float(fit["lambda_hat"].mean()),
                "mean_p":       float(fit["p_hat"].mean()),
                "decoded_counts": pert_counts,        # stored for hierarchical analysis
            }

    # Compute sensitivity: slope of mean(param) vs delta for each (dim, param)
    sensitivity = np.zeros((latent_dim, 2))          # [lambda_sens, p_sens]
    for l in range(latent_dim):
        deltas    = np.array(magnitudes)
        ml = np.array([results[l][d]["mean_lambda"] for d in magnitudes])
        mp = np.array([results[l][d]["mean_p"]      for d in magnitudes])
        if np.std(deltas) > 1e-8:
            sensitivity[l, 0] = float(np.polyfit(deltas, ml, 1)[0])  # lambda slope
            sensitivity[l, 1] = float(np.polyfit(deltas, mp, 1)[0])  # p slope

    return {
        "latent_dim":    latent_dim,
        "magnitudes":    magnitudes,
        "results":       results,
        "baseline_fit":  baseline_fit,
        "sensitivity":   sensitivity,     # (latent_dim, 2)
        "z_std":         z_std,
        "Z_encoded":     Z,
    }


def compute_per_gene_shift_vectors(
    screen_result: dict,
    baseline_fit: dict,
    delta: float = 1.0,
) -> dict:
    """
    For each latent dimension, compute per-gene shift vector for p and lambda
    at a specified magnitude (+delta std).

    Returns
    -------
    dict with keys:
      shift_lambda : array (latent_dim, n_genes)
      shift_p      : array (latent_dim, n_genes)
    """
    latent_dim = screen_result["latent_dim"]
    results    = screen_result["results"]

    # Find the closest magnitude to +delta
    target_delta = min(screen_result["magnitudes"], key=lambda d: abs(d - delta))

    n_genes = len(baseline_fit["lambda_hat"])
    shift_lambda = np.zeros((latent_dim, n_genes))
    shift_p      = np.zeros((latent_dim, n_genes))

    for l in range(latent_dim):
        shift_lambda[l] = results[l][target_delta]["lambda_hat"] - baseline_fit["lambda_hat"]
        shift_p[l]      = results[l][target_delta]["p_hat"]      - baseline_fit["p_hat"]

    return {"shift_lambda": shift_lambda, "shift_p": shift_p, "delta_used": target_delta}
