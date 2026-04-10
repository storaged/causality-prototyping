"""
Stage 3: PGM Parameter Recovery
=================================
Fit the NegBinom PGM back to observed count data using Method of Moments (MOM).

For each gene i with observations x = C[i, :]:
  mean_hat = mean(x)
  var_hat  = var(x, ddof=1)

  NegBinom MOM:
    p_hat     = mean_hat / var_hat          (prob of success)
    lambda_hat = mean_hat^2 / (var_hat - mean_hat)   (n = number of successes)

Edge cases:
  - var <= mean  → Poisson-like data; clip lambda_hat to large value, p_hat to ~1
  - mean ≈ 0     → clip both to safe defaults
"""

from __future__ import annotations
import numpy as np
from scipy import stats


def fit_pgm_mom(counts: np.ndarray) -> dict:
    """
    Fit NegBinom parameters per gene using Method of Moments.

    Parameters
    ----------
    counts : int/float array of shape (n_genes, n_patients)
        Can be raw counts or decoded (continuous) values from AE.

    Returns
    -------
    dict with:
      lambda_hat : array (n_genes,)
      p_hat      : array (n_genes,)
      mean_hat   : array (n_genes,)
      var_hat    : array (n_genes,)
    """
    counts = np.asarray(counts, dtype=float)
    n_genes, n_patients = counts.shape

    mean_hat = counts.mean(axis=1)                       # (n_genes,)
    var_hat  = counts.var(axis=1, ddof=min(1, n_patients - 1))  # (n_genes,)

    # Ensure numerical stability
    mean_hat = np.clip(mean_hat, 1e-6, None)

    # Require excess variance to be at least 10 % of the mean to avoid
    # division-by-tiny which inflates lambda_hat to astronomic values with
    # small sample sizes (N=20 is easily under-dispersed by chance).
    excess_var = var_hat - mean_hat
    excess_var = np.clip(excess_var, mean_hat * 0.10, None)
    var_hat    = mean_hat + excess_var                   # keep var > mean

    p_hat      = mean_hat / var_hat                      # ∈ (0, 1)
    p_hat      = np.clip(p_hat, 1e-4, 1 - 1e-4)

    lambda_hat = mean_hat ** 2 / excess_var              # > 0
    lambda_hat = np.clip(lambda_hat, 0.5, 100.0)        # bound to plausible range

    return {
        "lambda_hat": lambda_hat,
        "p_hat":      p_hat,
        "mean_hat":   mean_hat,
        "var_hat":    var_hat,
    }


def evaluate_recovery(fit_result: dict, true_lambda: np.ndarray, true_p: np.ndarray) -> dict:
    """
    Compute recovery metrics comparing fitted to true parameters.

    Returns
    -------
    dict with corr_lambda, corr_p, mae_lambda, mae_p, rmse_lambda, rmse_p
    """
    lambda_hat = fit_result["lambda_hat"]
    p_hat      = fit_result["p_hat"]

    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    corr_lambda = safe_corr(lambda_hat, true_lambda)
    corr_p      = safe_corr(p_hat,      true_p)

    mae_lambda  = float(np.mean(np.abs(lambda_hat - true_lambda)))
    mae_p       = float(np.mean(np.abs(p_hat      - true_p)))
    rmse_lambda = float(np.sqrt(np.mean((lambda_hat - true_lambda) ** 2)))
    rmse_p      = float(np.sqrt(np.mean((p_hat      - true_p) ** 2)))

    return {
        "corr_lambda":  corr_lambda,
        "corr_p":       corr_p,
        "mae_lambda":   mae_lambda,
        "mae_p":        mae_p,
        "rmse_lambda":  rmse_lambda,
        "rmse_p":       rmse_p,
    }


def evaluate_perturbation_detection(
    orig_fit: dict,
    pert_fit: dict,
    target_genes: list[int],
    n_genes: int,
    param: str = "p",
    top_k: int = 10,
) -> dict:
    """
    Evaluate how well parameter shifts identify the truly perturbed genes.

    Uses: shift = |param_hat_perturbed - param_hat_original| as a ranking signal.

    Returns
    -------
    dict with overlap_at_k, mean_shift_affected, mean_shift_unaffected, shift_ratio
    """
    if param == "p":
        orig_vals = orig_fit["p_hat"]
        pert_vals = pert_fit["p_hat"]
    else:
        orig_vals = orig_fit["lambda_hat"]
        pert_vals = pert_fit["lambda_hat"]

    shifts = np.abs(pert_vals - orig_vals)
    non_target = [i for i in range(n_genes) if i not in target_genes]

    top_k_genes = set(np.argsort(shifts)[::-1][:top_k])
    target_set  = set(target_genes)
    overlap = len(top_k_genes & target_set)

    mean_shift_affected   = shifts[target_genes].mean() if target_genes else 0.0
    mean_shift_unaffected = shifts[non_target].mean()   if non_target   else 0.0
    shift_ratio = mean_shift_affected / (mean_shift_unaffected + 1e-8)

    return {
        "overlap_at_k":          overlap,
        "top_k":                 top_k,
        "n_target":              len(target_genes),
        "mean_shift_affected":   float(mean_shift_affected),
        "mean_shift_unaffected": float(mean_shift_unaffected),
        "shift_ratio":           float(shift_ratio),
    }
