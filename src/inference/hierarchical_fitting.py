"""
Hierarchical PGM Inference  —  SVD-based Factor Recovery
==========================================================
The generative model is a rank-1 factor model in log-space:

    log_mu_{g,s} = w_g * z_s + b_g

So the centered log-count matrix X_c = log1p(C) - mean_per_gene
should be approximately w * z.T (outer product).

SVD of X_c gives the best rank-1 approximation, from which we recover:
  w_hat  (n_genes,)   gene factor loadings
  z_hat  (n_patients,) sample latent factors
  b_hat  (n_genes,)   gene log-baselines

Then p_g is recovered per-gene from residual counts (same MOM as simple model).

Sign ambiguity: SVD signs are arbitrary.  All recovery functions try both
orientations and report the one with better correlation to true values.
"""

from __future__ import annotations
import numpy as np
from src.inference.fitting import fit_pgm_mom


# ---------------------------------------------------------------------------
def fit_hierarchical_mom(counts: np.ndarray) -> dict:
    """
    Recover w, z, b, p from a hierarchical NegBinom count matrix.

    Parameters
    ----------
    counts : (n_genes, n_patients)  int or float

    Returns
    -------
    dict with keys:
      w_hat        : (n_genes,)     gene factor loadings
      z_hat        : (n_patients,)  sample factors (unit-variance convention)
      b_hat        : (n_genes,)     log-baseline per gene
      p_hat        : (n_genes,)     overdispersion per gene
      log_mu_hat   : (n_genes, n_patients)  reconstructed log-mean
      residual_counts : (n_genes, n_patients)  counts / exp(log_mu_hat)
      lambda_hat   : (n_genes,)     exp(b_hat)  [compat with simple pipeline]
      sv_ratio     : float          first-PC variance fraction
    """
    counts = np.asarray(counts, dtype=float)
    n_genes, n_patients = counts.shape

    # Step 1: log-transform
    X = np.log1p(counts)                              # (n_genes, n_patients)

    # Step 2: center per gene
    b_hat = X.mean(axis=1)                            # (n_genes,)
    X_c   = X - b_hat[:, None]

    # Step 3: rank-1 SVD
    U, sv, Vt = np.linalg.svd(X_c, full_matrices=False)
    # Best rank-1: s0 * u0 * v0^T
    w_hat = U[:, 0] * sv[0]                          # absorb sv into w
    z_hat = Vt[0, :]                                  # unit-norm patient scores

    sv_ratio = float(sv[0] ** 2 / (sv ** 2).sum())

    # Step 4: reconstruct log-mean
    log_mu_hat = np.outer(w_hat, z_hat) + b_hat[:, None]

    # Step 5: residuals in count space
    mu_hat   = np.exp(log_mu_hat)
    residual = np.clip(counts / (mu_hat + 1e-6), 1e-6, None)

    # Step 6: recover p per gene from residuals using MOM
    # Residual ~roughly~ scales with 1/r = (1-p)/(mu*p)
    # Use fit_pgm_mom directly on clipped residual counts (gives p of residual distr.)
    # Better: recover p from variance of original counts normalised by predicted mean
    # Var[C_{g,s}] = mu + mu²/r  where r = mu*p/(1-p)
    # => Var/mu² = 1/mu + (1-p)/p  => approx 1/r for large mu
    # Per-gene: mean(Var/mu²) ≈ (1-p)/p  =>  p = 1/(1 + mean(Var/mu²))
    p_hat_arr = np.zeros(n_genes)
    for g in range(n_genes):
        mu_g  = mu_hat[g, :]
        var_g = np.var(counts[g, :].astype(float) - mu_g, ddof=1) + 1e-6
        # NegBinom: Var = mu + mu²/r  =>  r ≈ mu²/(Var - mu)
        # then p = r/(r + mu)
        mu_m   = mu_g.mean()
        # Use ratio of per-patient: safer with heterogeneous mu
        ratio  = float(np.mean((counts[g, :] / (mu_g + 1e-6))))
        # MOM from fit_pgm_mom on original gene row:
        # just do it directly for robustness
        obs   = counts[g, :]
        m     = float(obs.mean())
        v     = float(obs.var(ddof=1)) if len(obs) > 1 else m + 1.0
        v     = max(v, m + 1e-6)
        excess = v - m
        excess = max(excess, m * 0.10)
        p_hat_arr[g] = np.clip(m / v, 1e-4, 1 - 1e-4)

    return {
        "w_hat":          w_hat,
        "z_hat":          z_hat,
        "b_hat":          b_hat,
        "p_hat":          p_hat_arr,
        "log_mu_hat":     log_mu_hat,
        "residual_counts": residual,
        "lambda_hat":     np.exp(b_hat),    # compat alias
        "sv_ratio":       sv_ratio,
    }


# ---------------------------------------------------------------------------
def _resolve_sign(
    w_hat: np.ndarray,
    z_hat: np.ndarray,
    true_w: np.ndarray,
    true_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SVD is sign-ambiguous: (w, z) and (-w, -z) give the same outer product.
    Choose the sign that maximises |corr(w_hat, true_w)|.
    """
    c1 = float(np.corrcoef(w_hat,  true_w)[0, 1]) if np.std(w_hat)  > 1e-8 else 0
    c2 = float(np.corrcoef(-w_hat, true_w)[0, 1]) if np.std(w_hat)  > 1e-8 else 0
    if c2 > c1:
        return -w_hat, -z_hat
    return w_hat, z_hat


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
def evaluate_hierarchical_recovery(
    fit: dict,
    true_w: np.ndarray,
    true_b: np.ndarray,
    true_p: np.ndarray,
    true_z: np.ndarray,
) -> dict:
    """
    Compare fitted parameters to true parameters.
    Handles sign ambiguity for w and z.
    """
    w_hat, z_hat = _resolve_sign(fit["w_hat"], fit["z_hat"], true_w, true_z)

    return {
        "corr_w":  _safe_corr(w_hat, true_w),
        "corr_z":  _safe_corr(z_hat, true_z),
        "corr_b":  _safe_corr(fit["b_hat"], true_b),
        "corr_p":  _safe_corr(fit["p_hat"], true_p),
        "mae_w":   float(np.mean(np.abs(w_hat - true_w))),
        "mae_z":   float(np.mean(np.abs(z_hat - true_z))),
        "mae_b":   float(np.mean(np.abs(fit["b_hat"] - true_b))),
        "sv_ratio": fit["sv_ratio"],
    }


# ---------------------------------------------------------------------------
def evaluate_z_shift_detection(
    orig_fit: dict,
    pert_fit: dict,
    delta: float,
    true_w: np.ndarray,
) -> dict:
    """
    After do(z_s + delta), check whether the inferred z_hat shifts in
    the expected direction (sign of delta * mean(w)).

    The expected mean shift in z_hat is delta (since z_hat has unit-variance
    convention and the true z_s was shifted by delta).
    """
    # Align signs consistently
    w_hat_o, z_hat_o = _resolve_sign(orig_fit["w_hat"], orig_fit["z_hat"], true_w, true_w)
    w_hat_p, z_hat_p = _resolve_sign(pert_fit["w_hat"], pert_fit["z_hat"], true_w, true_w)

    z_shift_per_patient = z_hat_p - z_hat_o       # (n_patients,)
    mean_z_shift        = float(z_shift_per_patient.mean())
    expected_direction  = np.sign(delta)           # z was shifted by +delta

    # The loadings should be largely unchanged by z-shift
    w_shift = w_hat_p - w_hat_o
    mean_w_shift = float(np.abs(w_shift).mean())

    corr_w_stability = _safe_corr(w_hat_o, w_hat_p)  # should be ~1

    return {
        "mean_z_shift":       mean_z_shift,
        "expected_direction": int(expected_direction),
        "direction_correct":  bool(np.sign(mean_z_shift) == expected_direction),
        "mean_w_shift_mag":   mean_w_shift,
        "corr_w_stability":   corr_w_stability,
    }
