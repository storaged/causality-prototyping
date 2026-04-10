"""
Hierarchical PGM Generator  (Option 2)
========================================
Generative model with a per-sample causal latent factor z_s:

  # Gene parameters
  w_g ~ N(0, loading_std)           gene factor loading
  b_g ~ N(baseline_mean, baseline_std²)  log-baseline expression
  p_g ~ Beta(a, b)                  overdispersion

  # Sample parameters
  z_s ~ N(0, 1)                     sample causal factor  ← the key causal axis

  # Structured mean
  log_mu_{g,s} = w_g * z_s + b_g   rank-1 factor model in log-space
  mu_{g,s}     = exp(log_mu_{g,s})

  # Count model (via Gamma-Poisson mixture; handles non-integer n)
  r_{g,s} = mu_{g,s} * p_g / (1 - p_g)   NegBinom "size" parameter
  C_{g,s} ~ NegBinom(r_{g,s}, p_g)

Key point: z_s is the causal "dose" for each patient.  Perturbing z_s shifts
gene g's expression by w_g * delta.  If an AE latent dimension also shifts
expression proportionally to w_g, that dimension is a causal handle.
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class HierarchicalPGMConfig:
    n_genes: int = 50
    n_patients: int = 1000
    loading_std: float = 1.0       # w_g ~ N(0, loading_std)
    baseline_mean: float = 1.5     # b_g ~ N(baseline_mean, baseline_std²)
    baseline_std: float = 0.5
    p_beta_a: float = 2.0
    p_beta_b: float = 5.0
    seed: int = 42


class HierarchicalPGMSimulator:
    """Generate count data from the hierarchical factor PGM."""

    def __init__(self, config: HierarchicalPGMConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    def generate(self) -> dict:
        """
        Returns
        -------
        dict with keys:
          counts       : int array (n_genes, n_patients)
          true_w       : float array (n_genes,)    factor loadings
          true_b       : float array (n_genes,)    log-baselines
          true_p       : float array (n_genes,)    overdispersion
          true_z       : float array (n_patients,) sample factors
          true_log_mu  : float array (n_genes, n_patients)
          config       : dict
        """
        cfg = self.config
        rng = self.rng

        # Gene parameters
        w_g = rng.normal(0.0, cfg.loading_std, size=cfg.n_genes)
        b_g = rng.normal(cfg.baseline_mean, cfg.baseline_std, size=cfg.n_genes)
        p_g = rng.beta(cfg.p_beta_a, cfg.p_beta_b, size=cfg.n_genes)

        # Sample factor
        z_s = rng.standard_normal(cfg.n_patients)

        # Structured mean
        log_mu = np.outer(w_g, z_s) + b_g[:, None]   # (n_genes, n_patients)
        mu     = np.exp(log_mu)

        # NegBinom size parameter (varies per gene × patient)
        p_g_bc = p_g[:, None]                         # broadcast over patients
        r      = mu * p_g_bc / (1.0 - p_g_bc)        # (n_genes, n_patients)
        r      = np.clip(r, 1e-3, None)

        # Sample counts via Gamma-Poisson mixture (exact; handles non-integer r)
        scale  = (1.0 - p_g_bc) / p_g_bc             # Gamma scale parameter
        gamma  = rng.gamma(shape=r, scale=scale)      # (n_genes, n_patients)
        counts = rng.poisson(gamma).astype(np.int64)

        return {
            "counts":      counts,
            "true_w":      w_g,
            "true_b":      b_g,
            "true_p":      p_g,
            "true_z":      z_s,
            "true_log_mu": log_mu,
            "config":      asdict(cfg),
        }

    # ------------------------------------------------------------------
    def sanity_check(self, data: dict) -> dict[str, bool]:
        counts = data["counts"]
        true_w = data["true_w"]
        true_z = data["true_z"]
        cfg    = self.config
        results: dict[str, bool] = {}

        # 1. No NaN/Inf, non-negative counts
        results["no_numerical_issues"] = bool(
            np.all(np.isfinite(counts)) and np.all(counts >= 0)
        )

        # 2. Correct shape
        results["correct_shape"] = (counts.shape == (cfg.n_genes, cfg.n_patients))

        # 3. Rank-1 structure visible: leading singular value explains >10% of variance
        X_c = np.log1p(counts.astype(float))
        X_c -= X_c.mean(axis=1, keepdims=True)
        _, sv, _ = np.linalg.svd(X_c, full_matrices=False)
        var_explained = sv[0] ** 2 / (sv ** 2).sum()
        results["rank1_structure_visible"] = bool(var_explained > 0.10)

        # 4. Overdispersion: at least half of genes have var > mean
        mean_g = counts.mean(axis=1)
        var_g  = counts.var(axis=1, ddof=1)
        results["overdispersed"] = bool((var_g > mean_g).mean() > 0.5)

        # 5. Reproducibility
        fresh     = HierarchicalPGMSimulator(self.config)
        fresh_dat = fresh.generate()
        results["reproducible_with_same_seed"] = bool(
            np.array_equal(counts, fresh_dat["counts"])
        )

        # 6. Patient-level z_s signal is detectable in log-expression space.
        # mean(w) may be near 0 (positive/negative loadings cancel in raw counts),
        # so test log1p total and variance-vs-|z_s| instead of raw mean.
        patient_log_total = np.log1p(counts.sum(axis=0))
        corr_z_log = float(np.corrcoef(patient_log_total, true_z)[0, 1])
        patient_var = counts.var(axis=0)
        corr_absz_var = float(np.corrcoef(patient_var, np.abs(true_z))[0, 1])
        results["patient_mean_correlates_with_z"] = bool(
            abs(corr_z_log) > 0.05 or abs(corr_absz_var) > 0.05
        )

        return results

    # ------------------------------------------------------------------
    def save(self, data: dict, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for key in ("counts", "true_w", "true_b", "true_p", "true_z", "true_log_mu"):
            np.save(path / f"{key}.npy", data[key])
        with open(path / "metadata.json", "w") as f:
            json.dump(data["config"], f, indent=2)
