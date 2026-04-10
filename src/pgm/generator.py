"""
Stage 1: PGM Simulator
======================
NegBinomial count model:
  lambda_i ~ Poisson(lambda_rate)    [per gene, clipped to >= 1]
  p_i      ~ Beta(a, b)              [per gene]
  C_{i,n}  ~ NegBinom(lambda_i, p_i) [count matrix: genes x patients]

NegBinom parameterisation (numpy / scipy):
  n = lambda_i (number of successes)
  p = p_i      (probability of success per trial)
  mean     = n * (1 - p) / p
  variance = n * (1 - p) / p^2
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PGMConfig:
    n_genes: int = 50
    n_patients: int = 20
    lambda_poisson_rate: float = 5.0
    p_beta_a: float = 2.0
    p_beta_b: float = 5.0
    seed: int = 42


class PGMSimulator:
    """Generate synthetic count data from the NegBinom PGM."""

    def __init__(self, config: PGMConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    def generate(self) -> dict:
        """
        Returns
        -------
        dict with keys:
          counts      : int array (n_genes, n_patients)
          true_lambda : float array (n_genes,)
          true_p      : float array (n_genes,)
          config      : dict
        """
        cfg = self.config
        rng = self.rng

        # Gene-level parameters
        lambda_i = rng.poisson(cfg.lambda_poisson_rate, size=cfg.n_genes).astype(float)
        lambda_i = np.clip(lambda_i, 1.0, None)   # NegBinom requires n >= 1

        p_i = rng.beta(cfg.p_beta_a, cfg.p_beta_b, size=cfg.n_genes)

        # Count matrix  C[gene, patient]
        counts = np.zeros((cfg.n_genes, cfg.n_patients), dtype=np.int64)
        for i in range(cfg.n_genes):
            counts[i, :] = rng.negative_binomial(
                int(lambda_i[i]), p_i[i], size=cfg.n_patients
            )

        return {
            "counts": counts,
            "true_lambda": lambda_i,
            "true_p": p_i,
            "config": asdict(cfg),
        }

    # ------------------------------------------------------------------
    def sanity_check(self, data: dict) -> dict[str, bool]:
        """Run Stage 1 sanity checks.  Returns pass/fail per check."""
        counts      = data["counts"]
        true_lambda = data["true_lambda"]
        true_p      = data["true_p"]
        cfg         = self.config

        results: dict[str, bool] = {}

        # 1. No NaN / Inf
        results["no_numerical_issues"] = bool(
            np.all(np.isfinite(counts)) and np.all(counts >= 0)
        )

        # 2. Correct shape
        results["correct_shape"] = (counts.shape == (cfg.n_genes, cfg.n_patients))

        # 3. Reasonable distributions
        empirical_mean = counts.mean()
        theoretical_mean = float(np.mean(true_lambda * (1 - true_p) / true_p))
        # Accept if within 50 % of theoretical (noisy with small N)
        results["reasonable_distribution"] = bool(
            abs(empirical_mean - theoretical_mean) / (theoretical_mean + 1e-8) < 0.5
        )

        # 4. Reproducibility – regenerate with fresh simulator using same seed
        fresh_sim  = PGMSimulator(self.config)
        fresh_data = fresh_sim.generate()
        results["reproducible_with_same_seed"] = bool(
            np.array_equal(counts, fresh_data["counts"])
        )

        # 5. Changing one hyperparameter visibly changes statistics
        alt_cfg = PGMConfig(**{**asdict(cfg), "lambda_poisson_rate": 15.0, "seed": cfg.seed + 1})
        alt_sim  = PGMSimulator(alt_cfg)
        alt_data = alt_sim.generate()
        results["hyperparameter_change_shifts_stats"] = bool(
            alt_data["counts"].mean() > counts.mean() * 1.5
        )

        return results

    # ------------------------------------------------------------------
    def save(self, data: dict, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "counts.npy", data["counts"])
        np.save(path / "true_lambda.npy", data["true_lambda"])
        np.save(path / "true_p.npy", data["true_p"])
        with open(path / "metadata.json", "w") as f:
            json.dump(data["config"], f, indent=2)
