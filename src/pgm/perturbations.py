"""
Stage 2: Ground-Truth Perturbation Engine
==========================================
Applies controlled interventions directly in the PGM generative space.

Supported intervention types:
  - 'p_prior'     : change Beta parameters for p_i of selected genes
  - 'lambda_shift': add constant to lambda_i of selected genes
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal

from src.pgm.generator import PGMConfig, PGMSimulator


@dataclass
class PerturbationManifest:
    intervention_type: str
    target_genes: list[int]
    original_params: dict
    perturbed_params: dict
    seed: int


class GenerativePerturbationEngine:
    """Perturb the PGM at the generative level and regenerate data."""

    def __init__(self, base_config: PGMConfig, rng_seed: int | None = None):
        self.base_config = base_config
        # Use a separate rng for perturbation experiments so base data is stable
        self.rng_seed = rng_seed if rng_seed is not None else base_config.seed + 100

    # ------------------------------------------------------------------
    def perturb_p_prior(
        self,
        target_genes: list[int],
        new_a: float,
        new_b: float | None = None,
    ) -> tuple[dict, dict, PerturbationManifest]:
        """
        Change the Beta prior for p_i of `target_genes` from Beta(a, b) to
        Beta(new_a, new_b or same b) and regenerate.

        Returns (original_data, perturbed_data, manifest).
        """
        cfg = self.base_config
        new_b = new_b if new_b is not None else cfg.p_beta_b

        # Original data
        orig_sim  = PGMSimulator(cfg)
        orig_data = orig_sim.generate()

        # Build perturbed data by re-running generator with modified priors
        perturbed_data = self._generate_with_modified_p(
            cfg, target_genes, new_a, new_b, orig_data
        )

        manifest = PerturbationManifest(
            intervention_type="p_prior",
            target_genes=target_genes,
            original_params={"p_beta_a": cfg.p_beta_a, "p_beta_b": cfg.p_beta_b},
            perturbed_params={"p_beta_a": new_a, "p_beta_b": new_b},
            seed=self.rng_seed,
        )
        return orig_data, perturbed_data, manifest

    # ------------------------------------------------------------------
    def perturb_lambda(
        self,
        target_genes: list[int],
        lambda_shift: float,
    ) -> tuple[dict, dict, PerturbationManifest]:
        """
        Shift lambda_i by `lambda_shift` for `target_genes` and regenerate counts.
        """
        cfg = self.base_config

        orig_sim  = PGMSimulator(cfg)
        orig_data = orig_sim.generate()

        perturbed_lambda = orig_data["true_lambda"].copy()
        perturbed_lambda[target_genes] = np.clip(
            perturbed_lambda[target_genes] + lambda_shift, 1.0, None
        )

        # Regenerate counts with modified lambda; keep same p and same rng seed
        rng = np.random.default_rng(cfg.seed)
        _ = rng.poisson(cfg.lambda_poisson_rate, size=cfg.n_genes)   # burn original draws
        _ = rng.beta(cfg.p_beta_a, cfg.p_beta_b, size=cfg.n_genes)

        counts = np.zeros((cfg.n_genes, cfg.n_patients), dtype=np.int64)
        for i in range(cfg.n_genes):
            lam = int(np.clip(perturbed_lambda[i], 1, None))
            counts[i, :] = rng.negative_binomial(lam, orig_data["true_p"][i], size=cfg.n_patients)

        perturbed_data = {
            "counts": counts,
            "true_lambda": perturbed_lambda,
            "true_p": orig_data["true_p"].copy(),
            "config": asdict(cfg),
        }

        manifest = PerturbationManifest(
            intervention_type="lambda_shift",
            target_genes=target_genes,
            original_params={"lambda": orig_data["true_lambda"][target_genes].tolist()},
            perturbed_params={"lambda_shift": lambda_shift},
            seed=cfg.seed,
        )
        return orig_data, perturbed_data, manifest

    # ------------------------------------------------------------------
    def sanity_check(
        self,
        orig_data: dict,
        pert_data: dict,
        manifest: PerturbationManifest,
    ) -> dict[str, bool]:
        target = manifest.target_genes
        non_target = [i for i in range(orig_data["counts"].shape[0]) if i not in target]

        # Mean absolute change per gene
        diff = np.abs(
            pert_data["counts"].mean(axis=1) - orig_data["counts"].mean(axis=1)
        )
        mean_affected   = diff[target].mean()
        mean_unaffected = diff[non_target].mean() if non_target else 0.0

        results = {}
        # 1. Affected genes change more than unaffected
        results["affected_change_more"] = bool(mean_affected > mean_unaffected)

        # 2. Overall stats shift (not exactly the same matrix)
        results["data_changed"] = bool(
            not np.array_equal(orig_data["counts"], pert_data["counts"])
        )

        # 3. Directional effect for p_prior: lower a → higher mean count for target genes
        if manifest.intervention_type == "p_prior":
            orig_a = manifest.original_params["p_beta_a"]
            new_a  = manifest.perturbed_params["p_beta_a"]
            orig_mean_target = orig_data["counts"][target, :].mean()
            pert_mean_target = pert_data["counts"][target, :].mean()
            # Lower a → lower p_i mean → higher NegBinom mean
            if new_a < orig_a:
                results["directional_effect_correct"] = bool(pert_mean_target > orig_mean_target)
            else:
                results["directional_effect_correct"] = bool(pert_mean_target < orig_mean_target)
        elif manifest.intervention_type == "lambda_shift":
            shift = manifest.perturbed_params["lambda_shift"]
            orig_mean_target = orig_data["counts"][target, :].mean()
            pert_mean_target = pert_data["counts"][target, :].mean()
            if shift > 0:
                results["directional_effect_correct"] = bool(pert_mean_target > orig_mean_target)
            else:
                results["directional_effect_correct"] = bool(pert_mean_target < orig_mean_target)

        return results

    # ------------------------------------------------------------------
    def save(
        self,
        orig_data: dict,
        pert_data: dict,
        manifest: PerturbationManifest,
        path: str | Path,
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "original_counts.npy", orig_data["counts"])
        np.save(path / "perturbed_counts.npy", pert_data["counts"])
        with open(path / "perturbation_manifest.json", "w") as f:
            json.dump(asdict(manifest), f, indent=2)

    # ------------------------------------------------------------------
    def _generate_with_modified_p(
        self,
        cfg: PGMConfig,
        target_genes: list[int],
        new_a: float,
        new_b: float,
        orig_data: dict,
    ) -> dict:
        """
        Regenerate counts using original lambda and original p for non-target genes,
        but resample p for target genes from Beta(new_a, new_b).
        """
        rng = np.random.default_rng(self.rng_seed)

        new_p = orig_data["true_p"].copy()
        new_p[target_genes] = rng.beta(new_a, new_b, size=len(target_genes))

        counts = np.zeros((cfg.n_genes, cfg.n_patients), dtype=np.int64)
        for i in range(cfg.n_genes):
            lam = int(np.clip(orig_data["true_lambda"][i], 1, None))
            counts[i, :] = rng.negative_binomial(lam, new_p[i], size=cfg.n_patients)

        return {
            "counts": counts,
            "true_lambda": orig_data["true_lambda"].copy(),
            "true_p": new_p,
            "config": asdict(cfg),
        }
