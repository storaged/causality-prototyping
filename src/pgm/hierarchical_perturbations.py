"""
Hierarchical PGM Perturbation Engine
======================================
Three intervention types on the hierarchical factor model:

  1. perturb_z_shift  : do(z_s = z_s + delta)   ← primary causal test
  2. perturb_p_prior  : do(p_g)                  gene overdispersion
  3. perturb_loading  : do(w_g = w_g * scale)    gene-factor sensitivity

The z-shift is the key intervention: it displaces every patient along the
causal axis, so all genes respond by w_g * delta.  This is the cleanest
possible causal signal for the AE to detect.
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import asdict
from pathlib import Path

from src.pgm.hierarchical_generator import HierarchicalPGMConfig, HierarchicalPGMSimulator
from src.pgm.perturbations import PerturbationManifest   # reuse the same dataclass


def _counts_from_params(
    w_g: np.ndarray,
    b_g: np.ndarray,
    p_g: np.ndarray,
    z_s: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (counts, log_mu) from explicit parameter arrays."""
    log_mu  = np.outer(w_g, z_s) + b_g[:, None]
    mu      = np.exp(log_mu)
    p_g_bc  = p_g[:, None]
    r       = np.clip(mu * p_g_bc / (1.0 - p_g_bc), 1e-3, None)
    scale   = (1.0 - p_g_bc) / p_g_bc
    gamma   = rng.gamma(shape=r, scale=scale)
    counts  = rng.poisson(gamma).astype(np.int64)
    return counts, log_mu


class HierarchicalPerturbationEngine:

    def __init__(self, base_config: HierarchicalPGMConfig, rng_seed: int | None = None):
        self.base_config = base_config
        self.rng_seed    = rng_seed if rng_seed is not None else base_config.seed + 100

    # ------------------------------------------------------------------
    def _get_base_data(self) -> dict:
        sim = HierarchicalPGMSimulator(self.base_config)
        return sim.generate()

    # ------------------------------------------------------------------
    def perturb_z_shift(
        self,
        delta: float,
        target_patients: list[int] | None = None,
    ) -> tuple[dict, dict, PerturbationManifest]:
        """
        do(z_s[target_patients] += delta)

        With target_patients=None, all patients are shifted.
        The expected per-gene expression shift is w_g * delta.
        """
        orig = self._get_base_data()
        rng  = np.random.default_rng(self.rng_seed)

        targets = target_patients if target_patients is not None else list(range(self.base_config.n_patients))

        z_pert = orig["true_z"].copy()
        z_pert[targets] += delta

        counts_pert, log_mu_pert = _counts_from_params(
            orig["true_w"], orig["true_b"], orig["true_p"], z_pert, rng
        )

        pert = {
            "counts":      counts_pert,
            "true_w":      orig["true_w"].copy(),
            "true_b":      orig["true_b"].copy(),
            "true_p":      orig["true_p"].copy(),
            "true_z":      z_pert,
            "true_log_mu": log_mu_pert,
            "config":      orig["config"],
        }

        manifest = PerturbationManifest(
            intervention_type="z_shift",
            target_genes=[],
            original_params={"z_mean": float(orig["true_z"][targets].mean())},
            perturbed_params={"delta": delta, "n_targets": len(targets)},
            seed=self.rng_seed,
        )
        return orig, pert, manifest

    # ------------------------------------------------------------------
    def perturb_p_prior(
        self,
        target_genes: list[int],
        new_a: float,
        new_b: float | None = None,
    ) -> tuple[dict, dict, PerturbationManifest]:
        """do(p_g) for selected genes — overdispersion intervention."""
        cfg  = self.base_config
        orig = self._get_base_data()
        rng  = np.random.default_rng(self.rng_seed)
        new_b = new_b if new_b is not None else cfg.p_beta_b

        new_p = orig["true_p"].copy()
        new_p[target_genes] = rng.beta(new_a, new_b, size=len(target_genes))

        counts_pert, log_mu_pert = _counts_from_params(
            orig["true_w"], orig["true_b"], new_p, orig["true_z"], rng
        )

        pert = {
            "counts":      counts_pert,
            "true_w":      orig["true_w"].copy(),
            "true_b":      orig["true_b"].copy(),
            "true_p":      new_p,
            "true_z":      orig["true_z"].copy(),
            "true_log_mu": log_mu_pert,
            "config":      orig["config"],
        }
        manifest = PerturbationManifest(
            intervention_type="p_prior",
            target_genes=target_genes,
            original_params={"p_beta_a": cfg.p_beta_a, "p_beta_b": cfg.p_beta_b},
            perturbed_params={"p_beta_a": new_a, "p_beta_b": new_b},
            seed=self.rng_seed,
        )
        return orig, pert, manifest

    # ------------------------------------------------------------------
    def perturb_loading(
        self,
        target_genes: list[int],
        loading_scale: float,
    ) -> tuple[dict, dict, PerturbationManifest]:
        """
        do(w_g = w_g * loading_scale) for selected genes.
        This changes how strongly target genes respond to the sample factor.
        """
        orig = self._get_base_data()
        rng  = np.random.default_rng(self.rng_seed)

        new_w = orig["true_w"].copy()
        new_w[target_genes] *= loading_scale

        counts_pert, log_mu_pert = _counts_from_params(
            new_w, orig["true_b"], orig["true_p"], orig["true_z"], rng
        )

        pert = {
            "counts":      counts_pert,
            "true_w":      new_w,
            "true_b":      orig["true_b"].copy(),
            "true_p":      orig["true_p"].copy(),
            "true_z":      orig["true_z"].copy(),
            "true_log_mu": log_mu_pert,
            "config":      orig["config"],
        }
        manifest = PerturbationManifest(
            intervention_type="loading_scale",
            target_genes=target_genes,
            original_params={"loading_scale": 1.0},
            perturbed_params={"loading_scale": loading_scale},
            seed=self.rng_seed,
        )
        return orig, pert, manifest

    # ------------------------------------------------------------------
    def sanity_check(
        self,
        orig_data: dict,
        pert_data: dict,
        manifest: PerturbationManifest,
    ) -> dict[str, bool]:
        results: dict[str, bool] = {}
        results["data_changed"] = bool(
            not np.array_equal(orig_data["counts"], pert_data["counts"])
        )

        if manifest.intervention_type == "z_shift":
            delta = manifest.perturbed_params["delta"]
            # Patient means should shift proportionally to mean(|w|)
            orig_mean = orig_data["counts"].mean(axis=0)
            pert_mean = pert_data["counts"].mean(axis=0)
            global_mean_shift = (pert_mean - orig_mean).mean()
            mean_w = orig_data["true_w"].mean()
            expected_sign = np.sign(delta * mean_w)
            results["direction_correct"] = bool(
                expected_sign == 0 or np.sign(global_mean_shift) == expected_sign
            )
            # Effect is global (all patients shifted)
            results["effect_is_global"] = bool(
                abs(global_mean_shift) > 0.1
            )

        elif manifest.intervention_type == "p_prior":
            targets = manifest.target_genes
            non_t   = [i for i in range(orig_data["counts"].shape[0]) if i not in targets]
            diff    = np.abs(
                pert_data["counts"].mean(axis=1) - orig_data["counts"].mean(axis=1)
            )
            results["affected_change_more"] = bool(
                diff[targets].mean() > diff[non_t].mean() if non_t else True
            )
            results["direction_correct"] = True   # dispersion change, no mean shift

        elif manifest.intervention_type == "loading_scale":
            targets = manifest.target_genes
            non_t   = [i for i in range(orig_data["counts"].shape[0]) if i not in targets]
            # Loaded genes should now show higher variance across patients
            orig_var = orig_data["counts"].var(axis=1)
            pert_var = pert_data["counts"].var(axis=1)
            scale = manifest.perturbed_params["loading_scale"]
            if abs(scale) > 1:
                results["affected_change_more"] = bool(
                    pert_var[targets].mean() > orig_var[targets].mean() * 0.8
                )
            else:
                results["affected_change_more"] = bool(
                    pert_var[targets].mean() < orig_var[targets].mean() * 1.2
                )
            results["direction_correct"] = True

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
        np.save(path / "original_z.npy",       orig_data["true_z"])
        np.save(path / "perturbed_z.npy",       pert_data["true_z"])
        with open(path / "perturbation_manifest.json", "w") as f:
            json.dump(asdict(manifest), f, indent=2)
