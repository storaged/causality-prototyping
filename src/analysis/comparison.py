"""
Stage 7: Compare Latent Perturbations to True Interventions
=============================================================
Metrics:
  - Cosine similarity between per-gene shift vectors
  - Correlation of shift vectors
  - Overlap of top-K most affected genes
"""

from __future__ import annotations
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Fraction of top-k genes (by absolute shift) shared between a and b."""
    top_a = set(np.argsort(np.abs(a))[::-1][:k])
    top_b = set(np.argsort(np.abs(b))[::-1][:k])
    if not top_a:
        return 0.0
    return len(top_a & top_b) / k


def compare_to_true_intervention(
    true_shift_p: np.ndarray,           # (n_genes,) shift in p from true perturbation
    true_shift_lambda: np.ndarray,      # (n_genes,) shift in lambda from true perturbation
    latent_shift_p: np.ndarray,         # (latent_dim, n_genes)
    latent_shift_lambda: np.ndarray,    # (latent_dim, n_genes)
    top_k: int = 10,
) -> dict:
    """
    For each latent dimension compute similarity to the true intervention.

    Returns
    -------
    dict with per-dimension metrics and best_dim identification.
    """
    latent_dim, n_genes = latent_shift_p.shape
    top_k = min(top_k, n_genes)

    per_dim = []
    for l in range(latent_dim):
        sp_lat = latent_shift_p[l]
        sl_lat = latent_shift_lambda[l]

        # Combined shift vector: concatenate p and lambda shifts
        true_vec  = np.concatenate([true_shift_p, true_shift_lambda])
        lat_vec   = np.concatenate([sp_lat, sl_lat])

        cos_combined = cosine_similarity(true_vec, lat_vec)
        cor_p        = safe_corr(true_shift_p, sp_lat)
        cor_lambda   = safe_corr(true_shift_lambda, sl_lat)
        overlap_p    = top_k_overlap(true_shift_p, sp_lat, top_k)
        overlap_l    = top_k_overlap(true_shift_lambda, sl_lat, top_k)

        per_dim.append({
            "latent_dim":        l,
            "cosine_combined":   cos_combined,
            "corr_p":            cor_p,
            "corr_lambda":       cor_lambda,
            "overlap_p_top_k":   overlap_p,
            "overlap_lam_top_k": overlap_l,
        })

    # Best latent dim by cosine similarity
    best_idx = int(np.argmax([d["cosine_combined"] for d in per_dim]))
    best     = per_dim[best_idx]

    # Null baseline: random direction
    rng = np.random.default_rng(0)
    null_cos = []
    for _ in range(500):
        rand_p   = rng.standard_normal(n_genes)
        rand_lam = rng.standard_normal(n_genes)
        rand_vec = np.concatenate([rand_p, rand_lam])
        null_cos.append(cosine_similarity(true_vec, rand_vec))
    null_mean = float(np.mean(null_cos))
    null_std  = float(np.std(null_cos))

    return {
        "per_dim":       per_dim,
        "best_dim":      best_idx,
        "best_metrics":  best,
        "null_cos_mean": null_mean,
        "null_cos_std":  null_std,
        "z_score":       (best["cosine_combined"] - null_mean) / (null_std + 1e-8),
    }


def compare_latent_to_z_shift(
    screen_result: dict,
    orig_fit: dict,         # hierarchical fit on original data  (has w_hat, z_hat, lambda_hat)
    true_w: np.ndarray,     # (n_genes,) true gene loadings — defines expected shift dir.
    delta: float,           # perturbation magnitude used for the positive-delta case
) -> dict:
    """
    For each latent dimension l perturbed by +delta std, measure whether the
    resulting change in decoded gene expression aligns with the true causal
    loading vector w_g.

    Causal reasoning: if AE latent dim l encodes the z factor, then increasing
    it by delta should push counts UP in genes where w_g > 0 and DOWN where
    w_g < 0.  The test statistic is the cosine similarity between:
      • per-gene mean count shift (decoded_pert - decoded_baseline, per gene)
      • true_w (the causal gene loading vector)

    Note: z_hat from SVD is unit-norm, so z_hat_pert - z_hat_orig ≈ 0 by
    construction.  We therefore work directly in count space, which is
    interpretable and scale-invariant.

    Returns per-dim metrics and the best dim for causal alignment.
    """
    latent_dim  = screen_result["latent_dim"]
    magnitudes  = screen_result["magnitudes"]
    pos_delta   = min(magnitudes, key=lambda d: abs(d - delta))
    neg_delta   = min(magnitudes, key=lambda d: abs(d + delta))

    # Baseline decoded counts (delta=0)
    baseline_counts = screen_result["results"][0].get(0.0, {}).get(
        "decoded_counts",
        screen_result["baseline_fit"].get("decoded_counts", None),
    )
    # Fallback: use delta=0 from any dim (they share the same Z_encoded baseline)
    if baseline_counts is None:
        for l0 in range(latent_dim):
            for mag in magnitudes:
                if abs(mag) < 1e-8:
                    baseline_counts = screen_result["results"][l0].get(mag, {}).get(
                        "decoded_counts", None
                    )
                    if baseline_counts is not None:
                        break
            if baseline_counts is not None:
                break

    per_dim = []
    for l in range(latent_dim):
        entry_pos = screen_result["results"][l].get(pos_delta, {})
        decoded_pos = entry_pos.get("decoded_counts", None)

        if decoded_pos is None or baseline_counts is None:
            per_dim.append({
                "latent_dim":    l,
                "cos_to_true_w": 0.0,
                "mean_count_shift": 0.0,
                "corr_to_true_w": 0.0,
                # Legacy keys kept for backward-compat with print statements
                "mean_z_shift":  0.0,
                "corr_z_shift":  0.0,
            })
            continue

        # Per-gene mean count shift: positive delta perturbation vs baseline
        count_shift = (decoded_pos - baseline_counts).mean(axis=1)   # (n_genes,)
        cos_w  = cosine_similarity(count_shift, true_w)
        corr_w = safe_corr(count_shift, true_w)
        mean_shift = float(count_shift.mean())

        per_dim.append({
            "latent_dim":       l,
            "cos_to_true_w":    cos_w,
            "corr_to_true_w":   corr_w,
            "mean_count_shift": mean_shift,
            # Legacy keys with updated semantics
            "mean_z_shift":     mean_shift,
            "corr_z_shift":     corr_w,
        })

    # Best dim: highest cosine to true_w (signed — we want positive alignment)
    best_idx = int(np.argmax([d["cos_to_true_w"] for d in per_dim]))

    # Null baseline: random direction cosines
    rng = np.random.default_rng(0)
    null_cos = [cosine_similarity(rng.standard_normal(len(true_w)), true_w)
                for _ in range(500)]
    null_mean = float(np.mean(null_cos))
    null_std  = float(np.std(null_cos))

    best_cos = per_dim[best_idx]["cos_to_true_w"]
    z_score  = (best_cos - null_mean) / (null_std + 1e-8)

    return {
        "per_dim":       per_dim,
        "best_dim":      best_idx,
        "null_cos_mean": null_mean,
        "null_cos_std":  null_std,
        "z_score":       z_score,
        "delta_used":    pos_delta,
        # Compatibility alias so hierarchical and simple modes share the same key
        "best_metrics":  {"cosine_combined": best_cos},
    }


def summarise_sensitivity(sensitivity: np.ndarray) -> dict:
    """
    Classify each latent dimension by its dominant PGM effect.

    sensitivity : (latent_dim, 2)   columns = [lambda_slope, p_slope]
    """
    labels = []
    for l in range(len(sensitivity)):
        lam_s = abs(sensitivity[l, 0])
        p_s   = abs(sensitivity[l, 1])
        if lam_s < 1e-4 and p_s < 1e-4:
            label = "inactive"
        elif lam_s > 3 * p_s:
            label = "lambda-dominated"
        elif p_s > 3 * lam_s:
            label = "p-dominated"
        else:
            label = "mixed"
        labels.append(label)
    return {
        "labels":       labels,
        "sensitivity":  sensitivity.tolist(),
    }
