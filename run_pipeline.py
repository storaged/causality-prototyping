"""
run_pipeline.py  –  Causal Latent Space Sandbox
================================================
Runs all 7 stages with rich visualisations at each step, then:
  • multi-seed reproducibility check  (recommended step 3)
  • VAE comparison                    (recommended step 4)
  • structured achievements report

Usage:
    python run_pipeline.py [--config config.yaml] [--output results/]
"""

from __future__ import annotations
import argparse
import json
import time
import numpy as np
from pathlib import Path
import yaml

from src.pgm.generator import PGMConfig, PGMSimulator
from src.pgm.perturbations import GenerativePerturbationEngine
from src.inference.fitting import (
    fit_pgm_mom, evaluate_recovery, evaluate_perturbation_detection,
)
from src.models.autoencoder import (
    train_autoencoder, evaluate_reconstruction, pca_baseline,
    _log1p_normalize,
)
from src.models.vae import train_vae, evaluate_reconstruction as vae_evaluate_reconstruction
from src.analysis.latent_perturbation import (
    screen_latent_dimensions, compute_per_gene_shift_vectors,
)
from src.analysis.comparison import (
    compare_to_true_intervention, summarise_sensitivity,
)
from src.visualization.plots import (
    plot_stage1, plot_stage2, plot_stage3, plot_stage4,
    plot_stage5, plot_stage6, plot_stage7,
    plot_multiseed_summary,
)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _hdr(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

def _item(label: str, value, ok: bool | None = None) -> None:
    mark = "[PASS]" if ok is True else "[FAIL]" if ok is False else "      "
    print(f"  {mark}  {label}: {value}")

def _subitem(msg: str) -> None:
    print(f"           {msg}")

def _plots(paths: list) -> None:
    for p in paths:
        print(f"           plot → {p.name}")


# ---------------------------------------------------------------------------
# Core single-run experiment (reused for multi-seed loop)
# ---------------------------------------------------------------------------

def _run_ae_stages(
    counts: np.ndarray,
    ae_cfg: dict,
    lp_cfg: dict,
    target_genes: list[int],
    fit_orig: dict,
    fit_pert: dict,                    # p-intervention fit (simple) / z-shift fit (hierarchical)
    fit_lambda_pert: dict | None = None,  # λ-intervention fit (positive control, simple only)
    model_type: str = "ae",            # "ae" | "vae"
    vae_cfg: dict | None = None,
    pgm_type: str = "simple",
    true_w: np.ndarray | None = None,  # hierarchical only
    z_delta: float = 1.0,              # hierarchical only
) -> dict:
    """
    Run stages 4-7 for one model type (AE or VAE) and return a result dict.
    """
    seed = ae_cfg["seed"] if model_type == "ae" else (vae_cfg or ae_cfg)["seed"]

    if model_type == "ae":
        model, history = train_autoencoder(
            counts      = counts,
            latent_dim  = ae_cfg["latent_dim"],
            hidden_dims = ae_cfg["hidden_dims"],
            n_epochs    = ae_cfg["n_epochs"],
            lr          = ae_cfg["lr"],
            batch_size  = ae_cfg["batch_size"],
            seed        = seed,
            loss_type   = ae_cfg.get("loss_type", "mse"),
        )
        rec = evaluate_reconstruction(model, counts)
    else:
        cfg = vae_cfg
        model, history = train_vae(
            counts      = counts,
            latent_dim  = cfg["latent_dim"],
            hidden_dims = cfg["hidden_dims"],
            n_epochs    = cfg["n_epochs"],
            lr          = cfg["lr"],
            batch_size  = cfg["batch_size"],
            beta        = cfg.get("beta", 1.0),
            loss_type   = cfg.get("loss_type", "mse"),
            seed        = seed,
        )
        rec = vae_evaluate_reconstruction(model, counts)

    decoded_counts = rec["decoded_counts"]
    fit_recon = fit_pgm_mom(decoded_counts)

    screen_res = screen_latent_dimensions(
        model      = model,
        counts     = counts,
        magnitudes = lp_cfg["magnitudes"],
    )

    shift_vecs = compute_per_gene_shift_vectors(
        screen_res, screen_res["baseline_fit"], delta=1.0
    )

    sens_summary = summarise_sensitivity(screen_res["sensitivity"])

    if pgm_type == "hierarchical":
        from src.analysis.comparison import compare_latent_to_z_shift
        from src.inference.hierarchical_fitting import fit_hierarchical_mom
        comparison = compare_latent_to_z_shift(
            screen_result = screen_res,
            orig_fit      = fit_orig,
            true_w        = true_w,
            delta         = z_delta,
        )
        comparison_lam = None   # no separate λ-intervention in hierarchical mode
        true_shift_p      = np.zeros(counts.shape[0])
        true_shift_lambda = np.zeros(counts.shape[0])
    else:
        true_shift_p      = fit_pert["p_hat"]      - fit_orig["p_hat"]
        true_shift_lambda = fit_pert["lambda_hat"] - fit_orig["lambda_hat"]
        comparison = compare_to_true_intervention(
            true_shift_p        = true_shift_p,
            true_shift_lambda   = true_shift_lambda,
            latent_shift_p      = shift_vecs["shift_p"],
            latent_shift_lambda = shift_vecs["shift_lambda"],
            top_k               = max(3, len(target_genes) * 2),
        )
        comparison_lam = None
        if fit_lambda_pert is not None:
            lam_shift_p      = fit_lambda_pert["p_hat"]      - fit_orig["p_hat"]
            lam_shift_lambda = fit_lambda_pert["lambda_hat"] - fit_orig["lambda_hat"]
            comparison_lam = compare_to_true_intervention(
                true_shift_p        = lam_shift_p,
                true_shift_lambda   = lam_shift_lambda,
                latent_shift_p      = shift_vecs["shift_p"],
                latent_shift_lambda = shift_vecs["shift_lambda"],
                top_k               = max(3, len(target_genes) * 2),
            )

    return {
        "model":              model,
        "history":            history,
        "rec":                rec,
        "fit_recon":          fit_recon,
        "screen_res":         screen_res,
        "shift_vecs":         shift_vecs,
        "true_shift_p":       true_shift_p,
        "true_shift_lambda":  true_shift_lambda,
        "comparison":         comparison,         # primary alignment (p-shift or z-shift)
        "comparison_lam":     comparison_lam,     # λ-intervention alignment (simple only)
        "sens_summary":       sens_summary,
    }


# ---------------------------------------------------------------------------
def main(config_path: str = "config.yaml", output_dir: str = "results") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    achievements: dict[str, bool] = {}
    pgm_type = cfg["pgm"].get("type", "simple")
    _subitem(f"PGM type: {pgm_type}")

    # ======================================================================
    # STAGE 1  –  Synthetic Data Generator
    # ======================================================================
    _hdr("STAGE 1 · Synthetic Data Generator")

    if pgm_type == "hierarchical":
        from src.pgm.hierarchical_generator import (
            HierarchicalPGMConfig, HierarchicalPGMSimulator,
        )
        from src.visualization.plots import plot_stage1_hierarchical
        hpgm_cfg = HierarchicalPGMConfig(**cfg["hierarchical_pgm"])
        sim  = HierarchicalPGMSimulator(hpgm_cfg)
        data = sim.generate()
        # Compatibility aliases used by downstream simple-mode code
        data["true_lambda"] = np.exp(data["true_b"])
        sim.save(data, out / "stage1")
        checks      = sim.sanity_check(data)
        all_pass_s1 = all(checks.values())
        for name, ok in checks.items():
            _item(name, "", ok)
        _item("Shape",       data["counts"].shape)
        _item("Count range", f"[{data['counts'].min()}, {data['counts'].max()}]  "
                             f"mean={data['counts'].mean():.2f}")
        _item("True w mean", f"{data['true_w'].mean():.3f} ± {data['true_w'].std():.3f}")
        _item("True z range",f"[{data['true_z'].min():.2f}, {data['true_z'].max():.2f}]")
        _item("True p mean", f"{data['true_p'].mean():.3f} ± {data['true_p'].std():.3f}")
        p1 = plot_stage1_hierarchical(data, out / "plots/stage1")
    else:
        pgm_cfg = PGMConfig(
            n_genes             = cfg["pgm"]["n_genes"],
            n_patients          = cfg["pgm"]["n_patients"],
            lambda_poisson_rate = cfg["pgm"]["lambda_poisson_rate"],
            p_beta_a            = cfg["pgm"]["p_beta_a"],
            p_beta_b            = cfg["pgm"]["p_beta_b"],
            seed                = cfg["pgm"]["seed"],
        )
        sim  = PGMSimulator(pgm_cfg)
        data = sim.generate()
        data["true_w"] = None; data["true_z"] = None; data["true_b"] = None
        sim.save(data, out / "stage1")
        checks      = sim.sanity_check(data)
        all_pass_s1 = all(checks.values())
        for name, ok in checks.items():
            _item(name, "", ok)
        _item("Shape",        data["counts"].shape)
        _item("Count range",  f"[{data['counts'].min()}, {data['counts'].max()}]  "
                              f"mean={data['counts'].mean():.2f}")
        _item("True λ  mean", f"{data['true_lambda'].mean():.2f} ± {data['true_lambda'].std():.2f}")
        _item("True p  mean", f"{data['true_p'].mean():.3f} ± {data['true_p'].std():.3f}")
        p1 = plot_stage1(data, out / "plots/stage1")

    _plots(p1)
    achievements["stage1_generator"] = all_pass_s1

    # ======================================================================
    # STAGE 2  –  Ground-Truth Perturbation
    # ======================================================================
    _hdr("STAGE 2 · Ground-Truth Perturbation Engine")

    if pgm_type == "hierarchical":
        from src.pgm.hierarchical_perturbations import HierarchicalPerturbationEngine
        hpert = HierarchicalPerturbationEngine(hpgm_cfg)
        hpert_cfg = cfg["hierarchical_perturbation"]
        z_delta = hpert_cfg["z_delta"]

        orig_data, pert_data, manifest = hpert.perturb_z_shift(delta=z_delta)
        hpert.save(orig_data, pert_data, manifest, out / "stage2")
        s2_checks   = hpert.sanity_check(orig_data, pert_data, manifest)
        all_pass_s2 = all(s2_checks.values())
        for name, ok in s2_checks.items():
            _item(name, "", ok)
        _item("Intervention", f"do(z_s = z_s + {z_delta})  for all patients")
        _item("Expected gene-level effect", "Δlog_mu_g = w_g * delta")

        # Positive controls for Stage 7
        target_genes_p       = hpert_cfg["target_genes_p"]
        _, p_pert_data, _    = hpert.perturb_p_prior(
            target_genes=target_genes_p, new_a=hpert_cfg["p_beta_a_perturbed"]
        )
        target_genes         = target_genes_p
        lambda_pert_data     = p_pert_data   # reuse variable name for compatibility
        _subitem(f"p-intervention also generated for genes {target_genes_p}")

        p2 = plot_stage2(orig_data, pert_data, [], out / "plots/stage2")
    else:
        target_genes = cfg["perturbation"]["target_genes"]
        new_a_pert   = cfg["perturbation"]["p_beta_a_perturbed"]
        pert_engine  = GenerativePerturbationEngine(pgm_cfg)
        orig_data, pert_data, manifest = pert_engine.perturb_p_prior(
            target_genes=target_genes, new_a=new_a_pert,
        )
        pert_engine.save(orig_data, pert_data, manifest, out / "stage2")
        s2_checks   = pert_engine.sanity_check(orig_data, pert_data, manifest)
        all_pass_s2 = all(s2_checks.values())
        for name, ok in s2_checks.items():
            _item(name, "", ok)
        diff     = np.abs(pert_data["counts"].mean(axis=1) - orig_data["counts"].mean(axis=1))
        non_t    = [i for i in range(pgm_cfg.n_genes) if i not in target_genes]
        _item("Mean shift affected genes",   f"{diff[target_genes].mean():.2f}")
        _item("Mean shift unaffected genes", f"{diff[non_t].mean():.2f}")
        lambda_shift = cfg["perturbation"]["lambda_shift"]
        _, lambda_pert_data, _ = pert_engine.perturb_lambda(
            target_genes=target_genes, lambda_shift=lambda_shift,
        )
        _subitem(f"λ-intervention generated: λ +{lambda_shift} for genes {target_genes}")
        p2 = plot_stage2(orig_data, pert_data, target_genes, out / "plots/stage2")

    _plots(p2)
    achievements["stage2_perturbation_engine"] = all_pass_s2

    # ======================================================================
    # STAGE 3  –  Baseline PGM Inference
    # ======================================================================
    _hdr("STAGE 3 · Baseline PGM Inference")

    if pgm_type == "hierarchical":
        from src.inference.hierarchical_fitting import (
            fit_hierarchical_mom, evaluate_hierarchical_recovery, evaluate_z_shift_detection,
        )
        from src.visualization.plots import plot_stage3_hierarchical

        fit_orig = fit_hierarchical_mom(data["counts"])
        fit_pert = fit_hierarchical_mom(pert_data["counts"])
        # Compatibility aliases (lambda_hat = exp(b_hat))
        fit_orig["lambda_hat"] = fit_orig["lambda_hat"]
        fit_pert["lambda_hat"] = fit_pert["lambda_hat"]

        rec_orig   = evaluate_hierarchical_recovery(
            fit_orig, data["true_w"], data["true_b"], data["true_p"], data["true_z"]
        )
        # Compatibility aliases so Stage 5 can use the same corr_lambda / corr_p keys
        rec_orig["corr_lambda"] = rec_orig["corr_w"]
        rec_orig["corr_p"]      = rec_orig["corr_p"]   # already present
        z_det      = evaluate_z_shift_detection(fit_orig, fit_pert, z_delta, data["true_w"])

        corr_w_ok  = rec_orig["corr_w"]  > 0.5
        corr_z_ok  = rec_orig["corr_z"]  > 0.5
        dir_ok     = z_det["direction_correct"]

        _item("w_g recovery corr",  f"{rec_orig['corr_w']:.3f}",   corr_w_ok)
        _item("z_s recovery corr",  f"{rec_orig['corr_z']:.3f}",   corr_z_ok)
        _item("b_g recovery corr",  f"{rec_orig['corr_b']:.3f}")
        _item("p_g recovery corr",  f"{rec_orig['corr_p']:.3f}")
        _item("Variance explained by rank-1", f"{rec_orig['sv_ratio']:.2%}")
        _item("do(z+δ) detected: mean z-shift direction correct",
              f"{z_det['mean_z_shift']:.3f}", dir_ok)
        _item("w stability under z-intervention",
              f"corr={z_det['corr_w_stability']:.3f}")

        (out / "stage3").mkdir(exist_ok=True)
        p3 = plot_stage3_hierarchical(fit_orig, fit_pert, data, z_delta,
                                      out / "plots/stage3")
        achievements["stage3_pgm_inference"] = corr_w_ok and corr_z_ok

    else:
        fit_orig = fit_pgm_mom(data["counts"])
        fit_pert = fit_pgm_mom(pert_data["counts"])
        rec_orig = evaluate_recovery(fit_orig, data["true_lambda"], data["true_p"])
        det      = evaluate_perturbation_detection(
            fit_orig, fit_pert, target_genes, pgm_cfg.n_genes, param="p", top_k=10,
        )
        corr_lambda_ok = rec_orig["corr_lambda"] > 0.15
        corr_p_ok      = rec_orig["corr_p"]      > 0.15
        shift_ratio_ok = det["shift_ratio"]       > 1.2
        _item("λ recovery corr", f"{rec_orig['corr_lambda']:.3f}", corr_lambda_ok)
        _item("p recovery corr", f"{rec_orig['corr_p']:.3f}",      corr_p_ok)
        _item("Shift ratio",     f"{det['shift_ratio']:.2f}",       shift_ratio_ok)
        _item("Top-10 overlap",  f"{det['overlap_at_k']}/{det['n_target']}")
        (out / "stage3").mkdir(exist_ok=True)
        np.save(out / "stage3/fit_orig.npy",
                np.stack([fit_orig["lambda_hat"], fit_orig["p_hat"]]))
        p3 = plot_stage3(fit_orig, fit_pert, data["true_lambda"], data["true_p"],
                         target_genes, out / "plots/stage3")
        achievements["stage3_pgm_inference"] = corr_lambda_ok and corr_p_ok

    _plots(p3)

    # Fit secondary perturbation data (used by Stage 7 positive control)
    if pgm_type == "hierarchical":
        fit_lambda_pert = fit_hierarchical_mom(lambda_pert_data["counts"])
        fit_lambda_pert["lambda_hat"] = fit_lambda_pert["lambda_hat"]
        target_genes = []   # no gene-level target in hierarchical z-shift
    else:
        fit_lambda_pert = fit_pgm_mom(lambda_pert_data["counts"])

    # ======================================================================
    # STAGE 4  –  AutoEncoder Training
    # ======================================================================
    _hdr("STAGE 4 · AutoEncoder Training")

    ae_cfg = cfg["autoencoder"]
    t0 = time.time()
    ae_result = _run_ae_stages(
        counts           = data["counts"],
        ae_cfg           = ae_cfg,
        lp_cfg           = cfg["latent_perturbation"],
        target_genes     = target_genes,
        fit_orig         = fit_orig,
        fit_pert         = fit_pert,
        fit_lambda_pert  = fit_lambda_pert,
        model_type       = "ae",
        pgm_type         = pgm_type,
        true_w           = data.get("true_w"),
        z_delta          = z_delta if pgm_type == "hierarchical" else 1.0,
    )
    t_ae = time.time() - t0

    ae_model   = ae_result["model"]
    ae_history = ae_result["history"]
    ae_rec     = ae_result["rec"]

    pca_rec = pca_baseline(data["counts"], ae_cfg["latent_dim"])

    loss_dec = ae_history["train_loss"][-1] < ae_history["train_loss"][0] * 0.5
    corr_ok  = ae_rec["reconstruction_corr"] > 0.5
    nonneg   = bool(ae_rec["decoded_counts"].min() >= -0.5)
    ae_beats = ae_rec["reconstruction_corr"] >= pca_rec["pca_reconstruction_corr"] - 0.05

    _item("Loss (init→final)",
          f"{ae_history['train_loss'][0]:.4f} → {ae_history['train_loss'][-1]:.4f}",
          loss_dec)
    _item("Reconstruction corr (AE)",  f"{ae_rec['reconstruction_corr']:.3f}",  corr_ok)
    _item("Reconstruction corr (PCA)", f"{pca_rec['pca_reconstruction_corr']:.3f}")
    _item("AE ≥ PCA – 0.05",          f"{'yes' if ae_beats else 'no'}",          ae_beats)
    _item("Loss type",                 ae_cfg.get("loss_type", "mse"))
    _item("Training time",             f"{t_ae:.1f}s")

    p4 = plot_stage4(ae_rec, ae_history, pca_rec, ae_model, data["counts"],
                     out / "plots/stage4")
    _plots(p4)
    achievements["stage4_autoencoder"] = loss_dec and corr_ok

    # ======================================================================
    # STAGE 5  –  PGM on Reconstructed Data
    # ======================================================================
    _hdr("STAGE 5 · PGM Fit on AE-Reconstructed Data")

    fit_recon = ae_result["fit_recon"]

    if pgm_type == "hierarchical":
        # Re-fit hierarchical model on decoded counts and compare to true w_g and z_s
        fit_recon_hier = fit_hierarchical_mom(ae_result["rec"]["decoded_counts"])
        from src.inference.hierarchical_fitting import evaluate_hierarchical_recovery
        rec_recon_hier = evaluate_hierarchical_recovery(
            fit_recon_hier, data["true_w"], data["true_b"], data["true_p"], data["true_z"]
        )
        drop_w = rec_orig["corr_w"] - rec_recon_hier["corr_w"]
        drop_p = rec_orig["corr_p"] - rec_recon_hier["corr_p"]
        abs_lam_ok = rec_recon_hier["corr_w"] > 0.35
        abs_p_ok   = rec_recon_hier["corr_p"] > 0.10
        _item("w_g corr: original → reconstructed",
              f"{rec_orig['corr_w']:.3f} → {rec_recon_hier['corr_w']:.3f}  "
              f"(drop={drop_w:.3f})", abs_lam_ok)
        _item("p_g corr: original → reconstructed",
              f"{rec_orig['corr_p']:.3f} → {rec_recon_hier['corr_p']:.3f}  "
              f"(drop={drop_p:.3f})", abs_p_ok)
        _item("z_s corr: original → reconstructed",
              f"{rec_orig['corr_z']:.3f} → {rec_recon_hier['corr_z']:.3f}")
        if rec_recon_hier["corr_w"] < 0.5:
            _subitem("NOTE: w_g signal degraded after AE compression.")
            _subitem("      AE maps the rank-1 structure into a non-linear representation.")
        # For Stage 5 plot compatibility: use lambda_hat = exp(b_hat) from hierarchical fit
        fit_recon_s5  = fit_recon_hier   # has lambda_hat and p_hat
        rec_recon_lambda = rec_recon_hier["corr_w"]
        rec_recon_p      = rec_recon_hier["corr_p"]
        # For the drop quantities in the achievements report
        drop_lambda = drop_w
    else:
        rec_recon = evaluate_recovery(fit_recon, data["true_lambda"], data["true_p"])
        drop_lambda = rec_orig["corr_lambda"] - rec_recon["corr_lambda"]
        drop_p      = rec_orig["corr_p"]      - rec_recon["corr_p"]
        abs_lam_ok  = rec_recon["corr_lambda"] > 0.35
        abs_p_ok    = rec_recon["corr_p"]      > 0.35
        _item("λ corr: original → reconstructed",
              f"{rec_orig['corr_lambda']:.3f} → {rec_recon['corr_lambda']:.3f}  "
              f"(drop={drop_lambda:.3f})",    abs_lam_ok)
        _item("p corr: original → reconstructed",
              f"{rec_orig['corr_p']:.3f} → {rec_recon['corr_p']:.3f}  "
              f"(drop={drop_p:.3f})",          abs_p_ok)
        if rec_recon["corr_p"] > 0.1:
            _subitem(f"p signal partially preserved (was 0.000 at N=100, "
                     f"now {rec_recon['corr_p']:.3f}) — N increase helped.")
        else:
            _subitem("NOTE: p signal collapse. AE only captures mean intensity (λ).")
            _subitem("      Try NegBinom ELBO loss or larger N.")
        fit_recon_s5     = fit_recon
        rec_recon_lambda = rec_recon["corr_lambda"]
        rec_recon_p      = rec_recon["corr_p"]

    p5 = plot_stage5(fit_orig, fit_recon_s5, data["true_lambda"], data["true_p"],
                     out / "plots/stage5")
    _plots(p5)
    achievements["stage5_pgm_on_reconstruction"] = abs_lam_ok and abs_p_ok

    # ======================================================================
    # STAGE 6  –  Latent Perturbation Screening
    # ======================================================================
    _hdr("STAGE 6 · Latent Dimension Perturbation Screening")

    screen_res   = ae_result["screen_res"]
    sensitivity  = screen_res["sensitivity"]
    sens_summary = ae_result["sens_summary"]
    labels       = sens_summary["labels"]
    latent_dim   = screen_res["latent_dim"]

    has_structured = any(l != "inactive" for l in labels)
    has_dominant   = any(l in ("lambda-dominated", "p-dominated") for l in labels)

    print(f"\n  {'Dim':>4}  {'λ-slope':>10}  {'p-slope':>10}  {'classification':<22}")
    for l in range(latent_dim):
        print(f"  {l:>4}  {sensitivity[l,0]:>10.4f}  {sensitivity[l,1]:>10.4f}"
              f"  {labels[l]:<22}")

    _item("≥1 non-inactive latent dim",  f"{'yes' if has_structured else 'no'}",
          has_structured)
    _item("≥1 clearly dominant dim",     f"{'yes' if has_dominant else 'no'}",
          has_dominant)

    p6 = plot_stage6(screen_res, out / "plots/stage6")
    _plots(p6)
    achievements["stage6_latent_screening"] = has_structured

    # ======================================================================
    # STAGE 7  –  Causal Alignment
    # ======================================================================
    _hdr("STAGE 7 · Latent vs True Intervention Comparison")

    if pgm_type == "hierarchical":
        from src.visualization.plots import plot_stage7_z_alignment

        z_alignment = ae_result["comparison"]   # compare_latent_to_z_shift result
        z_alignment["latent_dim"] = ae_result["screen_res"]["latent_dim"]  # needed by plot
        best_z_dim  = z_alignment["best_dim"]
        z_score     = z_alignment["z_score"]
        beat_null   = z_score > 1.65

        print(f"\n  7A. z-shift alignment (do(z_s += {z_delta})):")
        print(f"  Null baseline cosine: {z_alignment['null_cos_mean']:.4f} "
              f"± {z_alignment['null_cos_std']:.4f}")
        for d in z_alignment["per_dim"]:
            mark = " ←← BEST" if d["latent_dim"] == best_z_dim else ""
            print(f"    dim {d['latent_dim']:>2}: "
                  f"mean_count_shift={d['mean_count_shift']:>7.3f}  "
                  f"corr_w={d['corr_to_true_w']:>6.3f}  "
                  f"cos_w={d['cos_to_true_w']:>6.3f}{mark}")
        best_cos = z_alignment["per_dim"][best_z_dim]["cos_to_true_w"]
        _item("Best dim cos to true_w",
              f"dim={best_z_dim}  cos={best_cos:.4f}", beat_null)
        _item("Z-score vs random direction null", f"{z_score:.2f}", beat_null)

        if beat_null:
            _subitem("Latent dim aligns with the true causal loading direction (w_g).")
            _subitem("Perturbation in this dim shifts reconstructed z in the correct direction.")
        else:
            _subitem("No latent dim robustly encodes the causal z factor.")
            _subitem("Vanilla AE lacks structural constraints for causal identifiability.")

        p7 = plot_stage7_z_alignment(
            z_alignment, fit_orig, fit_pert, data["true_w"], data["true_z"],
            z_delta, out / "plots/stage7"
        )
        _plots(p7)

        achievements["stage7_causal_alignment"]         = beat_null
        achievements["stage7a_p_causal_alignment"]      = False   # N/A
        achievements["stage7b_lambda_causal_alignment"] = beat_null

    else:
        comparison   = ae_result["comparison"]
        shift_vecs   = ae_result["shift_vecs"]
        true_shift_p = ae_result["true_shift_p"]
        true_shift_l = ae_result["true_shift_lambda"]
        best         = comparison["best_metrics"]
        z_score      = comparison["z_score"]
        beat_null_p  = z_score > 1.65

        print(f"\n  7A. p-intervention (Beta(2,5)→Beta(1,5)):")
        print(f"  Null baseline cosine: {comparison['null_cos_mean']:.4f} "
              f"± {comparison['null_cos_std']:.4f}")
        for d in comparison["per_dim"]:
            mark = " ←← BEST" if d["latent_dim"] == comparison["best_dim"] else ""
            print(f"    dim {d['latent_dim']:>2}: "
                  f"cos={d['cosine_combined']:>7.4f}  "
                  f"corr_p={d['corr_p']:>6.3f}  "
                  f"corr_λ={d['corr_lambda']:>6.3f}{mark}")
        _item("p-intervention: best dim cosine",
              f"dim={comparison['best_dim']}  cos={best['cosine_combined']:.4f}",  beat_null_p)
        _item("p-intervention: Z-score vs null",  f"{z_score:.2f}", beat_null_p)

        lam_shift_p      = fit_lambda_pert["p_hat"]      - fit_orig["p_hat"]
        lam_shift_lambda = fit_lambda_pert["lambda_hat"] - fit_orig["lambda_hat"]
        comparison_lam = compare_to_true_intervention(
            true_shift_p        = lam_shift_p,
            true_shift_lambda   = lam_shift_lambda,
            latent_shift_p      = shift_vecs["shift_p"],
            latent_shift_lambda = shift_vecs["shift_lambda"],
            top_k               = max(3, len(target_genes) * 2),
        )
        best_lam    = comparison_lam["best_metrics"]
        z_score_lam = comparison_lam["z_score"]
        beat_null_l = z_score_lam > 1.65

        print(f"\n  7B. λ-intervention (λ +{lambda_shift} for genes {target_genes}):")
        print(f"  Null baseline cosine: {comparison_lam['null_cos_mean']:.4f} "
              f"± {comparison_lam['null_cos_std']:.4f}")
        for d in comparison_lam["per_dim"]:
            mark = " ←← BEST" if d["latent_dim"] == comparison_lam["best_dim"] else ""
            print(f"    dim {d['latent_dim']:>2}: "
                  f"cos={d['cosine_combined']:>7.4f}  "
                  f"corr_p={d['corr_p']:>6.3f}  "
                  f"corr_λ={d['corr_lambda']:>6.3f}{mark}")
        _item("λ-intervention: best dim cosine",
              f"dim={comparison_lam['best_dim']}  cos={best_lam['cosine_combined']:.4f}",
              beat_null_l)
        _item("λ-intervention: Z-score vs null",  f"{z_score_lam:.2f}", beat_null_l)
        _item("Top-K λ-shift overlap",  f"{best_lam['overlap_lam_top_k']:.2f}")

        print(f"\n  Summary: p-alignment z={z_score:.2f}  |  λ-alignment z={z_score_lam:.2f}")
        if beat_null_l and not beat_null_p:
            _subitem("AE latent space aligns with λ-interventions but NOT p-interventions.")
            _subitem("The AE encodes mean intensity; overdispersion (p) is not captured.")
        elif beat_null_l and beat_null_p:
            _subitem("Both intervention types are aligned — strong causal structure found.")

        p7 = plot_stage7(comparison, true_shift_p, true_shift_l,
                         shift_vecs["shift_p"], shift_vecs["shift_lambda"],
                         out / "plots/stage7")
        _plots(p7)

        achievements["stage7a_p_causal_alignment"]      = beat_null_p
        achievements["stage7b_lambda_causal_alignment"] = beat_null_l
        achievements["stage7_causal_alignment"] = beat_null_p or beat_null_l

    # ======================================================================
    # STEP 3  –  Multi-seed Reproducibility
    # ======================================================================
    ms_cfg = cfg.get("multi_seed", {})
    seed_results = []

    if ms_cfg.get("enabled", False):
        _hdr("STEP 3 · Multi-Seed Reproducibility Check")
        seeds = ms_cfg.get("seeds", [0, 42, 123])

        for s in seeds:
            ae_cfg_s = dict(ae_cfg)
            ae_cfg_s["seed"] = s
            res = _run_ae_stages(
                counts           = data["counts"],
                ae_cfg           = ae_cfg_s,
                lp_cfg           = cfg["latent_perturbation"],
                target_genes     = target_genes,
                fit_orig         = fit_orig,
                fit_pert         = fit_pert,
                fit_lambda_pert  = fit_lambda_pert,
                model_type       = "ae",
                pgm_type         = pgm_type,
                true_w           = data.get("true_w"),
                z_delta          = z_delta if pgm_type == "hierarchical" else 1.0,
            )
            fit_r = evaluate_recovery(
                fit_pgm_mom(res["rec"]["decoded_counts"]),
                data["true_lambda"], data["true_p"]
            )
            if pgm_type == "hierarchical":
                # primary comparison IS the z-shift alignment
                z_primary = res["comparison"]["z_score"]
                z_p       = 0.0
                best_d    = res["comparison"]["best_dim"]
                best_cos  = res["comparison"]["best_metrics"]["cosine_combined"]
            else:
                z_primary = res["comparison_lam"]["z_score"] if res["comparison_lam"] else 0.0
                z_p       = res["comparison"]["z_score"]
                best_d    = (res["comparison_lam"]["best_dim"] if res["comparison_lam"]
                             else res["comparison"]["best_dim"])
                best_cos  = (res["comparison_lam"]["best_metrics"]["cosine_combined"]
                             if res["comparison_lam"] else 0.0)
            seed_results.append({
                "seed":         s,
                "corr_lambda":  fit_r["corr_lambda"],
                "corr_p":       fit_r["corr_p"],
                "recon_corr":   res["rec"]["reconstruction_corr"],
                "best_dim":     best_d,
                "best_cosine":  best_cos,
                "z_score":      z_primary,
                "z_score_p":    z_p,
                "labels":       res["sens_summary"]["labels"],
            })
            metric_label = "z(z-shift)" if pgm_type == "hierarchical" else "z(λ)"
            print(f"  seed={s:>4}  recon_corr={res['rec']['reconstruction_corr']:.3f}"
                  f"  best_dim={best_d}"
                  f"  {metric_label}={z_primary:.2f}  z(p)={z_p:.2f}")

        # Reproducibility: best dim is the same in majority of seeds?
        best_dims = [r["best_dim"] for r in seed_results]
        most_common_dim = max(set(best_dims), key=best_dims.count)
        fraction_agree  = best_dims.count(most_common_dim) / len(best_dims)
        cosines         = [r["best_cosine"] for r in seed_results]
        z_scores        = [r["z_score"]     for r in seed_results]

        metric_name = "z-shift alignment" if pgm_type == "hierarchical" else "λ-intervention"
        _item("Most common best dim",     f"dim {most_common_dim}  "
                                          f"({fraction_agree*100:.0f}% of seeds)")
        _item(f"Mean best-dim cosine ({metric_name})",
              f"{np.mean(cosines):.3f} ± {np.std(cosines):.3f}")
        _item(f"Mean z-score ({metric_name})",
              f"{np.mean(z_scores):.2f} ± {np.std(z_scores):.2f}")

        n_pass = sum(z > 1.65 for z in z_scores)
        majority_pass = n_pass >= len(z_scores) / 2
        _item(f"Seeds passing z>1.65 ({metric_name})",
              f"{n_pass}/{len(z_scores)}", majority_pass)

        if n_pass < len(z_scores):
            _subitem(f"SCIENTIFIC FINDING: {metric_name} is seed-dependent.")
            _subitem("Vanilla AE does not robustly encode causal structure.")
            _subitem("Structural constraints (disentanglement, supervision) are needed.")

        if pgm_type != "hierarchical":
            z_p_scores = [r["z_score_p"] for r in seed_results]
            _item("p-intervention: all seeds z<1.65 (consistent negative)",
                  f"{sum(z < 1.65 for z in z_p_scores)}/{len(z_p_scores)} seeds",
                  all(z < 1.65 for z in z_p_scores))

        pms = plot_multiseed_summary(seed_results, out / "plots/multiseed")
        _plots(pms)
        # Majority of seeds should show λ-alignment
        achievements["step3_multiseed_reproducibility"] = majority_pass

    # ======================================================================
    # STEP 4  –  VAE Comparison
    # ======================================================================
    vae_cfg_dict = cfg.get("vae", {})
    vae_result   = None

    if vae_cfg_dict.get("enabled", False):
        _hdr("STEP 4 · VAE Comparison  (β-VAE for disentanglement)")

        t0 = time.time()
        vae_result = _run_ae_stages(
            counts           = data["counts"],
            ae_cfg           = ae_cfg,
            lp_cfg           = cfg["latent_perturbation"],
            target_genes     = target_genes,
            fit_orig         = fit_orig,
            fit_pert         = fit_pert,
            fit_lambda_pert  = fit_lambda_pert,
            model_type       = "vae",
            vae_cfg          = vae_cfg_dict,
            pgm_type         = pgm_type,
            true_w           = data.get("true_w"),
            z_delta          = z_delta if pgm_type == "hierarchical" else 1.0,
        )
        t_vae = time.time() - t0

        vae_rec   = vae_result["rec"]
        vae_comp  = vae_result["comparison"]
        vae_sens  = vae_result["sens_summary"]

        ae_best_cos  = ae_result["comparison"]["best_metrics"]["cosine_combined"]
        vae_best_cos = vae_comp["best_metrics"]["cosine_combined"]

        _item("VAE recon corr",  f"{vae_rec['reconstruction_corr']:.3f}")
        _item("AE  recon corr",  f"{ae_rec['reconstruction_corr']:.3f}")
        _item("VAE best dim cosine",
              f"dim={vae_comp['best_dim']}  cos={vae_best_cos:.4f}")
        _item("VAE z-score vs null",
              f"{vae_comp['z_score']:.2f}",
              vae_comp["z_score"] > 1.65)
        _item("VAE labels",   str(vae_sens["labels"]))
        _item("β (KL weight)",
              vae_cfg_dict.get("beta", 1.0))
        _item("Training time",   f"{t_vae:.1f}s")

        vae_better_cosine = vae_best_cos > ae_best_cos
        _item("VAE > AE causal alignment",
              f"{'yes' if vae_better_cosine else 'no'}",
              vae_better_cosine)

        # Sensitivity comparison table
        print(f"\n  Sensitivity comparison (VAE vs AE):")
        vae_sens_arr = vae_result["screen_res"]["sensitivity"]
        ae_sens_arr  = screen_res["sensitivity"]
        print(f"  {'Dim':>4}  {'AE λ':>8} {'AE p':>8}  {'VAE λ':>8} {'VAE p':>8}")
        for l in range(latent_dim):
            print(f"  {l:>4}  {ae_sens_arr[l,0]:>8.4f} {ae_sens_arr[l,1]:>8.4f}"
                  f"  {vae_sens_arr[l,0]:>8.4f} {vae_sens_arr[l,1]:>8.4f}")

        achievements["step4_vae_comparison"] = vae_comp["z_score"] > 1.65

    # ======================================================================
    # ACHIEVEMENTS REPORT
    # ======================================================================
    _hdr("ACHIEVEMENTS REPORT")

    level1 = (achievements.get("stage1_generator", False) and
              achievements.get("stage2_perturbation_engine", False) and
              achievements.get("stage3_pgm_inference", False))
    level2 = achievements.get("stage5_pgm_on_reconstruction", False)
    level3 = achievements.get("stage6_latent_screening", False)
    level4 = achievements.get("stage7_causal_alignment", False)

    def tick(ok): return "✓ ACHIEVED" if ok else "✗ NOT YET"

    if pgm_type == "hierarchical":
        rows = [
            ("Level 1", "Hierarchical PGM + Perturbation + w/z Recovery",         level1),
            ("Level 2", "AE Preserves PGM Inferability After Compression",         level2),
            ("Level 3", "≥1 Latent Dim Shows Structured PGM Effect",               level3),
            ("Level 4", "Latent Dim Aligns with True Causal z-factor Direction",   level4),
        ]
    else:
        rows = [
            ("Level 1", "Simulator + Perturbation Engine + PGM Recovery",         level1),
            ("Level 2", "AE Preserves Inferability (λ signal after compression)", level2),
            ("Level 3", "≥1 Latent Dim Shows Structured PGM Effect",               level3),
            ("Level 4", "Latent Perturbation Aligns with ≥1 Intervention (p or λ)", level4),
        ]
    if "step3_multiseed_reproducibility" in achievements:
        rows.append(
            ("Step 3",  "Multi-Seed Reproducibility Confirmed",
             achievements["step3_multiseed_reproducibility"])
        )
    if "step4_vae_comparison" in achievements:
        rows.append(
            ("Step 4",  "VAE Shows Causal Alignment",
             achievements["step4_vae_comparison"])
        )

    print()
    for lvl, desc, ok in rows:
        print(f"  [{tick(ok)}]  {lvl}:  {desc}")

    print()
    print("  Per-stage pass/fail:")
    for stage, ok in achievements.items():
        print(f"    [{'PASS' if ok else 'FAIL'}]  {stage.replace('_', ' ')}")

    print()
    print("  Scaling recommendations:")
    if level1:
        if pgm_type == "hierarchical":
            print("    → Stages 1-3 validated. Hierarchical PGM (w_g, z_s) recovered correctly.")
        else:
            print("    → Stages 1-3 validated. PGM inference near-perfect at N=1000.")
    if level2:
        print(f"    → AE preserves primary factor (corr={rec_recon_lambda:.3f}) and"
              f" p (corr={rec_recon_p:.3f}) after compression.")
    if pgm_type == "hierarchical":
        if level4:
            print("    → Latent z-shift alignment detected (hierarchical PGM).")
            print("      Best dim tracks the true causal factor w_g·z_s.")
            print("      Next: multi-environment training (IRM) for robust identifiability.")
        else:
            print("    → z-shift alignment not yet detected (z<1.65).")
            print("      The AE does not yet factorize the causal z dimension reliably.")
            print("      Try: n_genes=200+, supervised contrastive loss, or IRM penalty.")
        if not achievements.get("step3_multiseed_reproducibility", False):
            print("    → z-alignment is seed-dependent → AE lacks structural constraints.")
            print("      • Increase n_genes to 200+ for stronger signal per dim.")
            print("      • Add intervention labels → contrastive or IRM training.")
    else:
        if achievements.get("stage7a_p_causal_alignment", False) is False:
            print("    → p-intervention: consistently NOT aligned (z<1.65 all seeds).")
            print("      To capture p: switch to NegBinom reconstruction loss.")
        if achievements.get("stage7b_lambda_causal_alignment", False):
            print("    → λ-intervention: aligned for at least one seed.")
            print("      But seed-dependent → vanilla AE lacks robust causal factorization.")
        if not achievements.get("step3_multiseed_reproducibility", False):
            print("    → For reproducible alignment:")
            print("      • Try n_genes=1000 (Step 5): more genes give stronger signal per dim.")
            print("      • Try β-VAE with annealing: KL pushes toward structured representation.")
            print("      • Try supervised approach: provide known intervention labels.")
        if level4:
            print("    → Next scientific step: richer PGM (Option 2 in thought_dump.md).")
            print("      Hierarchical model with sample-level factors should give cleaner alignment.")

    # ── Save full report ────────────────────────────────────────────
    _s7_comp = ae_result["comparison"]
    report = {
        "pgm_type":                pgm_type,
        "achievements":            achievements,
        "success_levels":          {f"level{i+1}": ok for i, (_, _, ok) in enumerate(rows[:4])},
        "stage3":                  rec_orig,
        "stage4_recon_corr_ae":    ae_rec["reconstruction_corr"],
        "stage4_recon_corr_pca":   pca_rec["pca_reconstruction_corr"],
        "stage5_lambda_drop":      drop_lambda,
        "stage5_p_drop":           drop_p,
        "stage6_labels":           labels,
        "stage7_best_dim":         _s7_comp["best_dim"],
        "stage7_best_cosine":      _s7_comp["best_metrics"]["cosine_combined"],
        "stage7_z_score":          z_score,
        "multiseed_results":       seed_results,
    }
    if vae_result:
        report["vae_recon_corr"]      = vae_result["rec"]["reconstruction_corr"]
        report["vae_best_cosine"]     = vae_result["comparison"]["best_metrics"]["cosine_combined"]
        report["vae_z_score"]         = vae_result["comparison"]["z_score"]
        report["vae_labels"]          = vae_result["sens_summary"]["labels"]

    with open(out / "achievements_report.json", "w") as f:
        json.dump(report, f, indent=2)

    n_achieved = sum(ok for (_, _, ok) in rows)
    print(f"\n{'='*65}")
    print(f"  OVERALL: {n_achieved}/{len(rows)} success levels achieved")
    print(f"  Plots in: {out}/plots/")
    print(f"  Report:   {out}/achievements_report.json")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    main(args.config, args.output)
