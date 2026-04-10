"""
Visualisation module — one function per pipeline stage.
All functions save PNGs to a given output directory and return the list of paths.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats

# ------------------------------------------------------------------
# Shared style
# ------------------------------------------------------------------
STYLE   = "seaborn-v0_8-whitegrid"
PALETTE = sns.color_palette("tab10")
TARGET_COLOR   = PALETTE[1]   # orange
NONTARGET_COLOR = PALETTE[0]  # blue
AE_COLOR       = PALETTE[2]   # green
PCA_COLOR      = PALETTE[3]   # red
VAE_COLOR      = PALETTE[4]   # purple

DPI = 120

def _save(fig, path: Path, name: str) -> Path:
    p = path / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# ==================================================================
# STAGE 1  –  Data Generator
# ==================================================================

def plot_stage1(data: dict, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    counts      = data["counts"]          # (n_genes, n_patients)
    true_lambda = data["true_lambda"]
    true_p      = data["true_p"]
    n_genes, n_patients = counts.shape

    # ── 1a. Count matrix heatmap ──────────────────────────────────
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(min(12, n_patients * 0.4 + 2),
                                        min(10, n_genes * 0.18 + 1)))
        im = ax.imshow(np.log1p(counts), aspect="auto", cmap="YlOrRd")
        plt.colorbar(im, ax=ax, label="log1p(count)")
        ax.set_xlabel("Patient index")
        ax.set_ylabel("Gene index")
        ax.set_title(f"Count matrix  ({n_genes} genes × {n_patients} patients)  —  log1p scale")
        paths.append(_save(fig, out, "s1a_count_matrix_heatmap.png"))

    # ── 1b. True parameter histograms with theoretical overlays ───
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # lambda
        ax = axes[0]
        ax.hist(true_lambda, bins=15, density=True, color=PALETTE[0],
                alpha=0.7, edgecolor="white", label="Sampled λ")
        rate = data["config"]["lambda_poisson_rate"]
        x = np.arange(0, int(true_lambda.max()) + 2)
        ax.plot(x, stats.poisson.pmf(x, rate), "o-", color=PALETTE[1],
                ms=4, label=f"Poisson({rate:.0f}) pmf")
        ax.set_xlabel("λᵢ  (NegBinom n-parameter)")
        ax.set_ylabel("Density")
        ax.set_title("True λ distribution")
        ax.legend(fontsize=8)

        # p
        ax = axes[1]
        ax.hist(true_p, bins=20, density=True, color=PALETTE[2],
                alpha=0.7, edgecolor="white", label="Sampled p")
        a, b = data["config"]["p_beta_a"], data["config"]["p_beta_b"]
        x = np.linspace(0.001, 0.999, 300)
        ax.plot(x, stats.beta.pdf(x, a, b), color=PALETTE[3],
                lw=2, label=f"Beta({a},{b}) pdf")
        ax.set_xlabel("pᵢ  (NegBinom success-prob)")
        ax.set_ylabel("Density")
        ax.set_title("True p distribution")
        ax.legend(fontsize=8)

        fig.suptitle("Stage 1 — PGM gene parameters", fontsize=11, y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s1b_gene_param_histograms.png"))

    # ── 1c. Empirical mean vs variance (with NegBinom theory line) ─
    with plt.style.context(STYLE):
        gene_mean = counts.mean(axis=1)
        gene_var  = counts.var(axis=1, ddof=1)

        # Theoretical: var = mean / p  (NegBinom)
        theory_p_range = np.linspace(0.05, 0.95, 300)
        theory_mean    = np.linspace(gene_mean.min(), gene_mean.max(), 300)
        theory_var_lo  = theory_mean / np.percentile(true_p, 75)
        theory_var_hi  = theory_mean / np.percentile(true_p, 25)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(gene_mean, gene_var, s=30, alpha=0.6, color=PALETTE[0], label="Genes (empirical)")
        ax.fill_between(theory_mean, theory_var_lo, theory_var_hi,
                        alpha=0.2, color=PALETTE[1], label="NegBinom theory band (IQR of p)")
        ax.plot([0, gene_mean.max()], [0, gene_mean.max()], "k--", lw=1, label="Var = Mean (Poisson)")
        ax.set_xlabel("Empirical mean per gene")
        ax.set_ylabel("Empirical variance per gene")
        ax.set_title("Mean–Variance relationship\n(overdispersion expected)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        paths.append(_save(fig, out, "s1c_mean_variance.png"))

    # ── 1d. Per-patient count distribution (box-plot summary) ───────
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(max(6, n_patients * 0.35), 4))
        ax.boxplot([counts[:, n] for n in range(n_patients)],
                   notch=False, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=PALETTE[0], alpha=0.5))
        ax.set_xlabel("Patient index")
        ax.set_ylabel("Count")
        ax.set_title("Per-patient count distribution across all genes")
        fig.tight_layout()
        paths.append(_save(fig, out, "s1d_patient_count_distributions.png"))

    return paths


# ==================================================================
# STAGE 2  –  Ground-Truth Perturbation
# ==================================================================

def plot_stage2(orig_data: dict, pert_data: dict,
                target_genes: list[int], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    n_genes = orig_data["counts"].shape[0]
    non_target = [i for i in range(n_genes) if i not in target_genes]

    orig_mean = orig_data["counts"].mean(axis=1)
    pert_mean = pert_data["counts"].mean(axis=1)
    colors = [TARGET_COLOR if i in target_genes else NONTARGET_COLOR for i in range(n_genes)]

    # ── 2a. Per-gene mean shift bar chart ─────────────────────────
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        for ax, vals, title in [
            (axes[0], orig_mean, "Original"),
            (axes[1], pert_mean, "Perturbed"),
        ]:
            ax.bar(range(n_genes), vals, color=colors, edgecolor="none", alpha=0.8)
            for g in target_genes:
                ax.axvline(g, color=TARGET_COLOR, lw=1.5, alpha=0.4)
            ax.set_xlabel("Gene index")
            ax.set_ylabel("Mean count")
            ax.set_title(f"{title} — per-gene mean count")

        legend_els = [Patch(facecolor=TARGET_COLOR, label=f"Target genes {target_genes}"),
                      Patch(facecolor=NONTARGET_COLOR, label="Non-target genes")]
        axes[1].legend(handles=legend_els, fontsize=8, loc="upper right")
        fig.suptitle("Stage 2 — Effect of ground-truth p-prior intervention", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s2a_per_gene_mean_shift.png"))

    # ── 2b. Target gene count distributions before/after ─────────
    with plt.style.context(STYLE):
        n_target = len(target_genes)
        if n_target == 0:
            return paths   # no per-gene targets (e.g. hierarchical z-shift mode)
        fig, axes = plt.subplots(1, n_target, figsize=(4 * n_target, 4), squeeze=False)

        for idx, g in enumerate(target_genes):
            ax = axes[0, idx]
            orig_counts_g = orig_data["counts"][g, :]
            pert_counts_g = pert_data["counts"][g, :]
            all_max = max(orig_counts_g.max(), pert_counts_g.max())
            bins = np.linspace(0, all_max + 1, 20)
            ax.hist(orig_counts_g, bins=bins, alpha=0.6, color=NONTARGET_COLOR,
                    label="Original", density=True)
            ax.hist(pert_counts_g, bins=bins, alpha=0.6, color=TARGET_COLOR,
                    label="Perturbed", density=True)
            ax.set_title(f"Gene {g}")
            ax.set_xlabel("Count")
            ax.legend(fontsize=7)
            orig_m = np.mean(orig_counts_g)
            pert_m = np.mean(pert_counts_g)
            ax.text(0.97, 0.95, f"orig μ={orig_m:.1f}\npert μ={pert_m:.1f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        fig.suptitle("Stage 2 — Target gene count distributions (do(p))", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s2b_target_gene_distributions.png"))

    return paths


# ==================================================================
# STAGE 3  –  PGM Inference
# ==================================================================

def plot_stage3(fit_orig: dict, fit_pert: dict,
                true_lambda: np.ndarray, true_p: np.ndarray,
                target_genes: list[int], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    n_genes = len(true_lambda)

    # ── 3a. Parameter recovery scatter plots ─────────────────────
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        for ax, true_vals, hat_vals, name, unit in [
            (axes[0], true_lambda, fit_orig["lambda_hat"], "λ (NegBinom n)", "counts"),
            (axes[1], true_p,      fit_orig["p_hat"],      "p (success prob)", ""),
        ]:
            corr = float(np.corrcoef(true_vals, hat_vals)[0, 1]) if np.std(true_vals) > 1e-8 else 0
            ax.scatter(true_vals, hat_vals, s=30, alpha=0.7, color=PALETTE[0])
            mn = min(true_vals.min(), hat_vals.min())
            mx = max(true_vals.max(), hat_vals.max())
            ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="y=x (perfect)")
            ax.set_xlabel(f"True {name}")
            ax.set_ylabel(f"Inferred {name}")
            ax.set_title(f"{name} recovery  (corr={corr:.3f})")
            ax.legend(fontsize=8)

        fig.suptitle("Stage 3 — MOM parameter recovery on original data", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s3a_parameter_recovery.png"))

    # ── 3b. Perturbation detection — shift magnitudes ────────────
    with plt.style.context(STYLE):
        shift_p      = np.abs(fit_pert["p_hat"]      - fit_orig["p_hat"])
        shift_lambda = np.abs(fit_pert["lambda_hat"] - fit_orig["lambda_hat"])
        colors = [TARGET_COLOR if i in target_genes else NONTARGET_COLOR for i in range(n_genes)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, shifts, name in [
            (axes[0], shift_p,      "|Δ p̂|"),
            (axes[1], shift_lambda, "|Δ λ̂|"),
        ]:
            ax.bar(range(n_genes), shifts, color=colors, edgecolor="none", alpha=0.8)
            ax.set_xlabel("Gene index")
            ax.set_ylabel(name)
            ax.set_title(f"Perturbation detection: {name} per gene")
        legend_els = [Patch(facecolor=TARGET_COLOR, label=f"Targets {target_genes}"),
                      Patch(facecolor=NONTARGET_COLOR, label="Non-targets")]
        axes[0].legend(handles=legend_els, fontsize=8)
        fig.suptitle("Stage 3 — Do ground-truth perturbations show up in inferred parameters?",
                     y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s3b_perturbation_detection.png"))

    # ── 3c. Inferred p: original vs perturbed (scatter per gene) ─
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = [TARGET_COLOR if i in target_genes else NONTARGET_COLOR for i in range(n_genes)]
        ax.scatter(fit_orig["p_hat"], fit_pert["p_hat"], c=colors, s=40, alpha=0.8)
        mn = min(fit_orig["p_hat"].min(), fit_pert["p_hat"].min())
        mx = max(fit_orig["p_hat"].max(), fit_pert["p_hat"].max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="No change")
        legend_els = [Patch(facecolor=TARGET_COLOR, label="Target genes"),
                      Patch(facecolor=NONTARGET_COLOR, label="Non-target")]
        ax.legend(handles=legend_els, fontsize=8)
        ax.set_xlabel("p̂ (original data)")
        ax.set_ylabel("p̂ (perturbed data)")
        ax.set_title("Inferred p: original vs perturbed\n(targets should shift)")
        fig.tight_layout()
        paths.append(_save(fig, out, "s3c_p_hat_original_vs_perturbed.png"))

    return paths


# ==================================================================
# STAGE 4  –  AutoEncoder
# ==================================================================

def plot_stage4(ae_rec: dict, ae_history: dict, pca_rec: dict,
                model, counts: np.ndarray, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    from src.models.autoencoder import _log1p_normalize

    # ── 4a. Training loss curve ───────────────────────────────────
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(ae_history["train_loss"], color=AE_COLOR, lw=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reconstruction Loss")
        ax.set_title(f"AE training curve  "
                     f"(final={ae_history['train_loss'][-1]:.4f})")
        ax.set_yscale("log")
        fig.tight_layout()
        paths.append(_save(fig, out, "s4a_training_curve.png"))

    # ── 4b. Original vs reconstructed counts scatter ──────────────
    with plt.style.context(STYLE):
        decoded = ae_rec["decoded_counts"]       # (n_genes, n_patients)
        orig_flat = counts.ravel()
        rec_flat  = decoded.ravel()

        # Sample up to 2000 points to avoid overplotting
        rng = np.random.default_rng(0)
        idx = rng.choice(len(orig_flat), size=min(2000, len(orig_flat)), replace=False)

        corr = float(np.corrcoef(orig_flat, rec_flat)[0, 1]) if np.std(orig_flat) > 1e-8 else 0

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ax = axes[0]
        ax.scatter(orig_flat[idx], rec_flat[idx], s=8, alpha=0.3, color=AE_COLOR)
        mn, mx = min(orig_flat.min(), rec_flat.min()), max(orig_flat.max(), rec_flat.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set_xlabel("Original count")
        ax.set_ylabel("Reconstructed count")
        ax.set_title(f"AE: original vs reconstructed  (corr={corr:.3f})")

        ax = axes[1]
        # Same in log1p scale for visual clarity
        ax.scatter(np.log1p(orig_flat[idx]), np.log1p(rec_flat[idx]),
                   s=8, alpha=0.3, color=AE_COLOR, label="AE")
        # PCA comparison
        pca_corr = pca_rec.get("pca_reconstruction_corr", None)
        ax.set_xlabel("log1p(original count)")
        ax.set_ylabel("log1p(reconstructed count)")
        ax.set_title(f"log1p scale  (AE corr={corr:.3f},  PCA corr={pca_corr:.3f})")
        ax.plot([0, np.log1p(mx)], [0, np.log1p(mx)], "r--", lw=1, label="y=x")
        ax.legend(fontsize=8)

        fig.suptitle("Stage 4 — AE reconstruction quality", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s4b_reconstruction_scatter.png"))

    # ── 4c. Latent space visualisation ────────────────────────────
    with plt.style.context(STYLE):
        X = _log1p_normalize(counts).T                   # (n_patients, n_genes)
        Z = model.encode(X)                              # (n_patients, latent_dim)
        latent_dim = Z.shape[1]
        patient_means = counts.mean(axis=0)              # colour proxy

        if latent_dim == 2:
            fig, ax = plt.subplots(figsize=(5, 5))
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=patient_means, cmap="viridis", s=60)
            plt.colorbar(sc, ax=ax, label="Patient mean count")
            ax.set_xlabel("Latent dim 0")
            ax.set_ylabel("Latent dim 1")
            ax.set_title("Latent space (coloured by patient mean count)")
        else:
            # Pairwise scatter grid for first 4 dims
            n_show = min(latent_dim, 4)
            fig, axes = plt.subplots(n_show, n_show,
                                     figsize=(3 * n_show, 3 * n_show))
            for i in range(n_show):
                for j in range(n_show):
                    ax = axes[i][j]
                    if i == j:
                        ax.hist(Z[:, i], bins=12, color=PALETTE[0], alpha=0.7)
                        ax.set_title(f"dim {i}", fontsize=8)
                    else:
                        sc = ax.scatter(Z[:, j], Z[:, i], c=patient_means,
                                        cmap="viridis", s=20, alpha=0.7)
                    if i == n_show - 1:
                        ax.set_xlabel(f"dim {j}", fontsize=7)
                    if j == 0:
                        ax.set_ylabel(f"dim {i}", fontsize=7)
            fig.suptitle("Latent space pairwise scatter (coloured by patient mean count)",
                         fontsize=10)

        fig.tight_layout()
        paths.append(_save(fig, out, "s4c_latent_space.png"))

    # ── 4d. AE vs PCA reconstruction comparison (bar chart) ──────
    with plt.style.context(STYLE):
        labels = ["AE (MSE/Poisson)", "PCA"]
        values = [ae_rec["reconstruction_corr"], pca_rec["pca_reconstruction_corr"]]
        colors_bar = [AE_COLOR, PCA_COLOR]

        fig, ax = plt.subplots(figsize=(4, 3.5))
        bars = ax.bar(labels, values, color=colors_bar, edgecolor="white", alpha=0.8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Reconstruction correlation")
        ax.set_title(f"AE vs PCA  (latent dim={Z.shape[1]})")
        fig.tight_layout()
        paths.append(_save(fig, out, "s4d_ae_vs_pca.png"))

    return paths


# ==================================================================
# STAGE 5  –  PGM on Reconstructed Data
# ==================================================================

def plot_stage5(fit_orig: dict, fit_recon: dict,
                true_lambda: np.ndarray, true_p: np.ndarray,
                out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))

        pairs = [
            (axes[0, 0], true_lambda, fit_orig["lambda_hat"],  "True λ", "λ̂ (original data)"),
            (axes[0, 1], true_lambda, fit_recon["lambda_hat"], "True λ", "λ̂ (reconstructed data)"),
            (axes[1, 0], true_p,      fit_orig["p_hat"],       "True p", "p̂ (original data)"),
            (axes[1, 1], true_p,      fit_recon["p_hat"],      "True p", "p̂ (reconstructed data)"),
        ]
        for ax, xv, yv, xlabel, ylabel in pairs:
            corr = float(np.corrcoef(xv, yv)[0, 1]) if np.std(xv) > 1e-8 else 0
            ax.scatter(xv, yv, s=25, alpha=0.6, color=PALETTE[0])
            mn = min(xv.min(), yv.min()); mx = max(xv.max(), yv.max())
            ax.plot([mn, mx], [mn, mx], "r--", lw=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"corr = {corr:.3f}")

        fig.suptitle("Stage 5 — Does AE compression preserve PGM inferability?\n"
                     "(left = fit on original data, right = fit on AE reconstruction)",
                     fontsize=10, y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s5_pgm_fit_comparison.png"))

    # ── Inferred params: original vs reconstructed scatter ────────
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, key, name in [
            (axes[0], "lambda_hat", "λ̂"),
            (axes[1], "p_hat",      "p̂"),
        ]:
            orig_v = fit_orig[key]
            recon_v = fit_recon[key]
            corr = float(np.corrcoef(orig_v, recon_v)[0, 1]) if np.std(orig_v) > 1e-8 else 0
            ax.scatter(orig_v, recon_v, s=30, alpha=0.7, color=PALETTE[2])
            mn = min(orig_v.min(), recon_v.min())
            mx = max(orig_v.max(), recon_v.max())
            ax.plot([mn, mx], [mn, mx], "r--", lw=1)
            ax.set_xlabel(f"{name} (original data)")
            ax.set_ylabel(f"{name} (reconstructed data)")
            ax.set_title(f"Inferred {name}: original vs AE-reconstructed  (corr={corr:.3f})")
        fig.suptitle("Stage 5 — Parameter consistency through AE bottleneck", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s5b_param_consistency.png"))

    return paths


# ==================================================================
# STAGE 6  –  Latent Perturbation Screening
# ==================================================================

def plot_stage6(screen_res: dict, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    sensitivity = screen_res["sensitivity"]   # (latent_dim, 2)
    results     = screen_res["results"]
    magnitudes  = screen_res["magnitudes"]
    latent_dim  = screen_res["latent_dim"]

    # ── 6a. Sensitivity heatmap ───────────────────────────────────
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(5, max(3, latent_dim * 0.7)))
        vmax = np.abs(sensitivity).max() + 0.01
        sns.heatmap(sensitivity, annot=True, fmt=".3f",
                    cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    xticklabels=["λ slope", "p slope"],
                    yticklabels=[f"dim {l}" for l in range(latent_dim)],
                    ax=ax, linewidths=0.5)
        ax.set_title("Latent dim sensitivity\n(slope of mean-param vs perturbation δ)")
        fig.tight_layout()
        paths.append(_save(fig, out, "s6a_sensitivity_heatmap.png"))

    # ── 6b. Sensitivity curves: mean_param vs delta ───────────────
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, param_key, param_label in [
            (axes[0], "mean_lambda", "Mean inferred λ"),
            (axes[1], "mean_p",      "Mean inferred p"),
        ]:
            for l in range(latent_dim):
                vals = [results[l][d][param_key] for d in magnitudes]
                ax.plot(magnitudes, vals, "o-", label=f"dim {l}",
                        color=PALETTE[l % len(PALETTE)], ms=5)
            ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
            ax.set_xlabel("Perturbation magnitude (δ × std_l)")
            ax.set_ylabel(param_label)
            ax.set_title(f"Sensitivity curve — {param_label}")
            ax.legend(fontsize=8, ncol=2)

        fig.suptitle("Stage 6 — How does each latent dim affect inferred PGM parameters?",
                     y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s6b_sensitivity_curves.png"))

    # ── 6c. Per-gene shift magnitude at δ=+1 for each latent dim ─
    with plt.style.context(STYLE):
        from src.analysis.latent_perturbation import compute_per_gene_shift_vectors
        baseline_fit = screen_res["baseline_fit"]
        shift_vecs   = compute_per_gene_shift_vectors(screen_res, baseline_fit, delta=1.0)

        n_genes = shift_vecs["shift_lambda"].shape[1]
        fig, axes = plt.subplots(latent_dim, 2,
                                 figsize=(12, max(4, latent_dim * 2.5)),
                                 squeeze=False)
        for l in range(latent_dim):
            for ax, shifts, name in [
                (axes[l, 0], shift_vecs["shift_lambda"][l], "Δλ̂"),
                (axes[l, 1], shift_vecs["shift_p"][l],      "Δp̂"),
            ]:
                colors = [PALETTE[l % len(PALETTE)]] * n_genes
                ax.bar(range(n_genes), shifts, color=colors, edgecolor="none", alpha=0.7)
                ax.axhline(0, color="k", lw=0.8)
                ax.set_ylabel(name)
                if l == 0:
                    ax.set_title(f"{name} per gene")
                ax.text(0.02, 0.93, f"dim {l}", transform=ax.transAxes,
                        fontsize=9, fontweight="bold")

        axes[-1, 0].set_xlabel("Gene index")
        axes[-1, 1].set_xlabel("Gene index")
        fig.suptitle("Stage 6 — Per-gene parameter shift at δ=+1 std  for each latent dim",
                     y=1.01)
        fig.tight_layout()
        paths.append(_save(fig, out, "s6c_per_gene_shifts.png"))

    return paths


# ==================================================================
# STAGE 7  –  Causal Alignment
# ==================================================================

def plot_stage7(comparison: dict, true_shift_p: np.ndarray,
                true_shift_lambda: np.ndarray,
                latent_shift_p: np.ndarray,
                latent_shift_lambda: np.ndarray,
                out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    latent_dim = latent_shift_p.shape[0]

    # ── 7a. Cosine similarity bar chart ───────────────────────────
    with plt.style.context(STYLE):
        cosines = [d["cosine_combined"] for d in comparison["per_dim"]]
        best    = comparison["best_dim"]
        colors  = [TARGET_COLOR if l == best else PALETTE[0] for l in range(latent_dim)]

        fig, ax = plt.subplots(figsize=(max(5, latent_dim * 1.2), 4))
        bars = ax.bar([f"dim {l}" for l in range(latent_dim)], cosines,
                      color=colors, edgecolor="white", alpha=0.85)
        ax.axhline(comparison["null_cos_mean"], color="k", lw=1.5, ls="--",
                   label=f"Random-direction null = {comparison['null_cos_mean']:.3f}")
        ax.axhline(comparison["null_cos_mean"] + 1.65 * comparison["null_cos_std"],
                   color="red", lw=1, ls=":", label="Null + 1.65σ  (one-sided p<0.05)")
        ax.set_ylabel("Cosine similarity to true intervention")
        ax.set_title(f"Stage 7 — Which latent dim best mimics the true causal intervention?\n"
                     f"(z-score of best={comparison['z_score']:.2f})")
        ax.legend(fontsize=8)
        for bar, v in zip(bars, cosines):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.005 if v >= 0 else v - 0.02,
                    f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        fig.tight_layout()
        paths.append(_save(fig, out, "s7a_cosine_similarity.png"))

    # ── 7b. Best-dim latent shift vs true shift (scatter per gene) ─
    with plt.style.context(STYLE):
        bd = comparison["best_dim"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, true_s, lat_s, name in [
            (axes[0], true_shift_p,      latent_shift_p[bd],      "Δp̂"),
            (axes[1], true_shift_lambda, latent_shift_lambda[bd], "Δλ̂"),
        ]:
            corr = float(np.corrcoef(true_s, lat_s)[0, 1]) if np.std(true_s) > 1e-8 else 0
            ax.scatter(true_s, lat_s, s=30, alpha=0.7, color=TARGET_COLOR)
            mn = min(true_s.min(), lat_s.min())
            mx = max(true_s.max(), lat_s.max())
            ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5)
            ax.axhline(0, color="k", lw=0.5, alpha=0.3)
            ax.axvline(0, color="k", lw=0.5, alpha=0.3)
            ax.set_xlabel(f"True intervention shift ({name})")
            ax.set_ylabel(f"Best latent-dim shift (dim {bd}, {name})")
            ax.set_title(f"{name} alignment  (corr={corr:.3f})")
        fig.suptitle(f"Stage 7 — True vs latent-perturbation per-gene shifts\n"
                     f"(best latent dim = {bd})", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s7b_shift_alignment_scatter.png"))

    # ── 7c. Heatmap of latent shift vectors across all dims ───────
    with plt.style.context(STYLE):
        n_genes = latent_shift_p.shape[1]
        # Show combined (p + lambda) shift heatmap
        combined = np.vstack([latent_shift_p, latent_shift_lambda])
        row_labels = ([f"dim {l} (Δp̂)"   for l in range(latent_dim)] +
                      [f"dim {l} (Δλ̂)"  for l in range(latent_dim)])
        # Add true intervention rows
        combined_with_true = np.vstack([
            true_shift_p[None, :],
            true_shift_lambda[None, :],
            combined,
        ])
        row_labels_full = ["TRUE Δp̂", "TRUE Δλ̂"] + row_labels

        fig, ax = plt.subplots(figsize=(min(20, n_genes * 0.3 + 2),
                                        max(4, len(row_labels_full) * 0.6)))
        vmax = np.abs(combined_with_true).max() * 0.9
        sns.heatmap(combined_with_true, cmap="RdBu_r", center=0,
                    vmin=-vmax, vmax=vmax,
                    yticklabels=row_labels_full,
                    xticklabels=[str(i) if i % 10 == 0 else "" for i in range(n_genes)],
                    ax=ax, linewidths=0)
        ax.set_xlabel("Gene index")
        ax.set_title("Stage 7 — Shift vectors: true intervention (top) vs all latent dims")
        fig.tight_layout()
        paths.append(_save(fig, out, "s7c_shift_vector_heatmap.png"))

    return paths


# ==================================================================
# MULTI-SEED SUMMARY
# ==================================================================

def plot_multiseed_summary(seed_results: list[dict], out: Path) -> list[Path]:
    """
    seed_results: list of dicts, one per seed, each with keys:
      seed, corr_lambda, corr_p, recon_corr, best_dim, best_cosine, z_score,
      sensitivity_labels
    """
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    seeds = [r["seed"] for r in seed_results]
    n_seeds = len(seeds)

    # ── Metric consistency bar chart ─────────────────────────────
    with plt.style.context(STYLE):
        metrics = ["corr_lambda", "corr_p", "recon_corr", "best_cosine", "z_score"]
        labels  = ["λ recovery\ncorr", "p recovery\ncorr",
                   "AE recon\ncorr", "Best dim\ncosine", "Causal\nz-score"]
        n_m = len(metrics)

        fig, axes = plt.subplots(1, n_m, figsize=(3 * n_m, 4))
        for ax, m, lab in zip(axes, metrics, labels):
            vals = [r[m] for r in seed_results]
            ax.bar(range(n_seeds), vals,
                   color=[PALETTE[i % len(PALETTE)] for i in range(n_seeds)],
                   edgecolor="white", alpha=0.8)
            ax.axhline(np.mean(vals), color="k", lw=1.5, ls="--",
                       label=f"mean={np.mean(vals):.3f}")
            ax.set_xticks(range(n_seeds))
            ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=8)
            ax.set_title(lab, fontsize=9)
            ax.legend(fontsize=7)

        fig.suptitle("Multi-seed reproducibility — key metrics across seeds", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "ms_metric_consistency.png"))

    # ── Best dim consistency table ────────────────────────────────
    with plt.style.context(STYLE):
        best_dims  = [r["best_dim"]  for r in seed_results]
        z_scores   = [r["z_score"]   for r in seed_results]
        cosines    = [r["best_cosine"] for r in seed_results]
        fig, ax = plt.subplots(figsize=(6, max(3, n_seeds * 0.8)))
        ax.axis("off")
        table_data = [[f"seed {r['seed']}", f"dim {r['best_dim']}",
                       f"{r['best_cosine']:.3f}", f"{r['z_score']:.2f}"]
                      for r in seed_results]
        table_data.append(["mean", "—",
                           f"{np.mean(cosines):.3f} ± {np.std(cosines):.3f}",
                           f"{np.mean(z_scores):.2f} ± {np.std(z_scores):.2f}"])
        tbl = ax.table(cellText=table_data,
                       colLabels=["Seed", "Best latent dim", "Cosine sim", "Z-score"],
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
        ax.set_title("Multi-seed: best causal latent dimension", pad=20)
        fig.tight_layout()
        paths.append(_save(fig, out, "ms_best_dim_table.png"))

    return paths


# ==================================================================
# HIERARCHICAL PGM — STAGE 1
# ==================================================================

def plot_stage1_hierarchical(data: dict, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    counts   = data["counts"]
    true_w   = data["true_w"]
    true_b   = data["true_b"]
    true_p   = data["true_p"]
    true_z   = data["true_z"]
    log_mu   = data["true_log_mu"]
    cfg      = data["config"]
    n_genes, n_patients = counts.shape

    # ── H-1a. Count matrix heatmap (same as simple) ───────────────
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(min(12, n_patients * 0.012 + 2),
                                        min(10, n_genes * 0.18 + 1)))
        # Sort patients by true_z so the rank-1 structure is visible
        order = np.argsort(true_z)
        im = ax.imshow(np.log1p(counts[:, order]), aspect="auto", cmap="YlOrRd")
        plt.colorbar(im, ax=ax, label="log1p(count)")
        ax.set_xlabel("Patient (sorted by true z_s)")
        ax.set_ylabel("Gene index")
        ax.set_title(f"Count matrix sorted by causal factor z_s  "
                     f"({n_genes} genes × {n_patients} patients)")
        paths.append(_save(fig, out, "s1a_count_matrix_sorted_by_z.png"))

    # ── H-1b. True log_mu heatmap — shows rank-1 causal structure ─
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(min(12, n_patients * 0.012 + 2),
                                        min(10, n_genes * 0.18 + 1)))
        order = np.argsort(true_z)
        gene_order = np.argsort(true_w)
        im = ax.imshow(log_mu[np.ix_(gene_order, order)],
                       aspect="auto", cmap="RdBu_r")
        plt.colorbar(im, ax=ax, label="log μ_{g,s}")
        ax.set_xlabel("Patient (sorted by z_s)")
        ax.set_ylabel("Gene (sorted by loading w_g)")
        ax.set_title("True log-mean matrix  — rank-1 structure = causal signal")
        paths.append(_save(fig, out, "s1b_log_mu_heatmap.png"))

    # ── H-1c. Gene parameter distributions ───────────────────────
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        # w_g
        ax = axes[0]
        ax.hist(true_w, bins=20, density=True, color=PALETTE[0], alpha=0.7,
                edgecolor="white", label="Sampled w_g")
        x = np.linspace(true_w.min() - 0.5, true_w.max() + 0.5, 200)
        ax.plot(x, stats.norm.pdf(x, 0, cfg["loading_std"]),
                color=PALETTE[1], lw=2, label=f"N(0,{cfg['loading_std']}) pdf")
        ax.set_xlabel("w_g  (factor loading)")
        ax.set_title("Gene loadings")
        ax.legend(fontsize=8)

        # b_g
        ax = axes[1]
        ax.hist(true_b, bins=20, density=True, color=PALETTE[2], alpha=0.7,
                edgecolor="white", label="Sampled b_g")
        x = np.linspace(true_b.min() - 0.5, true_b.max() + 0.5, 200)
        ax.plot(x, stats.norm.pdf(x, cfg["baseline_mean"], cfg["baseline_std"]),
                color=PALETTE[3], lw=2,
                label=f"N({cfg['baseline_mean']},{cfg['baseline_std']}) pdf")
        ax.set_xlabel("b_g  (log-baseline)")
        ax.set_title("Gene log-baselines")
        ax.legend(fontsize=8)

        # p_g
        ax = axes[2]
        ax.hist(true_p, bins=20, density=True, color=PALETTE[4], alpha=0.7,
                edgecolor="white", label="Sampled p_g")
        x = np.linspace(0.001, 0.999, 300)
        a, b = cfg["p_beta_a"], cfg["p_beta_b"]
        ax.plot(x, stats.beta.pdf(x, a, b), color=PALETTE[5 % len(PALETTE)],
                lw=2, label=f"Beta({a},{b}) pdf")
        ax.set_xlabel("p_g  (overdispersion)")
        ax.set_title("Gene overdispersion")
        ax.legend(fontsize=8)

        fig.suptitle("Stage 1 (Hierarchical) — True gene parameters", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s1c_gene_params.png"))

    # ── H-1d. z_s distribution + patient mean vs z_s ─────────────
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        ax.hist(true_z, bins=30, density=True, color=PALETTE[0], alpha=0.7,
                edgecolor="white", label="Sampled z_s")
        x = np.linspace(-4, 4, 200)
        ax.plot(x, stats.norm.pdf(x, 0, 1), color=PALETTE[1], lw=2,
                label="N(0,1) pdf")
        ax.set_xlabel("z_s  (sample causal factor)")
        ax.set_title("z_s distribution")
        ax.legend(fontsize=8)

        ax = axes[1]
        patient_mean = counts.mean(axis=0)
        ax.scatter(true_z, patient_mean, s=10, alpha=0.4, color=PALETTE[2])
        ax.set_xlabel("True z_s")
        ax.set_ylabel("Patient mean count")
        ax.set_title(f"Patient mean count vs causal factor z_s\n"
                     f"(corr={float(np.corrcoef(true_z, patient_mean)[0,1]):.3f})")

        fig.suptitle("Stage 1 (Hierarchical) — Sample causal factor z_s", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s1d_z_distribution.png"))

    return paths


# ==================================================================
# HIERARCHICAL PGM — STAGE 3
# ==================================================================

def plot_stage3_hierarchical(
    fit_orig: dict,
    fit_pert: dict,
    data: dict,
    delta: float,
    out: Path,
) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    true_w, true_b, true_p, true_z = (
        data["true_w"], data["true_b"], data["true_p"], data["true_z"]
    )
    n_patients = len(true_z)

    # Resolve signs before plotting
    from src.inference.hierarchical_fitting import _resolve_sign, _safe_corr
    w_hat, z_hat = _resolve_sign(fit_orig["w_hat"], fit_orig["z_hat"], true_w, true_z)
    w_pert, z_pert = _resolve_sign(fit_pert["w_hat"], fit_pert["z_hat"], true_w, true_z)

    # ── H-3a. 4-panel parameter recovery ─────────────────────────
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        pairs = [
            (axes[0, 0], true_w,  w_hat,  "True w_g",  "w̃_g"),
            (axes[0, 1], true_b,  fit_orig["b_hat"], "True b_g",  "b̃_g"),
            (axes[1, 0], true_p,  fit_orig["p_hat"], "True p_g",  "p̃_g"),
            (axes[1, 1], true_z,  z_hat,  "True z_s",  "z̃_s"),
        ]
        for ax, xv, yv, xl, yl in pairs:
            corr = _safe_corr(xv, yv)
            ax.scatter(xv, yv, s=20, alpha=0.6, color=PALETTE[0])
            mn = min(xv.min(), yv.min()); mx = max(xv.max(), yv.max())
            ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="y=x")
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(f"corr = {corr:.3f}")
        fig.suptitle("Stage 3 (Hierarchical) — SVD parameter recovery", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s3a_hierarchical_recovery.png"))

    # ── H-3b. z_hat shift after do(z + delta) ────────────────────
    with plt.style.context(STYLE):
        z_shift = z_pert - z_hat
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ax = axes[0]
        ax.scatter(true_z, z_hat,  s=15, alpha=0.4, color=PALETTE[0], label="Original")
        ax.scatter(true_z, z_pert, s=15, alpha=0.4, color=TARGET_COLOR, label="After do(z+δ)")
        ax.set_xlabel("True z_s")
        ax.set_ylabel("Inferred z̃_s")
        ax.set_title(f"z̃_s before and after do(z_s + {delta:.1f})")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.scatter(true_z, z_shift, s=15, alpha=0.5, color=PALETTE[2])
        ax.axhline(z_shift.mean(), color="red", lw=1.5, ls="--",
                   label=f"mean shift = {z_shift.mean():.3f}")
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.set_xlabel("True z_s")
        ax.set_ylabel("Δz̃_s  (inferred)")
        ax.set_title(f"Per-patient z̃ shift after do(z + {delta:.1f})")
        ax.legend(fontsize=8)

        fig.suptitle("Stage 3 (Hierarchical) — Causal intervention detection", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s3b_z_shift_detection.png"))

    # ── H-3c. w_hat stability after z-intervention ───────────────
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 5))
        corr = _safe_corr(w_hat, w_pert)
        ax.scatter(w_hat, w_pert, s=25, alpha=0.7, color=PALETTE[4])
        mn = min(w_hat.min(), w_pert.min()); mx = max(w_hat.max(), w_pert.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set_xlabel("w̃_g (original fit)")
        ax.set_ylabel("w̃_g (after do(z+δ) fit)")
        ax.set_title(f"Gene loading w stability under z-intervention\n"
                     f"(corr={corr:.3f}, should be ~1 if intervention is causal)")
        fig.tight_layout()
        paths.append(_save(fig, out, "s3c_w_stability.png"))

    return paths


# ==================================================================
# HIERARCHICAL PGM — STAGE 7  (z-shift alignment)
# ==================================================================

def plot_stage7_z_alignment(
    z_alignment: dict,
    orig_fit: dict,
    pert_fit: dict,
    true_w: np.ndarray,
    true_z: np.ndarray,
    delta: float,
    out: Path,
) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    latent_dim = z_alignment["latent_dim"]
    per_dim    = z_alignment["per_dim"]
    best       = per_dim[z_alignment["best_dim"]]
    from src.inference.hierarchical_fitting import _resolve_sign

    # ── H-7a. Per-gene count shift alignment with true_w ──────────
    with plt.style.context(STYLE):
        mean_shifts = [d.get("mean_count_shift", d.get("mean_z_shift", 0.0))
                       for d in per_dim]
        cos_w       = [d["cos_to_true_w"] for d in per_dim]
        best_d      = z_alignment["best_dim"]
        colors      = [TARGET_COLOR if l == best_d else PALETTE[0]
                       for l in range(latent_dim)]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        bars = ax.bar([f"dim {l}" for l in range(latent_dim)],
                      mean_shifts, color=colors, edgecolor="white", alpha=0.85)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Mean count shift (decoded_pert − baseline)")
        ax.set_title("Mean gene expression shift per latent dim")
        for bar, v in zip(bars, mean_shifts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.01 * np.sign(v) if v != 0 else 0.02,
                    f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8)

        ax = axes[1]
        ax.bar([f"dim {l}" for l in range(latent_dim)],
               cos_w, color=colors, edgecolor="white", alpha=0.85)
        ax.axhline(z_alignment["null_cos_mean"], color="k", lw=1.5, ls="--",
                   label=f"Null mean={z_alignment['null_cos_mean']:.3f}")
        ax.axhline(z_alignment["null_cos_mean"] + 1.65 * z_alignment["null_cos_std"],
                   color="red", lw=1, ls=":", label="p<0.05 threshold")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel("cos(count_shift, true_w)")
        ax.set_title(f"Count-shift alignment with true gene loadings\n"
                     f"(z-score of best = {z_alignment['z_score']:.2f})")
        ax.legend(fontsize=8)

        fig.suptitle(f"Stage 7 (Hierarchical) — Causal alignment: latent dim vs do(z+{delta:.1f})",
                     y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s7h_z_alignment_bars.png"))

    # ── H-7b. Best dim: per-gene count shift vs true_w ────────────
    with plt.style.context(STYLE):
        bd = z_alignment["best_dim"]
        w_orig, z_orig = _resolve_sign(orig_fit["w_hat"], orig_fit["z_hat"], true_w, true_z)
        from src.inference.hierarchical_fitting import _safe_corr

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        ax = axes[0]
        ax.scatter(true_z, z_orig, s=12, alpha=0.5, color=PALETTE[0],
                   label=f"corr={_safe_corr(true_z, z_orig):.3f}")
        ax.set_xlabel("True z_s"); ax.set_ylabel("Inferred z̃_s (original)")
        ax.set_title("z factor recovery (original data)")
        ax.legend(fontsize=8)

        ax = axes[1]
        best_count_shift = best.get("mean_count_shift", best.get("mean_z_shift", 0.0))
        ax.scatter(true_w, best_count_shift * np.ones(len(true_w)),
                   s=12, alpha=0.3, color=TARGET_COLOR)
        ax.axhline(best_count_shift, color="red", lw=2, ls="--",
                   label=f"mean count shift = {best_count_shift:.2f}")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("True w_g (gene loading)")
        ax.set_ylabel("Mean count shift per gene")
        ax.set_title(f"Best dim {bd}: count shift vs true loadings\n"
                     f"cos(shift, w) = {best['cos_to_true_w']:.3f}")
        ax.legend(fontsize=8)

        fig.suptitle(f"Stage 7H — Best latent dim {bd} causal alignment", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, out, "s7h_best_dim_z_shift.png"))

    return paths
