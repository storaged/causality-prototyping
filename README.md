# Causal Latent Space Sandbox

A controlled computational sandbox for testing whether unsupervised neural representations (autoencoders, VAEs) capture causal structure latent in single-cell-like count data.

The central question: **can a simple learned latent space recover dimensions that behave like actionable generative factors under intervention?**

---

## What it implements

The pipeline has seven stages, each with sanity checks before proceeding:

| Stage | Description |
|-------|-------------|
| 1 | **Hierarchical PGM** — synthesises a gene × patient NegBinom count matrix with a known rank-1 causal factor `z_s` |
| 2 | **Ground-truth interventions** — Pearl's *do*-calculus applied directly to the generative parameters (`do(z_s → z_s + δ)`) |
| 3 | **Baseline PGM inference** — method-of-moments + SVD recovery of all parameters from raw counts |
| 4 | **AE / β-VAE training** — Poisson NLL loss in log-expression space, 4-dimensional latent space |
| 5 | **PGM inference on reconstructed counts** — checks that the AE preserves inferability |
| 6 | **Latent perturbation screening** — each AE dimension is perturbed ±1,2 σ and the induced gene-expression shift is measured |
| 7 | **Causal alignment test** — cosine similarity between per-gene count shift and true loading vector `w_g`; Z-score against a random-direction null |

### Generative model (hierarchical PGM)

```
w_g ~ N(0, σ_w²)       gene loading
b_g ~ N(μ_b, σ_b²)     log-baseline expression
p_g ~ Beta(α, β)        overdispersion
z_s ~ N(0, 1)           patient causal factor

log μ_{g,s} = w_g · z_s + b_g
C_{g,s} ~ NegBinom(r_{g,s}, p_g)
```

A full mathematical description with plate diagrams and proofs is in [notes/causal_latent_space_description.pdf](notes/causal_latent_space_description.pdf).

### Current results (toy setup, G=50, S=1000)

| Metric | Value |
|--------|-------|
| SVD recovery corr(w_g) | 0.991 |
| AE reconstruction corr | 0.992 |
| Best causal alignment ρ* | 0.501 |
| Z-score (vs random null) | 3.54 |
| Multi-seed pass rate | 5 / 5 |
| Success levels | 4 / 4 |

---

## Repository layout

```
.
├── run_pipeline.py          # entry point — runs all 7 stages
├── config.yaml              # all hyperparameters (see below)
├── requirements.txt
├── src/
│   ├── pgm/                 # generative models and interventions
│   ├── inference/           # method-of-moments + SVD fitting
│   ├── models/              # AE and β-VAE architectures
│   ├── analysis/            # latent perturbation + causal alignment
│   └── visualization/       # all plots
├── results/
│   ├── achievements_report.json   # structured success-level summary
│   └── plots/                     # per-stage figures
└── notes/
    ├── causal_latent_space_description.tex   # full mathematical writeup
    ├── causal_latent_space_description.pdf
    └── others/
        ├── programming_guidelines.md   # original stage-by-stage roadmap
        └── thought_dump.md             # early design notes
```

---

## How to run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python run_pipeline.py
```

This executes all 7 stages, saves plots to `results/plots/`, and writes `results/achievements_report.json`.

### Run with a custom config

```bash
python run_pipeline.py --config config.yaml
```

---

## How to control it

All experiment parameters live in `config.yaml`. Key sections:

### PGM type

```yaml
pgm:
  type: "hierarchical"   # "simple" | "hierarchical"
```

- `hierarchical` — rank-1 factor model with patient causal axis `z_s` (recommended)
- `simple` — gene-independent NegBinom, no causal structure

### Scale

```yaml
hierarchical_pgm:
  n_genes: 50        # number of genes G
  n_patients: 1000   # number of patients S
  loading_std: 1.0   # w_g ~ N(0, loading_std²)
```

### Intervention

```yaml
hierarchical_perturbation:
  z_delta: 2.0       # magnitude of do(z_s → z_s + delta)
```

### AutoEncoder

```yaml
autoencoder:
  latent_dim: 4
  hidden_dims: [64, 32]
  n_epochs: 500
  loss_type: "poisson"   # "poisson" | "mse"
```

### Reproducibility across seeds

```yaml
multi_seed:
  enabled: true
  seeds: [0, 42, 123, 1234, 12345]
```

### VAE comparison

```yaml
vae:
  enabled: true
  beta: 0.5
```

---

## Background and theory

See [notes/causal_latent_space_description.pdf](notes/causal_latent_space_description.pdf) for:

- Full generative model with plate diagram
- Biomedical interpretation (scRNA-seq / tumour microenvironment)
- Baseline inference derivations (method-of-moments, SVD)
- AE and β-VAE architecture details
- Causal alignment metric derivation
- **Appendix A**: nonlinear ICA theory and the identifiability problem
