# Roadmap for implementation

## Overall goal

Build a sandbox in which we can test whether a simple AutoEncoder latent space captures directions that correspond in a stable way to factors from a known generative model.

The implementation should proceed from the smallest possible setup to more realistic experiments. At each stage, we should only move forward if the previous stage passes basic sanity checks.

---

# Stage 0. General rules

Before starting, enforce the following:

* all experiments must be reproducible from a fixed random seed,
* every run must save:

  * simulation parameters,
  * perturbation type and magnitude,
  * trained model config,
  * inferred parameters / posterior summaries,
  * evaluation metrics,
* keep the first version intentionally simple,
* no advanced biology-inspired complexity until the toy setup works.

---

# Stage 1. Smallest possible synthetic generator

## Deliverable

Implement the tiniest PGM-like simulator that produces count data.

Suggested first model:

* for each gene (i):

  * (\lambda_i \sim \text{Poisson}(5))
  * (p_i \sim \text{Beta}(2,5))
* for each patient (n):

  * (C_{i,n} \sim \text{NegBinom}(\lambda_i, p_i))

This gives a matrix (C \in \mathbb{N}^{I \times N}).

## Minimal setup

* (N = 20) patients
* (I = 50) genes

## Sanity checks

Programmer should verify:

1. counts are generated without numerical issues,
2. dimensions are correct,
3. distributions look reasonable,
4. rerunning with same seed gives identical data,
5. changing one hyperparameter visibly changes data statistics.

## Output

* `counts_matrix`
* `true_lambda`
* `true_p`
* metadata file with seed and config

Do not move further until this works.

---

# Stage 2. Controlled ground-truth perturbation engine

## Deliverable

Implement interventions on the generator side.

Examples:

* perturb one selected gene:

  * (p_i: \text{Beta}(2,5) \rightarrow \text{Beta}(1,5))
* perturb a small group of genes,
* perturb (\lambda_i) upward or downward.

## Required functionality

* select target genes,
* define perturbation type,
* define perturbation strength,
* regenerate perturbed dataset,
* log exactly what was changed.

## Sanity checks

For each perturbation:

1. affected genes change more than unaffected genes,
2. stronger perturbation gives stronger effect,
3. repeated runs show same directional effect.

## Output

* original dataset
* perturbed dataset
* perturbation manifest

This stage defines the **ground-truth baseline intervention**.

---

# Stage 3. Baseline inference without AutoEncoder

## Deliverable

Fit the same simple model back to the generated data.

At first this does not need full MCMC. A practical roadmap is:

1. start with rough parameter estimation / MAP / method-of-moments,
2. later replace or extend with Bayesian inference.

The point here is not elegance, but checking whether the model can recover what generated the data.

## Sanity checks

On unperturbed data:

* inferred (\lambda_i) correlates with true (\lambda_i),
* inferred (p_i) correlates with true (p_i).

On perturbed data:

* inferred parameters shift in the expected direction,
* affected genes are identifiable better than random.

## Metrics

* correlation between true and inferred parameters,
* MAE / RMSE,
* ranking of most affected genes.

## Purpose

This is the key baseline:
if we cannot recover perturbations directly from synthetic data, then nothing with AE will be interpretable.

---

# Stage 4. Small AutoEncoder baseline

## Deliverable

Train a minimal AE on the count matrix.

Keep it simple:

* fully connected encoder,
* bottleneck latent space,
* fully connected decoder.

Start with:

* input = patient vector or gene vector, but choose one convention and keep it fixed,
* latent dimension small, e.g. 2, 4, 8,
* standard reconstruction loss first.

## Minimal experiments

Train on:

* only unperturbed data first.

## Sanity checks

1. reconstruction error decreases during training,
2. decoded outputs are stable,
3. reconstructed data remain numerically sensible,
4. simple statistics of reconstructed data resemble original ones.

## Important baseline

Compare AE against trivial baselines:

* identity / no compression baseline,
* PCA with same latent dimension.

If AE does not outperform or at least match PCA in reconstruction and stability, do not overinterpret its latent space.

---

# Stage 5. Fit the PGM to reconstructed data

## Deliverable

Take:

* original data (C),
* reconstructed data (\hat C = \text{Dec}(\text{Enc}(C))),

and fit the PGM/inference model to both.

## Sanity checks

* inferred parameters from reconstructed data should be close to those from original data,
* reconstruction should not destroy the signal needed for inference,
* perturbation signatures should remain at least partially visible after AE compression/reconstruction.

## Purpose

This is a critical checkpoint:
if decoded data are no longer interpretable by the PGM, latent perturbation analysis will not be meaningful.

---

# Stage 6. Single-latent perturbation experiments

## Deliverable

For one encoded observation:

1. encode data into latent vector,
2. perturb one latent coordinate at a time,
3. decode back to data space,
4. fit the PGM to decoded perturbed data,
5. record which inferred parameters changed.

## Perturbation scheme

For each latent dimension (l):

* negative shift: (-\delta)
* no shift: (0)
* positive shift: (+\delta)

Later:

* multiple magnitudes, e.g. (0.5, 1.0, 2.0) standard deviations.

## Sanity checks

For each latent dimension:

1. perturbation effect is reproducible,
2. effect size changes smoothly with perturbation magnitude,
3. some dimensions may have minimal effect,
4. ideally some dimensions show targeted effect rather than diffuse global noise.

---

# Stage 7. Compare latent perturbations to true perturbations

## Deliverable

Systematically compare:

* perturbation introduced in the true generator,
* perturbation introduced in latent space.

The main question is:
does any latent direction reproduce the parameter shift pattern of a known intervention?

## Practical comparison

For each true intervention:

* compute inferred parameter shift vector,
* compute analogous shift vector for each latent perturbation,
* compare similarity.

## Metrics

Use simple metrics first:

* correlation of shift vectors,
* cosine similarity,
* overlap in top affected genes.

## Success signal

A good result is:

* some latent coordinate consistently mimics a specific true intervention better than others.

---

# Stage 8. Scaling up complexity

Only after the previous stages work should complexity increase.

## Order of scaling

1. increase number of genes and patients,
2. increase latent dimension,
3. add stronger/noisier perturbations,
4. move from single-gene to group-level perturbations,
5. introduce hierarchical sample effects,
6. replace simple fitting with proper Bayesian inference,
7. test richer PGM.

Do not change many things at once.

---

# Baselines to control throughout

The programmer should always keep the following baselines:

## Baseline A. No perturbation

Encoded-decoded data without any perturbation.

Purpose:

* distinguish reconstruction artifacts from perturbation effects.

## Baseline B. True perturbation in generator

Known intervention directly in PGM.

Purpose:

* gold standard for causal signal.

## Baseline C. Random noise perturbation in data space

Add random noise of similar scale without structured intervention.

Purpose:

* test whether observed effects are specific or just noise response.

## Baseline D. PCA instead of AE

Use PCA latent space and perturb principal components.

Purpose:

* verify whether AE gives something beyond generic low-dimensional compression.

## Baseline E. Random latent direction

Perturb random directions, not aligned to single latent axes.

Purpose:

* check whether axis-level interpretation is special or arbitrary.

---

# Minimal definition of success

At the implementation stage, success should be defined modestly.

## First success level

* simulator works,
* perturbations work,
* fitting recovers parameters approximately.

## Second success level

* AE reconstructs data without destroying inferability.

## Third success level

* at least one latent coordinate shows stable, non-random effect on inferred parameters.

## Fourth success level

* at least one latent coordinate partially matches the signature of a true generator-side perturbation.

That is enough for the first project milestone.

---

# Concrete programmer task list

## Task 1

Implement synthetic generator with config file and fixed seeds.

## Task 2

Implement perturbation module for generator-level interventions.

## Task 3

Implement basic parameter recovery / fitting routine.

## Task 4

Implement experiment logger saving:

* seed,
* config,
* perturbation details,
* reconstruction metrics,
* inferred parameters.

## Task 5

Implement minimal AE training pipeline.

## Task 6

Implement latent perturbation module.

## Task 7

Implement comparison scripts:

* original vs perturbed,
* true perturbation vs latent perturbation,
* AE vs PCA baseline.

## Task 8

Implement simple plots:

* reconstruction error,
* inferred vs true parameter scatterplots,
* heatmap of parameter changes under latent perturbations,
* similarity matrix between true and latent interventions.

---

# Suggested implementation order

The safest order is:

1. synthetic generator,
2. generator perturbations,
3. basic inference / recovery,
4. sanity plots,
5. simple AE,
6. PGM fit on reconstructed data,
7. latent one-axis perturbations,
8. comparison to true interventions,
9. PCA and random-noise baselines,
10. scale-up experiments.

---

# One-line principle for the programmer

At every step, the question is:

> Does the newly added component preserve or improve our ability to detect a known controlled intervention?

If the answer is no, do not move to a more complex setup yet.

