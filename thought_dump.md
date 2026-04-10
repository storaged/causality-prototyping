# Project note: probing causal structure with a simple PGM and latent perturbations

## 1. Objective

The goal of this project is to build a **controlled sandbox** in which we can study whether latent representations learned by an AutoEncoder capture factors that have a **clear causal interpretation** with respect to a known generative model.

The general workflow is:

1. define a simple **Probabilistic Graphical Model (PGM)** that generates synthetic count data,
2. generate a ground-truth dataset from this model,
3. train a simple **AutoEncoder (AE)** on these data,
4. perform **controlled perturbations** either:

   * directly at the level of the ground-truth data / PGM parameters, or
   * in the learned latent space,
5. decode perturbed latent points back to data space,
6. refit the PGM to these perturbed data,
7. analyse whether posterior estimates of selected PGM variables change in a **regular, interpretable, and functionally controlled** way.

The main scientific question is:

> Can any latent dimension be linked in a stable and interpretable way to a specific generative factor of the PGM?

If yes, this would suggest that the latent space is not merely geometric, but captures something closer to a causal factorization of the data-generating process.

---

## 2. High-level intuition

We assume we have synthetic observations generated from a known PGM. This gives us **full control** over the true underlying parameters and allows us to apply **interventions**.

We then compare two types of perturbations:

### A. Perturbation in data / generative space

We modify the ground-truth mechanism directly, for example by changing one hyperparameter or one local variable in the PGM, and regenerate the data.

### B. Perturbation in latent space

We encode the observed data into latent space, perturb one latent coordinate, and decode back to data space.

The key idea is that decoded perturbed data should still remain “understandable” by the original PGM, meaning that the PGM can still be fitted to them and yield meaningful posterior estimates.

Then we ask:

* does changing latent dimension ( l ) induce a specific and regular change in some PGM quantity?
* or do all latent perturbations induce only diffuse, noisy, Brownian-like changes across all inferred parameters?

The second scenario would be uninformative. The first would suggest identifiable structure.

---

## 3. Minimal generative model

We should begin with the simplest possible model that still allows causal interpretation.

## Option 1: very simple gene-patient count model

For each gene ( i \in {1,\dots,I} ):

* ( \lambda_i \sim \text{Poisson}(5) )
* ( p_i \sim \text{Beta}(2,5) )

Then for each patient ( n \in {1,\dots,N} ):

* ( C_{i,n} \sim \text{NegBinom}(\lambda_i, p_i) )

Interpretation:

* ( \lambda_i ) controls mean/abundance of gene ( i ),
* ( p_i ) controls dispersion / count generation characteristics,
* ( C_{i,n} ) is the observed count matrix.

This gives a synthetic dataset ( C \in \mathbb{N}^{I \times N} ).

A controlled intervention can then be defined as:

[
\text{do}\big(p_i \sim \text{Beta}(2,5) \rightarrow \text{Beta}(1,5)\big)
]

for one selected gene or a subset of genes. This mimics an experimental perturbation affecting the generative mechanism of a gene.

Example initial scale:

* ( N = 100 ) patients,
* ( I = 1000 ) genes.

---

## Option 2: slightly richer structured model

If we want a more structured setting, we can define latent sample-specific factors and gene-specific parameters, for example:

* ( N_s \sim \mathcal{N}(l_s, \sigma_N^2) )
* ( p_g \sim \text{Exponential}(\alpha_p) )
* ( \mu_{g,s} = f(z_s, w_g, \ldots) )
* ( C_{g,s} \sim \text{NegBinom}(N_s, p_g \cdot \mu_{g,s}) )

Here:

* ( s ) indexes samples,
* ( g ) indexes genes,
* ( N_s ) could represent a sample-level latent effect,
* ( p_g ) a gene-level parameter,
* ( \mu_{g,s} ) the structured mean.

This variant is closer to a proper hierarchical PGM and may be useful after the first toy model works.

---

## 4. Core hypothesis

After training the AE and perturbing one latent coordinate, we decode the perturbed latent representation:

[
L' \rightarrow \text{Dec}(L')
]

Then we fit the PGM to the decoded data and inspect posterior distributions of the PGM variables.

The central hypothesis is:

* if for **all** latent coordinates ( l ), the inferred quantities such as ( N'_s ), ( p'_i ), or ( \lambda'_i ) change only in an irregular, diffuse, Brownian-like manner, then the latent space does **not** provide interpretable control over the generative factors;
* if there **exists** a latent coordinate ( l ) such that perturbing it induces a **systematic, regular, functional dependence** in some inferred PGM variable, then that latent coordinate may correspond to a meaningful generative axis.

---

## 5. Proposed work packages

## WP1. Build the synthetic data generator

Implement a simple PGM simulator that can generate count matrices under known parameters.

Requirements:

* sample from the chosen priors,
* generate observed count matrix,
* save both:

  * observed data,
  * true latent/generative parameters.

Outputs:

* synthetic datasets,
* metadata with full ground truth.

Minimum deliverable:

* a reproducible script producing datasets for fixed random seeds.

---

## WP2. Define controlled perturbations in generative space

Implement interventions directly on the PGM side.

Examples:

* modify prior of selected ( p_i ),
* shift selected ( \lambda_i ),
* perturb sample-specific latent ( N_s ),
* perturb a block of genes instead of a single gene.

Important:

* perturbations should be explicit and logged,
* perturbation strength should be tunable,
* both local and global interventions should be supported.

Outputs:

* original dataset ( C ),
* perturbed dataset ( C' ),
* exact record of intervention type, target, and magnitude.

---

## WP3. Train a simple AutoEncoder

Train a basic AutoEncoder on the synthetic count data.

Initial recommendation:

* start with a very simple fully connected AE,
* no need for complex architectures at first,
* latent dimension should be small enough to force compression but large enough to reconstruct the data reasonably.

Suggested first setup:

* input: patient-level or sample-level count vectors,
* encoder: MLP,
* bottleneck: low-dimensional latent vector,
* decoder: MLP,
* reconstruction loss:

  * MSE as a first baseline,
  * later possibly count-aware loss if needed.

Outputs:

* trained encoder,
* trained decoder,
* latent representations of original and perturbed data.

---

## WP4. Perturb latent coordinates

For each latent dimension ( l ):

1. encode original data,
2. perturb coordinate ( l ) in a controlled way,
3. decode the modified latent code back to data space,
4. store resulting decoded data.

Perturbation scheme should include:

* positive and negative shifts,
* several magnitudes,
* one-coordinate-at-a-time perturbations,
* optionally multi-coordinate perturbations later.

Outputs:

* decoded datasets after latent perturbations,
* full perturbation log.

---

## WP5. Fit the PGM back to observed / decoded data

Fit the original PGM to:

* original data,
* PGM-perturbed data,
* AE-latent-perturbed decoded data.

Two inference routes may be considered:

### A. MCMC / Metropolis-within-Gibbs

Advantages:

* conceptually clean,
* full posterior sampling.

Disadvantages:

* slow,
* convergence diagnostics may be cumbersome.

### B. Variational Inference

Advantages:

* faster,
* scalable.

Disadvantages:

* requires deriving/implementing ELBO and approximating posterior family.

Recommendation:

* start with the simplest inference approach that is practical,
* it is acceptable to begin with approximate Bayesian fitting or even MAP if needed for proof of concept,
* later upgrade to a fuller posterior treatment.

Outputs:

* posterior samples or approximate posterior summaries,
* MAP/posterior mean estimates for all key variables.

---

## WP6. Analyse causal interpretability of latent dimensions

This is the main scientific analysis.

For each latent dimension, quantify whether perturbing it induces a structured change in PGM parameters.

Examples of questions:

* does latent coordinate ( l ) mainly affect one family of PGM parameters?
* is the direction of effect monotonic?
* is the effect repeatable across random seeds?
* does perturbing ( l ) reproduce the effect of a known generative intervention?

This should be analysed both visually and quantitatively.

---

## 6. Experimental plan

## Experiment 1. Sanity check on the PGM alone

Generate data from the PGM and fit the same PGM back to them.

Goal:

* verify that the inference procedure can recover the known generative parameters at least approximately.

Success criterion:

* true parameters lie close to posterior means / MAP,
* error remains within acceptable tolerance,
* posterior uncertainty behaves sensibly.

---

## Experiment 2. Controlled generative intervention

Apply a known intervention in the PGM, regenerate data, and refit the model.

Goal:

* verify that the fitting procedure detects the induced change.

Success criterion:

* posterior estimates shift in the expected direction,
* effect size reflects intervention magnitude.

---

## Experiment 3. AE reconstruction quality

Train AE on original data and measure reconstruction fidelity.

Goal:

* ensure that the AE preserves enough information for the decoded data to remain compatible with the PGM.

Success criterion:

* reasonable reconstruction error,
* decoded data remain within plausible count/data ranges,
* PGM can still be fitted to reconstructions.

---

## Experiment 4. Latent perturbation screening

Perturb each latent coordinate systematically and refit the PGM to decoded data.

Goal:

* determine whether some latent coordinates align with specific generative factors.

Success criterion:

* at least some latent dimensions show structured, reproducible influence on selected PGM parameters,
* the effect is stronger than random/noise baselines.

---

## Experiment 5. Comparison between true interventions and latent interventions

Compare:

* change induced by explicit intervention in the PGM,
* change induced by perturbing latent coordinate ( l ).

Goal:

* test whether latent perturbations can mimic true causal interventions.

Success criterion:

* similarity between posterior shifts under the two scenarios,
* identifiable mapping from latent perturbation to generative parameter change.

---

## 7. Measures of success

The project should define success at several levels.

## A. Generative recovery

How well can the PGM recover the known parameters from synthetic data?

Possible metrics:

* RMSE / MAE between true and inferred parameters,
* posterior coverage,
* correlation between true and estimated parameters.

## B. AE fidelity

How well does the AE reconstruct the data?

Possible metrics:

* reconstruction loss,
* correlation between original and reconstructed counts,
* preservation of marginal distributions,
* ability of the PGM to fit reconstructed data.

## C. Latent controllability

Does perturbing one latent coordinate induce a regular effect?

Possible metrics:

* monotonicity of change in inferred parameters vs perturbation magnitude,
* variance explained in a target PGM parameter by one latent coordinate,
* sensitivity curves,
* repeatability across runs.

## D. Causal alignment

Do latent perturbations reproduce known generative interventions?

Possible metrics:

* similarity of posterior shifts,
* distance between intervention signatures,
* ranking agreement of affected genes/samples/parameters.

---

## 8. Negative outcome that is still scientifically useful

A useful negative result is also possible.

If all latent dimensions produce only diffuse, unstable, Brownian-like changes in inferred PGM quantities, then this would suggest that:

* the AE latent space is mainly reconstructive, not causal,
* the chosen architecture/objective does not disentangle generative factors,
* stronger inductive bias or structured latent models may be needed.

This is not a failure; it is a meaningful conclusion about the limitations of naive latent spaces for causal interpretation.

---

## 9. Practical implementation components

The sandbox should contain the following modules.

## 1. PGM simulator

* synthetic data generator,
* parameter logging,
* support for controlled interventions.

## 2. AutoEncoder

* simple baseline architecture,
* training pipeline,
* encoding and decoding utilities.

## 3. Perturbation engine

* perturbation in generative space,
* perturbation in latent space,
* controlled magnitude and direction.

## 4. PGM inference module

* fitting routine for the PGM,
* ideally Bayesian,
* posterior summaries and diagnostics.

## 5. Data handler / experiment registry

Must record:

* random seed,
* simulation setting,
* perturbation type,
* perturbation target,
* perturbation magnitude,
* reconstruction metrics,
* inferred posterior summaries,
* downstream evaluation metrics.

This is important so that experiments are reproducible and comparable.

---

## 10. Recommended order of implementation

To reduce complexity, the student should proceed in the following order:

1. implement the simplest PGM simulator,
2. verify that synthetic data can be regenerated reproducibly,
3. fit the same model back to its own synthetic data,
4. introduce one explicit generative intervention,
5. train a minimal AE,
6. test reconstruction quality,
7. implement one-latent-dimension perturbations,
8. fit the PGM to decoded perturbed data,
9. quantify whether any latent direction corresponds to a structured generative effect,
10. only then move to richer models or more advanced inference.

---

## 11. Minimum expected outputs

At minimum, the first stage of the project should deliver:

* a simple PGM simulator for count data,
* a Bayesian or approximate Bayesian fitting procedure,
* a basic AutoEncoder,
* a latent perturbation pipeline,
* a structured experiment table,
* plots showing how inferred PGM parameters respond to:

  * true interventions,
  * latent perturbations,
* a short written summary answering:

  * whether latent directions are interpretable,
  * whether any latent direction aligns with a known generative factor,
  * whether AE perturbations can approximate causal interventions.

---

## 12. Final expected scientific message

The intended message of this sandbox is not to prove causality in general, but to test the following narrower claim:

> In a fully controlled synthetic setting with known generative structure, can a simple learned latent space recover dimensions that behave like actionable generative factors under intervention?

This provides a clean first step before moving to more complex biological settings.
