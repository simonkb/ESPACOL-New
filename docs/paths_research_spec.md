# PATHS-v3: PGF concentration with signed adjacent probability transport

Status: prospective implementation candidate on `paths-signed-transport-v3`

Audited base: ORIGIN-v3 bounded-rate checkpoint, loaded by explicit SHA-256

Protocol identifier: `paths-v3-sapt`

## Scope and claim discipline

PATHS-v3 is an architecture for image-level ordinal severity labels. It is not
yet a foundation model: that term requires broad pretraining, transfer across
tasks and modalities, and scaling evidence that the current APTOS/EyePACS
experiments do not provide. The intended initial claim is narrower:

> a severity-grading architecture that converts a nested multiscale spatial
> ordinal field into a mass-normalized PGF concentration ledger, then transports
> an audited base posterior through a signed, adjacent, row-stochastic ordinal
> kernel, with exact replay of the complete predictor after internal cell
> deletion.

The individual ingredients have precedents in adjacent literatures, including
continuation-ratio ordinal regression, additive MIL, probability-generating
functions, and Markov transport. The research claim concerns their precise
coupling and its verifiable computation. Any paper must therefore use a
qualified “to our knowledge” statement and establish the contribution against
matched controls. Code, a single validation fold, or a good accuracy number
cannot guarantee novelty, acceptance, or generalization.

## Why ORIGIN-v3 is the controlled base

ORIGIN-v3 is accurate and structurally auditable, but its multiscale spatial
field is reduced to positive cumulative rate totals. For boundary `k`, its
computation has the form

\[
\lambda_k=\pi_k+b_kR\sum_s w_{s,k}\frac{1}{n_s}
\sum_i u_{s,i,k}.
\]

Consequently, fields with the same first moment can produce the same rate even
when one is focal and the other diffuse. Its pure-birth construction also has
no explicit mechanism for moving posterior mass downward when the base model
over-grades an image. On the audited APTOS fold, V3 obtained 86.01% accuracy and
0.9265 QWK but recalled 0/15 grade-3 cases. V4/V5 locality restrictions lost
substantial accuracy, while V6--V8 relation modules produced weak or
control-equivalent effects. PATHS-v3 therefore preserves the complete audited
V3 computation and tests one controlled addition: whether spatial
concentration can improve the posterior through conservative ordinal
transport.

## Forward computation

### 1. Nested local ordinal field

Let `K` be the number of ordered grades, `s` a feature scale, and `i` a valid
spatial cell. The ORIGIN-v3 pointwise head emits a local ordinal-state simplex.
Its normalized cumulative atoms are

\[
u_{s,i,k}=P(S_{s,i}>k),\qquad
1\ge u_{s,i,0}\ge\cdots\ge u_{s,i,K-2}\ge0.
\]

The implementation obtains `u` by dividing stored cumulative atoms by their
architectural mass cap. These are model-internal ordinal evidence quantities,
not calibrated lesion probabilities.

### 2. Tangent-removed PGF concentration ledger

For a fixed probe `z` in `(0,1)`, put `a=1-z` and `L=-log(z)`. PATHS uses

\[
\phi_z(u)=\frac{-\log(1-au)-au}{L-a}.
\]

This basis satisfies

\[
\phi_z(0)=\phi'_z(0)=0,\quad \phi_z(1)=1,\quad
0\le\phi_z(u)\le u,
\]

and is strictly convex on `(0,1)`. At fixed evidence mass, its sum is therefore
larger for a more concentrated field. Removing the linear tangent prevents
many weak responses from simply reproducing the V3 first moment.

For each scale and boundary, the original evidence mass and local
concentration terms are

\[
M_{s,k}=\sum_{i\in V_s}u_{s,i,k},\qquad
\ell_{s,i,k,z}=\begin{cases}
\phi_z(u_{s,i,k})/M_{s,k},&M_{s,k}>0,\\
0,&M_{s,k}=0.
\end{cases}
\]

The mass denominator is fixed from the original forward pass and is not
renormalized after an intervention. Invalid cells contribute neither mass nor
concentration. Learned probe weights `alpha` and inherited V3 scale weights
`w` are nonnegative simplexes, giving the additive ledger

\[
q_k=\sum_s\sum_i w_{s,k}\sum_z\alpha_{k,z}\ell_{s,i,k,z},
\qquad 0\le q_k\le1.
\]

Uniform replication leaves this concentration unchanged; a saturated
singleton gives maximal concentration, while fixed total evidence spread over
increasingly many weak cells approaches zero.

### 3. Signed adjacent probability transport

Let `p^(0)` be the normalized ORIGIN-v3 grade posterior. For each ordinal
boundary, PATHS learns a bounded gain `g_k`, threshold `theta_k`, and positive
slope `s_k`. With transport strength `rho`, it computes

\[
d_k=\tanh\!\left(s_k(q_k-\theta_k)\right),
\]

\[
o_k^+=\rho g_kq_k\frac{1+d_k}{2},\qquad
o_k^-=\rho g_kq_k\frac{1-d_k}{2}.
\]

For grade row `j`, the unnormalized weights are `1` for staying in grade `j`,
`o_j^+` for moving to `j+1`, and `o_{j-1}^-` for moving to `j-1`, where defined.
Row normalization produces a tridiagonal Markov kernel `T(q)`:

\[
T_{j,j}+T_{j,j-1}+T_{j,j+1}=1,\qquad T_{j,l}=0
\ \text{for}\ |j-l|>1.
\]

The final posterior is

\[
p=p^{(0)}T(q).
\]

Thus refinement conserves probability mass, cannot jump over a grade in its
single transport step, and can correct both under-grading and over-grading. It
is not the earlier one-sided nonnegative continuation-logit correction. The
reported `boundary_correction` is only the *derived* change between the final
and base conditional continuation log-odds; it is not an additive evidence
ledger.

The transported posterior is factorized exactly into at-risk continuation
probabilities for loss evaluation and ordinal diagnostics:

\[
h_k=P(Y>k\mid Y\ge k),\qquad
p_y=\left(\prod_{j<y}h_j\right)(1-h_y),
\]

with `p_(K-1)=prod_j h_j`. There is no global classifier, attention score,
concept-logit path, or post-hoc explanation bypass.

When `rho=0`, the refiner returns the actual `OriginOutput` object. This is the
exact strength-zero risk-objective control. The SAPT treatment currently uses
`rho=1` and gain initialization `0.05`, so the treatment itself is **not** an
exact V3 predictor at epoch zero; static V3 reproduction is audited separately.

## Proper rare-boundary objective

Categorical continuation NLL decomposes into at-risk binary decisions. PATHS
uses a fixed positive weight per boundary, computed only from the training
fold:

\[
c_k\propto \widehat P_{train}(Y\ge k)^{-\alpha},\qquad
\frac1{K-1}\sum_kc_k=1.
\]

The risk-set score is

\[
L_{RS}=\sum_k c_k\,1[y\ge k]\,
\operatorname{BCEWithLogits}(\eta_k,1[y>k]).
\]

For fixed positive `c_k`, population regret is a positive weighted sum of
Bernoulli KL divergences. The conditional score remains strictly proper and
does not alter the target in the manner of outcome-dependent class weighting.
The registered objective is

\[
L=L_{RS}+0.25L_{RPS},
\]

with `alpha=0.5` in the current pilot.

## Controlled training protocol

APTOS fold 0 uses two matched learned runs plus one immutable reference:

1. **SAPT treatment:** `paths_variant=signed_transport`, `rho=1`; V3 generator
   and SAPT refiner are optimized.
2. **Risk-objective control:** `paths_variant=risk_objective_v3`, `rho=0`; the
   refiner is frozen and the same V3 generator is optimized with the same
   risk-set objective.
3. **Static V3:** the hash-bound source checkpoint is evaluated without an
   update.

The learned treatment and risk-objective control use the same V3 checkpoint,
split, seed, data order, objective, batch size, scheduler, and stopping rule.
Their critical configurations may differ only in run location, variant, and
transport strength. In the registered APTOS pilot:

- V3 generator learning rate: `1e-5`;
- SAPT refiner learning rate: `5e-4` (inactive in the strength-zero control);
- encoder learning rate if enabled: `1e-5`;
- encoder: frozen for all 12 epochs;
- `correction_only_epochs=0`, so there is no refiner-only warm-up;
- batch size: 8; selection rule: accuracy then QWK; locked outer test skipped.

The gated EyePACS pilot keeps the encoder frozen for all 25 epochs and uses the
same generator/refiner learning rates. These are pilot schedules, not universal
hyperparameters.

The selected learned checkpoint is preserved as a byte-exact
`best_learned.pth` alias of `best.pth`. Results, checkpoints, certificates, the
implementation source set, git commit, split signature, configuration, and V3
source checkpoint are hash-bound by the promotion audit.

## Exact interpretation contract

For explanation, the model enumerates valid multiscale cells and applies the
same singleton intervention to the complete predictor. Removing a cell:

1. removes its stored ORIGIN-v3 local rate entries;
2. removes its stored per-cell PGF concentration entries;
3. preserves the original PGF mass denominator and all learned parameters;
4. recomputes the V3 posterior, concentration `q`, transport kernel `T(q)`, and
   final posterior.

Removed concentration is computed independently from the masked stored local
ledger. The audit verifies the rate and concentration partitions, kernel row
sums and nonnegativity, absence of non-adjacent transitions, posterior replay,
and signed-flow reconstruction. Derived log-odds and net-flow changes are
reported as nonlinear intervention effects, not claimed as additive ledgers.

For a small deterministic validation batch, the certificate selects the
positive supporter whose deletion most reduces the configured decision margin,
breaking ties by the predicted-grade log-probability reduction. It records the
rate, concentration, transport, continuation, and class-probability changes;
canonical replay must match vectorized candidate evaluation. It also records
the theoretical/clipped encoder receptive field and a deterministic
scale-matched random deletion. If no cell has a positive effect, the certificate
says so instead of inventing a witness.

A separate audit traverses the complete inner-validation split and verifies
that the selected checkpoint uses an active valid adjacent Markov kernel. It
records positive and negative net-flow counts, concentration and direction
quantiles, transport magnitude, mass conservation, row-stochasticity, and
non-adjacent entries.

These are exact internal-computation interventions. They are not gradients,
attention maps, or post-hoc saliency. They do **not** establish that a latent
cell is a named lesion, that its receptive field is pixel-local, or that
deleting an internal cell is equivalent to editing the input image.

## Pre-registered APTOS promotion gate

The first experiment is APTOS fold 0 on the V3 split. The outer test remains
locked. EyePACS remains blocked unless the SAPT treatment passes every identity,
performance, and structural check against both controls.

Performance requirements are:

- at least 253/293 validation images correct;
- strictly more correct predictions than static V3 and the matched
  risk-objective control;
- QWK at least static V3 + 0.003 and strictly above the risk-objective control;
- no regression versus static V3 in MAE, balanced accuracy, or macro-F1;
- at least 4/15 grade-3 cases correct.

Protocol and structural requirements include:

- identical commit, split, V3 source, matched configuration, and reproduced
  static-V3 predictions across treatment and control;
- checkpoint, result, certificate, architecture, and implementation hash
  consistency;
- a nonzero final SAPT effect and exact joint rate/concentration replay;
- a nonnegative, row-stochastic, strictly adjacent transport kernel whose
  posterior and boundary-flow identities replay within tolerance;
- active bidirectional net flow over the full 293-image validation split.

Failure rejects or revises the current mechanism before EyePACS. Passing one
fold authorizes an EyePACS pilot; it is not evidence for a paper claim or full
cross-validation.

## Required paper controls and evaluation

1. Static ORIGIN-v3 with its original NLL/RPS objective.
2. Strength-zero V3 trained with the risk-set-balanced proper objective.
3. SAPT trained with the same risk-set-balanced objective.
4. SAPT with ordinary continuation NLL/RPS to separate architecture from loss.
5. Mean-only, max/top-k, quantile/OWA, and parameter-matched pooling controls.
6. Removal or replacement of the tangent-removed PGF concentration statistic.
7. Fixed versus learned probe weights, thresholds, slopes, and gains.
8. Non-nested local states and constant-map controls with matched first moments.
9. Strength-zero identity plus label, model, and spatial-ledger randomization.
10. Top-supporter versus random/bottom deletion at matched scale and support.
11. Receptive-field-aware localization on external masks, if a localization
    claim is made.

Full claims require multiple seeds/folds, patient-disjoint evaluation,
confidence intervals, per-grade recall, macro-F1, balanced accuracy, QWK, MAE,
calibration, cross-dataset transfer, and exact replay checks. Hyperparameters
selected on APTOS must be frozen before EyePACS evaluation.

## Current limitations

- SAPT depends on an already trained and audited ORIGIN-v3 base; it is not yet
  an independently pretrained severity representation.
- The current transport is one tridiagonal Markov step. It encodes ordinal
  adjacency but not a clinical disease-progression process over time.
- PGF concentration is label-supervised and has no named-lesion semantics.
- The singleton certificate covers a small deterministic validation batch;
  only the aggregate transport audit covers the complete validation split.
- Bidirectional flow demonstrates that both numerical directions are used; it
  does not prove that individual upward or downward moves are clinically
  correct.
- Large encoder receptive fields limit pixel-level localization claims even
  when the selected internal cell is exactly faithful to the computation.
- Treatment transport is initialized nonzero. Static V3 and strength-zero
  controls prevent this from being hidden, but the treatment is not an exact
  zero-perturbation warm start.
- APTOS fold-0 development and its promotion thresholds are model selection,
  not an unbiased estimate of generalization.
- No present result supports “foundation model,” universal SOTA, or clinical
  deployment language.

## Primary sources used to set the novelty boundary

- CORN: <https://arxiv.org/abs/2111.08851>
- Additive MIL: <https://papers.nips.cc/paper_files/paper/2022/file/82764461a05e933cc2fd9d312e107d12-Paper-Conference.pdf>
- LoCo counting/localization: <https://proceedings.mlr.press/v97/schroeter19a.html>
- Cardinality-potential MIL: <https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hajimirsadeghi_Visual_Recognition_by_2015_CVPR_paper.pdf>
- CLOC ordinal contrastive learning: <https://openaccess.thecvf.com/content/CVPR2025/html/Pitawela_CLOC_Contrastive_Learning_for_Ordinal_Classification_with_Multi-Margin_N-pair_Loss_CVPR_2025_paper.html>
- Concept bottleneck models: <https://proceedings.mlr.press/v119/koh20a.html>
- Saliency sanity checks: <https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html>
