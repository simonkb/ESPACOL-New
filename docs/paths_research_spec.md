# PATHS: PGF Analysis of Tangent-removed Hotspot Spectra for ordinal severity grading

Status: implementation candidate on `severity-foundation-v2`

Base commit: `b0f4609b8847d862554c84b29dec59bc562d256f` (audited ORIGIN-v3)

## Scope and claim discipline

PATHS is an architecture for image-level ordinal severity labels.  It is not
yet a foundation model: that term requires broad pretraining and transfer
evidence that the current EyePACS/APTOS experiments do not provide.  The
intended initial claim is narrower:

> a foundation-compatible severity architecture whose prediction is computed
> from a nested spatial ordinal field, a tangent-removed
> probability-generating-function focality spectrum, and adjacent
> continuation decisions, with exact full-predictor singleton intervention
> replay.

The literature audit found the individual ingredients in adjacent work:
continuation-ratio ordinal regression (including CORN), additive MIL, learned
pooling, and count/cardinality models.  It did not find the exact coupling
above.  Any paper must therefore use a qualified “to our knowledge” claim and
must establish the contribution with matched controls.  Code alone cannot
guarantee novelty, acceptance, or an accuracy improvement.

## Why V3 is the correct base

ORIGIN-v3 is accurate and structurally auditable, but its visual computation
collapses every multiscale feature lattice to four spatial first moments.  For
boundary `k`, its computation is of the form

\[
\lambda_k=\pi_k+b_kR\sum_s w_{s,k}\frac{1}{n_s}
\sum_i u_{s,i,k}.
\]

Consequently, one strong focal response and many weak diffuse responses can
be indistinguishable.  V3 is also positive-only, and its pure-birth endpoint
loss does not isolate the clinically important “stop at grade 3 versus advance
to grade 4” decision.  On the audited APTOS fold, V3 obtained 86.01% accuracy
and 0.9265 QWK but recalled 0/15 grade-3 cases.  V4/V5 locality restrictions
lost substantial accuracy, while V6--V8 relation modules had negligible or
control-equivalent effects.  PATHS therefore retains deep semantic features,
abandons explicit pair relations, and changes the aggregation and probability
law.

## Mechanism

Let `K` be the number of ordered grades, `s` a feature scale, and `i` a valid
spatial cell.  The pointwise head emits one local ordinal-state simplex.  Its
nested boundary evidence is

\[
u_{s,i,k}=P(S_{s,i}>k),\qquad
1\ge u_{s,i,0}\ge\cdots\ge u_{s,i,K-2}\ge0.
\]

The V3 cumulative atom simplex already supplies this field.  PATHS normalizes
the stored cumulative atoms by their architectural mass cap; it does not call
them lesion probabilities.

For a fixed probe `z` in `(0,1)`, put `a=1-z` and `L=-log(z)`.  PATHS uses
the tangent-removed normalized log-PGF basis

\[
\phi_z(u)=\frac{-\log(1-au)-au}{L-a}.
\]

It satisfies

\[
\phi_z(0)=\phi'_z(0)=0,\quad \phi_z(1)=1,\quad
0\le\phi_z(u)\le u,
\]

and is strictly convex on `(0,1)`.  Therefore, at fixed evidence mass, its sum
is larger for a more concentrated field (by majorization), while the removed
linear tangent prevents diffuse weak responses from reproducing the V3 first
moment.

For each scale and boundary, PATHS records the original evidence mass and its
focality spectrum:

\[
M_{s,k}=\sum_{i\in V_s}u_{s,i,k},\qquad
C_{s,k,z}=\begin{cases}
\dfrac{\sum_{i\in V_s}\phi_z(u_{s,i,k})}{M_{s,k}},&M_{s,k}>0,\\
0,&M_{s,k}=0.
\end{cases}
\]

This construction is bounded in `[0,1]`, invariant to uniform replication of
the spatial lattice, equals one for a saturated singleton, and tends to zero
when unit evidence mass is spread uniformly over increasingly many cells.
Invalid cells contribute neither mass nor focality.  Learned probe weights and
the inherited V3 scale weights are nonnegative simplexes.  Each boundary has a
nonnegative capped gain, so

\[
C_k=\sum_s w_{s,k}\sum_z\alpha_{k,z}C_{s,k,z},\qquad
\delta_k=\gamma g_k C_k,\qquad 0\le\delta_k\le\gamma G.
\]

The correction is one-sided by design: focal evidence may strengthen a
continuation decision, but it cannot manufacture a negative counter-signal.
The V3 posterior `p^(0)` is converted to conditional continuation log-odds

\[
b_k=\log\sum_{j>k}p^{(0)}_j-\log p^{(0)}_k.
\]

PATHS refines those decisions with `eta_k=b_k+delta_k` and

\[
h_k=\sigma(\eta_k)=P(Y>k\mid Y\ge k).
\]

Its normalized grade posterior is

\[
p_y=\left(\prod_{j<y}h_j\right)(1-h_y),\quad y<K-1,
\qquad p_{K-1}=\prod_{j=0}^{K-2}h_j.
\]

There is no global classifier, attention score, concept-logit bypass, or
post-hoc explainer.  Setting `gamma=0` returns the actual V3 output object,
rather than numerically reconstructing it, and is the exact safety/control
path.  Training begins with only the refiner active; later, three explicit
optimizer groups use learning rates `5e-4` (PATHS), `5e-5` (V3 generator), and
`1e-5` (encoder), protecting the audited representation from warm-start
erasure.

## Proper rare-boundary objective

Categorical continuation NLL decomposes into at-risk binary decisions.  PATHS
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

For every fixed positive `c_k`, its population regret is a positive weighted
sum of Bernoulli KL divergences.  It is therefore strictly proper and does not
change the target posterior in the way outcome-dependent class weighting
does.  The initial objective is `L_RS + 0.25 L_RPS`.

## Exact interpretation contract

For explanation, the model enumerates every valid multiscale cell and performs
the same exact singleton intervention on the complete predictor.  Removing a
declared cell:

1. removes its stored V3 local rate entries;
2. removes its stored tangent-removed PGF focality entries;
3. keeps the original baseline mass denominator and all parameters fixed;
4. recomputes the V3 posterior, continuation correction, and final posterior.

The certificate selects the positive supporter whose deletion most reduces
the configured decision margin, breaking ties by the predicted-grade
log-probability reduction.  It records the base-rate, focality, continuation,
and class-probability changes; a canonical replay must match the vectorized
enumeration.  It also records the theoretical/clipped receptive field and one
deterministic scale-matched random deletion.  If no cell has a positive effect,
the certificate says so instead of inventing a witness.

The resulting change is the exact effect of that internal feature-cell
intervention on the full PATHS computation.  It is not a gradient, attention
map, or correction-only ranking.  The contract does **not** claim that a latent
cell is a named lesion, that a cell is pixel-local when its receptive field is
large, or that internal deletion is equivalent to editing input pixels.

## Pre-registered first gate

The first run is APTOS fold 0 on the same split as V3, warm-started only from
the audited V3 checkpoint.  The locked outer test remains unused.

Promotion to EyePACS requires all of the following on inner validation:

- at least 253/293 correct (strictly above V3's 252/293);
- QWK at least 0.9295 (V3 + 0.003);
- no regression in MAE, balanced accuracy, or macro-F1;
- at least 4/15 grade-3 cases correct;
- a learned checkpoint better than the exact strength-zero V3 control;
- nonzero correction and exact full-predictor replay identities on the saved
  certificate.

Failure means the architecture is rejected or revised before full CV.  Passing
one fold is evidence for an EyePACS pilot, not evidence for a paper claim.

## Required paper controls

1. V3 with the original NLL/RPS objective.
2. V3 with the risk-set-balanced proper score.
3. PATHS with ordinary continuation NLL/RPS.
4. PATHS with the risk-set-balanced score.
5. Mean-only, max/top-k, quantile/OWA, and parameter-matched pooled controls.
6. Non-nested local states.
7. Constant spatial maps with the same per-scale mean.
8. Strength-zero identity and label/model randomization.
9. Top-evidence versus random/bottom deletion at matched area.
10. Receptive-field-aware localization on external masks, when used.

Full claims require multiple seeds/folds, patient-disjoint evaluation,
confidence intervals, cross-dataset transfer, per-grade recall, macro-F1,
balanced accuracy, QWK, MAE, calibration, and exact replay checks.

## Primary sources used to set the novelty boundary

- CORN: <https://arxiv.org/abs/2111.08851>
- Additive MIL: <https://papers.nips.cc/paper_files/paper/2022/file/82764461a05e933cc2fd9d312e107d12-Paper-Conference.pdf>
- LoCo counting/localization: <https://proceedings.mlr.press/v97/schroeter19a.html>
- Cardinality-potential MIL: <https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hajimirsadeghi_Visual_Recognition_by_2015_CVPR_paper.pdf>
- CLOC ordinal contrastive learning: <https://openaccess.thecvf.com/content/CVPR2025/html/Pitawela_CLOC_Contrastive_Learning_for_Ordinal_Classification_with_Multi-Margin_N-pair_Loss_CVPR_2025_paper.html>
- Concept bottleneck models: <https://proceedings.mlr.press/v119/koh20a.html>
- Saliency sanity checks: <https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html>
