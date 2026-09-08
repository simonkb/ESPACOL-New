# ORIGIN research source report

Audience: internal project team and supervising professor

Date: 8 September 2026

Status: pre-experimental architecture specification

## Scope

This report asks whether a reusable image architecture can make an ordinal
severity prediction *from* spatial evidence, rather than attach a heatmap to an
otherwise independent classifier. The practical scope is image-level labels
only, initially APTOS and EyePACS diabetic-retinopathy grading, with BUSI as a
three-level cross-modality test. No lesion masks, reports, new annotation, or
clinical adjudication are assumed for training.

## Assumptions

- Labels are ordered states `0,...,K-1`, with `K >= 3` for the claimed new
  mechanism.
- Higher states are reached through adjacent severity boundaries. This is a
  label-space inductive bias, not a claim that a cross-sectional image records
  biological time.
- Image evidence can be represented by nonnegative local support for crossing
  those boundaries. Counter-evidence is represented by missing support, not a
  negative rate.
- A local unit is interpreted at the receptive-field footprint of its encoder
  feature. Exact intervention in the model is not automatically a causal pixel
  intervention in the patient.

## Direct answer

The recommended architecture is **ORIGIN: Ordinal Regional Intervention
Generator Network**. A multi-scale convolutional encoder produces a spatial
field of nonnegative, cumulative boundary evidence. Every local evidence unit
is converted into an upper-bidiagonal continuous-time Markov-chain generator;
the image generator is the sum of those local generators plus a visible null
prior. The class posterior is the finite-time transition distribution from
grade zero:

\[
  p(y\mid x)=e_0^\top\exp\!\left(Q_\varnothing+
  \sum_{s,i}Q_{s,i}(x)\right).
\]

There is no pooled classifier, ordinal-logit residual, or attention readout.
The evidence field is therefore the only image-dependent prediction path.
Removing a reported ledger unit subtracts its generator contribution and
reruns the same decoder exactly. The resulting distribution change is the
computational explanation; its receptive field states the input context on
which that unit depended.

The search found no exact or near-exact published system with this complete
coupling through 8 September 2026. That is not a guarantee that no unpublished
or missed work exists. The safe contribution is the *joint construction*, not
CTMCs, matrix exponentials, ordinal transitions, additive MIL, noisy-OR, or
deletion tests separately.

## Mechanism

### 1. Multi-scale evidence encoder

A pretrained ConvNeXt-Tiny trunk exposes its native feature maps at strides 4,
8, 16, and 32. A separate channel-only projection and one-by-one evidence head
at each raw stage emits `K-1` nonnegative incremental severity atoms; the first
implementation deliberately has no top-down feature fusion:

\[
  a_{s,i,m}=\operatorname{softplus}(z_{s,i,m})\ge0,
  \qquad m=0,\ldots,K-2.
\]

An atom at level `m` supports that boundary and every prerequisite boundary:

\[
  \rho_{s,i,k}=\sum_{m=k}^{K-2}a_{s,i,m},\qquad
  k=0,\ldots,K-2.
\]

Boundary-specific positive calibration and fixed scale allocation give the
local transition-rate contribution

\[
  r_{s,i,k}=\pi_{s,k}\,\operatorname{softplus}(\gamma_k)\,
             w_{s,i}\,\rho_{s,i,k},
\]

where `pi` is a learned simplex over scales and `w` is fixed geometry/validity
weighting. Weights are never renormalized after an intervention.

### 2. Conserved regional generator

Each local rate vector defines an upper-bidiagonal generator contribution:

\[
  [Q_{s,i}]_{k,k}=-r_{s,i,k},\qquad
  [Q_{s,i}]_{k,k+1}=r_{s,i,k}.
\]

All other entries are zero. The image-dependent generator is additive:

\[
  Q(x)=Q_\varnothing+\sum_{s,i}Q_{s,i}(x).
\]

This conservation law makes the output invariant to regrouping or partitioning
an unchanged evidence ledger. It also prevents an unreported global feature
from bypassing the explanation.

### 3. Ordinal transition posterior

The final state is absorbing. At fixed unit exposure,

\[
  \mathbf p(x)=e_0^\top\exp(Q(x)).
\]

Because `Q` is a conservative Metzler matrix, `p` is normalized and
nonnegative. The cumulative probabilities

\[
  c_k=P(Y>k)=\sum_{j=k+1}^{K-1}p_j
\]

are nested by construction. Increasing any rate cannot decrease expected
severity `E[Y]=sum_k c_k`. A posterior median is available when a monotone
discrete decision is required. The accuracy-oriented benchmark protocol
prospectively uses class MAP and logs the median separately because MAP itself
need not be monotone under a rate intervention.

### 4. Exact same-circuit ledger explanations

For any reported ledger unit or set `A`,

\[
  Q^{(-A)}=Q-\sum_{(s,i)\in A}Q_{s,i},\qquad
  p^{(-A)}=e_0^\top\exp(Q^{(-A)}).
\]

The exact boundary effect and expected-grade effect are

\[
  \Delta_{A,k}=P(Y>k)-P^{(-A)}(Y>k),\qquad
  \Delta_A^E=E[Y]-E[Y^{(-A)}].
\]

These are finite changes in the actual prediction circuit, not gradients or
attention weights. Exported certificates record scale and receptive-field
metadata, boundary rates, full prediction, removed prediction, and a replay
checksum. They are exact interventions on stored generator terms—not claims
that the corresponding input pixels were causally removed, because receptive
fields overlap and contextual information can remain in other ledger units.

## Objective

The primary objective is the proper categorical likelihood, with ranked
probability score as an ordinal auxiliary:

\[
  \mathcal L_{\mathrm{NLL}}=-\log p_y,
  \qquad
  \mathcal L_{\mathrm{RPS}}=\frac1{K-1}\sum_{k=0}^{K-2}
    (c_k-\mathbf1[y>k])^2.
\]

An optional total-rate magnitude penalty can be applied after a dense warm-up:

\[
  \mathcal L=\mathcal L_{\mathrm{NLL}}+
  \eta\mathcal L_{\mathrm{RPS}}+
  \lambda_{\mathrm{budget}}\frac1N\sum_n\log(1+\sum_{s,i,k}r_{n,s,i,k}).
\]

This penalty is an ablation, not a prerequisite for correctness, and it does
not favor a sparse spatial allocation at fixed total rate. It also makes the
overall objective no longer a proper scoring rule. The default model therefore
sets it to zero and uses unweighted NLL+RPS with ordinary random sampling.
Empirical calibration is measured separately; neither proper scoring nor an
unweighted loss proves that a finite fitted model is calibrated.

## Why this is specifically ordinal

- The output is not a `K`-way softmax. Probability must flow through adjacent
  ordered states.
- Local high-severity atoms automatically support prerequisite crossings.
- The architecture guarantees nested cumulative probabilities and monotone
  expected severity.
- A local ledger unit has a vector of boundary-specific effects, so an explanation
  states *which transition* it supports, not merely which class it resembles.
- The absorbing final state represents the top reported grade or greater.

For `K=2`, this decoder is exactly noisy-OR:

\[
  P(Y=1)=1-e^{-(\lambda_\varnothing+\sum_i\lambda_i)}
        =1-(1-p_\varnothing)\prod_i(1-p_i)
\]

under `lambda_i=-log(1-p_i)` (including the visible null-prior cause).
Binary classification is therefore an important control, not part of the
architectural novelty claim.

## Literature analysis and disconfirmation

The closest work occupies individual pieces but not the complete mechanism:

- CORAL enforces rank-consistent cumulative outputs through shared classifier
  weights; it does not build a spatial generator
  ([Cao et al., 2020](https://arxiv.org/abs/1901.07884)).
- CORN models conditional continuation between ordinal ranks, but its decisions
  come from global neural logits
  ([Shi et al., 2021](https://arxiv.org/abs/2111.08851)).
- Ord2Seq models ordinal decisions as a sequence, not as an additive local
  transition generator
  ([Wang et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Ord2Seq_Regarding_Ordinal_Regression_as_Label_Sequence_Prediction_ICCV_2023_paper.html)).
- SATOMIL provides threshold-specific spatial aggregation for ordinal MIL, but
  uses a Transformer rather than a conserved generator and exact replay
  ([Shiku et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Shiku_Ordinal_Multiple-Instance_Learning_for_Ulcerative_Colitis_Severity_Estimation_with_Selective_WACV_2025_paper.html)).
- Additive MIL provides exact regional additive accounting for ordinary class
  evidence, not a multi-state ordinal posterior
  ([Javed et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/82764461a05e933cc2fd9d312e107d12-Abstract-Conference.html)).
- Sparse Activations makes fine-grained evidence part of disease grading, but
  retains multiclass linear aggregation
  ([Donteu et al., MIDL 2024](https://proceedings.mlr.press/v227/donteu24a.html)).
- Lesion-Aware Transformers learn DR lesion tokens with image labels, but do
  not provide exact prediction-circuit interventions
  ([Sun et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Lesion-Aware_Transformers_for_Diabetic_Retinopathy_Grading_CVPR_2021_paper.html)).
- Supporting Evidence explicitly studies compatible and sufficient evidence
  for medical severity classification, but its evidence is regularized rather
  than being a Markov generator that exclusively produces the grade
  ([Wang et al., ML4H 2021](https://proceedings.mlr.press/v158/wang21a.html)).
- Pure-birth regression and CTMC classifier ensembles establish that neither
  pure-birth probabilities nor CTMC classification are new alone
  ([Faddy, 1997](https://doi.org/10.1002/BIMJ.4710390405),
  [Du and He, IJCAI 2017](https://arxiv.org/abs/1709.02123)).
- DiDiCM is a direct 2026 warning against claiming the first class-space CTMC
  image classifier; it applies discrete diffusion classification rather than
  spatial pure-birth generator addition
  ([DiDiCM, CVPR 2026](https://arxiv.org/abs/2511.20263)).
- RETFound demonstrates the data and transfer scale normally expected of a
  medical foundation model: 1.6 million retinal images and multiple downstream
  tasks ([Zhou et al., Nature 2023](https://www.nature.com/articles/s41586-023-06555-x)).

Targeted searches included the phrases “matrix exponential ordinal image
classifier,” “pure-birth neural classifier,” “spatial generator
classification,” “image-conditioned phase-type ordinal,” “regional CTMC
intervention,” and close noisy-OR/noisy-MAX equivalents. No exact end-to-end
collision was located. Search convergence is evidence, not a proof of universal
novelty.

### Gap matrix

| Research question | Needed evidence | Primary sources located | Remaining gap | Status |
|---|---|---|---|---|
| Are cumulative or adjacent ordinal decisions already established? | Neural ordinal heads with consistency or continuation guarantees | CORAL, CORN, Ord2Seq | None; this ingredient is occupied | Closed |
| Does prior work aggregate boundary-specific local image evidence? | Ordinal MIL with a distinct spatial aggregator per threshold | SATOMIL | It does not form a conserved rate generator or replay the same decoder | Closed as near collision |
| Are prediction-native local evidence maps already established? | Models whose local evidence directly produces the class score | BagNet, Additive MIL, Sparse Activations | They use ordinary class evidence rather than a multi-state ordinal generator | Closed as near collision |
| Has a CTMC or matrix exponential been used for image classification? | Image classifier whose output uses a Markov process or learned matrix exponential | DiDiCM, Du and He, Intelligent Matrix Exponentiation | No spatial pure-birth generator sum or exact regional replay | Closed as near collision |
| Are pure-birth endpoint distributions already regression models? | Statistical pure-birth count/ordinal models | Faddy; recent pure-birth count algorithms | No learned spatial image decomposition | Closed as near collision |
| Is exact removal/replay already used for interpretable image prediction? | Additive or intervention-based regional explanation | Additive MIL; GCE-MIL; sufficiency/necessity rationale work | No removal of a local generator term followed by the identical ordinal transition decoder | Closed as near collision |
| Does an exact end-to-end duplicate of ORIGIN exist? | Per-location nonnegative generators, additive superposition, upper-bidiagonal pure-birth posterior, exact subtraction/replay | None found across targeted searches through 2026-09-08 | Universal absence cannot be proved; unpublished work may exist | Open, moderate confidence |
| Can this be called a foundation model now? | Broad pretraining corpus, multiple diseases/modalities, transfer and adaptation evidence | RETFound and contemporary medical foundation models | The current project has only task datasets and no pretraining evidence | No |

## Limitations and falsification criteria

1. **Ledger non-identifiability.** Image-level labels constrain boundary-rate
   totals, not unique allocations across scales, cells, atoms, or the learned
   null prior. Compensating head/scale/boundary parameterizations can produce
   the same total, and a wrong region can carry the right evidence. The
   aggregate rate penalty does not resolve this ambiguity.
2. **No negative evidence.** The model can express missing positive support,
   not an explicitly protective region.
3. **Circuit-local, not automatically pixel-causal.** Contextual receptive
   fields mean deleting one ledger entry differs from blacking out pixels and
   re-encoding.
4. **Pure-birth is a parameterization.** At fixed time, flexible rates can
   represent essentially any interior class distribution. The novel part is
   regional conservation and replay, not extra universal approximation power.
5. **Cross-sectional semantics.** Rates are latent evidence-transition rates,
   never biological progression speeds.
6. **Weakly supervised localization can cheat.** Camera, illumination, or
   acquisition artifacts may correlate with grades. Background, coordinate,
   shuffle, crop, and external-domain audits are mandatory.
7. **Accuracy is unknown.** Neither SOTA nor venue acceptance can be guaranteed
   before controlled experiments.

Reject the architecture if any of these occur:

- the exact same encoder with a conventional strong ordinal head beats ORIGIN
  by more than 1.0 percentage point on APTOS fold-0 validation;
- DR fold-0 remains below the existing OPTIC-C reference after a locked
  hyperparameter screen;
- evidence deletion cannot be replayed to numerical tolerance;
- predictions remain accurate after spatially shuffling evidence units without
  their coordinates in a setting where location should matter;
- evidence concentrates on the fixed fundus border/background rather than
  retinal tissue;
- localization does not beat matched weak-supervision baselines when masks are
  used for evaluation only.

## Recommendations

1. Implement ORIGIN in new files and keep all MOSAIC checkpoint-hashed code
   unchanged.
2. Run mathematical/unit tests before any GPU job.
3. Use APTOS fold 0 for a 12-epoch overfit/sanity run, then a 40-epoch viability
   run. Do not use the held-out outer fold during architecture selection.
4. Compare four heads on the same cached encoder features: softmax, CORAL,
   CORN, and ORIGIN. This separates architecture from encoder gains.
5. If APTOS passes, run EyePACS fold 0 with the locked configuration, followed
   by full cross-validation only after the development gate.
6. Treat BUSI (`K=3`) as the minimum cross-modality novelty case. Report binary
   tasks only as non-novel controls because of the noisy-OR equivalence.
7. Call the result a reusable or foundation-compatible severity architecture
   until multi-dataset pretraining and transfer experiments justify “foundation
   model.”

## Claim-to-source ledger

| Claim | Evidence | Scope note |
|---|---|---|
| Cumulative ordinal heads and rank consistency are prior art | [CORAL](https://arxiv.org/abs/1901.07884), [CORN](https://arxiv.org/abs/2111.08851) | Do not claim ordered outputs alone |
| Sequential adjacent ordinal decisions are prior art | [Ord2Seq](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Ord2Seq_Regarding_Ordinal_Regression_as_Label_Sequence_Prediction_ICCV_2023_paper.html) | Do not claim sequence alone |
| Threshold-local ordinal MIL is prior art | [SATOMIL](https://openaccess.thecvf.com/content/WACV2025/html/Shiku_Ordinal_Multiple-Instance_Learning_for_Ulcerative_Colitis_Severity_Estimation_with_Selective_WACV_2025_paper.html) | ORIGIN needs generator/replay distinction |
| Exact additive regional accounting is prior art | [Additive MIL](https://proceedings.neurips.cc/paper_files/paper/2022/hash/82764461a05e933cc2fd9d312e107d12-Abstract-Conference.html) | Do not claim exact local attribution alone |
| Fine-grained weakly supervised disease maps are prior art | [Sparse Activations](https://proceedings.mlr.press/v227/donteu24a.html) | Need ordinal transition semantics |
| Lesion-aware DR representations are prior art | [Lesion-Aware Transformers](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Lesion-Aware_Transformers_for_Diabetic_Retinopathy_Grading_CVPR_2021_paper.html) | Do not claim first DR lesion discovery |
| CTMC image classification is prior art | [DiDiCM](https://arxiv.org/abs/2511.20263), [Du and He](https://arxiv.org/abs/1709.02123) | Do not claim first CTMC classifier |
| Pure-birth count distributions are prior art | [Faddy 1997](https://doi.org/10.1002/BIMJ.4710390405) | Do not claim pure-birth layer alone |
| “Foundation model” implies broad pretraining and transfer | [RETFound](https://www.nature.com/articles/s41586-023-06555-x) | ORIGIN is not yet a foundation model |
| Exact four-part spatial pure-birth/replay collision not found | Targeted audit through 2026-09-08 | Moderate confidence; use “to our knowledge” |
