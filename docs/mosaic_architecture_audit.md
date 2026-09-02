# MOSAIC architecture audit: proof-to-grade path

Date: 2026-09-02

Scope: code-level audit of the implemented prediction path and the completed
APTOS/EyePACS pilot behavior. This audit does not use either outer test split
and does not add a conventional classifier.

## Verdict

The exclusivity claim is implemented correctly:

```text
full canvas -> bounded local encoder -> pointwise ordinal states
-> cardinality circuit -> selected dual proof -> continuation cascade -> grade
```

There is no pooled feature, CTOT token, CORAL logit, text branch, or residual
classifier capable of bypassing the reported proof. The main architectural
novelty therefore remains intact.

The audit found one theoretical decoder caveat, one metric-selection tension,
and one measurable training risk. The completed APTOS fold-0 audit rejects
analytic deweighting as the operating decoder. The safe default remains the
historical `rounded_expected` rule while the independent EyePACS audit is
pending.

## Theoretical caveat: weighted scores are not calibrated posteriors

For boundary `k`, the implemented loss applies weights `w[k,0]` to a stop and
`w[k,1]` to an advance. If the natural continuation posterior is
`pi=P(Y>k | X,Y>=k)`, the population optimum of that weighted log loss is

```text
c* = w_advance*pi / (w_advance*pi + w_stop*(1-pi)).
```

Consequently `c*` is a cost-sensitive score. Feeding it directly into the
class-probability cascade is not probabilistically correct. At the weighted
population optimum, the exact inverse link is

```text
c = w_stop*c* / (w_stop*c* + w_advance*s*)
s = w_advance*s* / (w_stop*c* + w_advance*s*).
```

Equivalently, `logit(c) = logit(c*) + log(w_stop/w_advance)`. The correction
is evaluated in log space so a directly computed, extremely small stop
probability is not destroyed by `1-c` cancellation.

For a finite neural model this is only a loss-consistent, pre-specified
candidate correction, not a guarantee of better calibration. In particular,
the derivation assumes that the learned score is at the weighted population
optimum; representation error, regularization, the projected/dense compound
loss, and finite-sample fitting can violate that assumption. The candidate
therefore had to pass the fixed-checkpoint audit before it could become the
default. It did not pass on APTOS.

This decoder question is not the claimed architectural novelty. MOSAIC's
contribution remains that the selected minimum proof is the exclusive grade
path and can be replayed interventionally.

This is material, not cosmetic. The observed boundary logit offsets are:

| Run | 0->1 | 1->2 | 2->3 | 3->4 |
|---|---:|---:|---:|---:|
| APTOS fold 0 | +0.0132 | +1.0341 | -0.5493 | +0.3867 |
| EyePACS fold 0 | -0.0005 | +0.1447 | -0.3123 | -0.1415 |

## Metric tension: the point rule and checkpoint metric

The historical grade is `round(E[Y])`. A posterior mean is aligned with
ordinal distance, whereas checkpoints and early stopping currently use exact
accuracy. Class MAP would be the Bayes action for exact accuracy only if its
inputs formed a trustworthy class posterior. That condition is not established
for these cost-sensitive finite-model scores, so metric matching alone does not
justify changing the decoder.

Every audited rule has:

- zero learned parameters;
- zero validation-fitted parameters;
- no image, feature, or global input;
- no change to proof membership or the cardinality circuit; and
- a complete replay trace from proof scores and training-fold weights.

The alternatives are retained as diagnostics, not architectural branches or
post-hoc choices.

## APTOS fold-0 result: decoder promotion rejected

The audit exactly reproduced the epoch-13 checkpoint on all 293 inner-
validation images. Results were:

| Proof-only rule | Acc. | Bal. Acc. | Macro-F1 | QWK | MAE |
|---|---:|---:|---:|---:|---:|
| `rounded_expected` | 82.25 | **61.33** | **0.6322** | **0.9030** | 0.2150 |
| `class_map` | **83.96** | 59.67 | 0.6112 | 0.8719 | 0.2253 |
| `posterior_median` | 83.62 | 61.00 | 0.6321 | 0.8929 | **0.2116** |
| `deweighted_mean_round` | 79.86 | 53.36 | 0.5500 | 0.8783 | 0.2526 |
| `deweighted_class_map` | 81.23 | 52.31 | 0.5286 | 0.8671 | 0.2526 |
| `deweighted_posterior_median` | 81.23 | 52.92 | 0.5462 | 0.8694 | 0.2491 |

Analytic deweighting reduced every reported metric relative to
`rounded_expected`, so it is rejected as the default. Raw `class_map` improved
accuracy by 1.71 points, but simultaneously worsened balanced accuracy,
macro-F1, QWK, and MAE. Promoting it from the same validation fold on which it
was discovered would be post-hoc rule selection. Moreover, neither apparent
accuracy gain is statistically compelling on this single fold: relative to
`rounded_expected`, `posterior_median` has 6 corrected versus 2 newly wrong
predictions (exact paired test, `p=0.289`), and `class_map` has 10 versus 5
(`p=0.302`). `rounded_expected` therefore remains the safe operating rule
pending the independently specified EyePACS audit. All six rules remain logged
for diagnosis only.

## Measurable risk: empty-proof advance gradients

The minimum-prefix rule permits an empty proof when the dense transition is no
larger than the sufficiency tolerance. Its projected continuation is then
exactly zero. If the label requires advancing at that boundary, the clamped
projected log-likelihood has zero primary recovery gradient. The existing
dense auxiliary term still provides a gradient at weight 0.1, so this is not a
total dead end, but it can starve rare late boundaries.

The trainer and fixed-checkpoint audit now report, for every boundary:

- empty-proof rate;
- empty-proof rate among advance targets;
- empty-proof rate among stop targets; and
- exact-zero transition rate among advance targets.

No dense-loss or proof-tolerance change is justified until those rates are
known. If they are non-trivial, the next isolated ablation should replace the
hard-prefix gradient with a dense-ledger surrogate while retaining the exact
selected proof in the forward pass. That intervention must not be mixed into
the decoder comparison.

The complete decoder audit additionally requires positive stop and advance
support at every configured boundary. Training fails before epoch 1 if a fold
omits either outcome, rather than emitting incomplete deweighting diagnostics
or silently changing the declared number of grades.

## Validation-only audit

APTOS fold 0 has been completed. Run the EyePACS audit before making any
cross-dataset decoder claim:

```bash
sbatch submit_mosaic_decoder_audit.sh dr \
  runs/mosaic_dr_f0_ampfix_policy_v2/fold0/best.pth
```

The utility is locked to inner validation. It first reproduces the historical
checkpoint metrics, then reports six pre-specified proof-only rules, full
confusions, help-versus-harm counts, probability invariants, and the hard-proof
diagnostics. It writes `decoder_audit/summary.json` and `predictions.csv` next
to each checkpoint.

No alternative rule is promoted from the APTOS table. Do not select whichever
of the six rows happens to win on one fold. The EyePACS audit is an independent
diagnostic of whether the APTOS ordering generalizes; it is not permission to
choose a dataset-specific decoder after inspecting both tables.

## Changes intentionally not made

- no larger backbone or extra transformer;
- no global classification bypass;
- no validation-tuned thresholds or temperatures;
- no learning-rate or scheduler change;
- no change to the clinical proof definition;
- no change to receptive field, count truncation, or proof tolerance; and
- no stronger imbalance weights.

The operating decoder also remains unchanged: selected-proof continuation
scores are converted to a normalized ordinal distribution and the prediction
is `round(E[Y])`. Analytic deweighting, MAP, and posterior-median rules remain
available only in the audit report.

Those would confound the single question this audit answers. New runs under
the audited source must use a fresh run directory because the decision rule is
part of the checkpoint and implementation identity.
