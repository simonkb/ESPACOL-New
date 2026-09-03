# MOSAIC architecture audit: proof-to-grade path

Date: 2026-09-03

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
and one measurable training risk. Independent fold-0 audits on APTOS and
EyePACS reject analytic deweighting and raw class MAP as operating rules. Raw
`posterior_median` is now locked prospectively for new folds: it improved both
accuracy and MAE on both datasets, while incurring an approximately 0.01 QWK
reduction that must be reported rather than hidden.

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
default. It did not pass on APTOS, and the EyePACS audit supplied no reason to
reverse that conclusion.

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

## APTOS fold-0 result: initial audit

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
(`p=0.302`). APTOS alone therefore did not justify a decoder change;
`posterior_median` was carried into the independently specified EyePACS audit
as the only alternative that improved both accuracy and MAE without the much
larger ordinal degradation of class MAP.

## EyePACS fold-0 result: independent replication

The EyePACS audit also reproduced the historical checkpoint before comparing
the same six fixed proof-only rules:

| Proof-only rule | Acc. | Bal. Acc. | Macro-F1 | QWK | MAE |
|---|---:|---:|---:|---:|---:|
| `rounded_expected` | 81.63 | 52.08 | 0.5225 | **0.8093** | 0.2283 |
| `class_map` | **84.38** | 51.44 | **0.5635** | 0.7865 | 0.2201 |
| `posterior_median` | 83.93 | **52.92** | 0.5593 | 0.7999 | **0.2170** |
| `deweighted_mean_round` | 81.91 | 50.74 | 0.5076 | 0.8070 | 0.2255 |
| `deweighted_class_map` | 84.31 | 49.39 | 0.5450 | 0.7848 | 0.2207 |
| `deweighted_posterior_median` | 83.97 | 50.82 | 0.5466 | 0.7931 | 0.2185 |

Raw `posterior_median` improved accuracy by 2.30 points and reduced MAE by
0.0113 relative to `rounded_expected`; it also improved balanced accuracy and
macro-F1. Its 95 corrected versus 22 newly wrong predictions give an exact
paired `p=5.3144e-12`. The trade-off is a QWK decrease from 0.8093 to 0.7999
(-0.0094). This mirrors APTOS, where accuracy rose by 1.37 points and MAE fell
by 0.0034 while QWK decreased from 0.9030 to 0.8929 (-0.0101).

This cross-dataset consistency is sufficient to lock raw `posterior_median`
prospectively for untouched folds. It is not evidence that the QWK cost is
zero; future reporting must include accuracy, MAE, and QWK. Raw class MAP is
rejected: relative to posterior median, it adds only 0.34 accuracy points on
APTOS and 0.45 on EyePACS, while worsening balanced accuracy, QWK, and MAE on
both (and macro-F1 on APTOS). Analytic deweighting is rejected because it
offers no consistent cross-dataset benefit and causes large APTOS degradation.
The choice is global and prospective, never selected separately per dataset
or fold. Historical checkpoints retain their serialized decision rule.

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

Both fold-0 audits are complete. Neither accessed an outer test split.

The utility is locked to inner validation. It first reproduces the historical
checkpoint metrics, then reports six pre-specified proof-only rules, full
confusions, help-versus-harm counts, probability invariants, and the hard-proof
diagnostics. It writes `decoder_audit/summary.json` and `predictions.csv` next
to each checkpoint.

The same six-rule table remains an audit output, but the decision is now frozen:
new untouched folds use raw `posterior_median`. Do not select whichever row
happens to win on a later fold, and do not introduce dataset-specific decision
rules.

## Changes intentionally not made

- no larger backbone or extra transformer;
- no global classification bypass;
- no validation-tuned thresholds or temperatures;
- no learning-rate or scheduler change;
- no change to the clinical proof definition;
- no change to receptive field, count truncation, or proof tolerance; and
- no stronger imbalance weights.

Only the zero-parameter point decision changes: selected-proof cumulative
probabilities now produce the raw posterior-median grade
`sum_k 1[P(Y>k) >= 0.5]`. Analytic deweighting, raw/deweighted MAP,
deweighted median, and rounded expectation remain audit diagnostics.

Posterior median is a conventional decision rule, not a new contribution. It
preserves proof exclusivity because it consumes only cumulative probabilities
replayed from the retained proof and has no learned or validation-fitted
parameters. The novelty remains the exact cardinality circuit, deterministic
minimum dual proof, and replayable internal intervention.

Those would confound the single question this audit answers. New runs under
the audited source must use a fresh run directory because the decision rule is
part of the checkpoint and implementation identity.
