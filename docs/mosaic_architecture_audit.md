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

The audit found two definite decoder mismatches and one measurable training
risk.

## Confirmed mismatch 1: weighted scores were treated as posteriors

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

For a finite neural model this remains a loss-consistent, pre-specified
correction rather than a guarantee of perfect calibration; the fixed-checkpoint
audit below measures its actual effect. It is a correctness repair, not the
claimed architectural novelty. MOSAIC's contribution remains that the selected
minimum proof is the exclusive grade path and can be replayed interventionally.

This is material, not cosmetic. The observed boundary logit offsets are:

| Run | 0->1 | 1->2 | 2->3 | 3->4 |
|---|---:|---:|---:|---:|
| APTOS fold 0 | +0.0132 | +1.0341 | -0.5493 | +0.3867 |
| EyePACS fold 0 | -0.0005 | +0.1447 | -0.3123 | -0.1415 |

## Confirmed mismatch 2: the point rule did not match model selection

The previous grade was `round(E[Y])`. A posterior mean is the Bayes action for
squared ordinal error, whereas checkpoints and early stopping use exact
accuracy. Class MAP is the Bayes action for exact accuracy when applied to the
natural class posterior. MOSAIC now defaults to the analytically deweighted
class MAP decision.

This correction has:

- zero learned parameters;
- zero validation-fitted parameters;
- no image, feature, or global input;
- no change to proof membership or the cardinality circuit; and
- a complete replay trace from proof scores and training-fold weights.

It is not presented as MOSAIC's central novelty. It removes an evaluation
confound so that the proposed proof architecture is tested fairly.

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

Run the completed checkpoints before another long training job:

```bash
sbatch submit_mosaic_decoder_audit.sh aptos \
  runs/mosaic_aptos_f0_ampfix_policy_v2/fold0/best.pth

# Run this after the active EyePACS job has finished.
sbatch submit_mosaic_decoder_audit.sh dr \
  runs/mosaic_dr_f0_ampfix_policy_v2/fold0/best.pth
```

The utility is locked to inner validation. It first reproduces the historical
checkpoint metrics, then reports six pre-specified proof-only rules, full
confusions, help-versus-harm counts, probability invariants, and the hard-proof
diagnostics. It writes `decoder_audit/summary.json` and `predictions.csv` next
to each checkpoint.

The accuracy-targeted rule was fixed in advance as
`deweighted_class_map`. Do not select whichever of the six rows happens to win
on one fold. The remaining rows diagnose whether the previous plateau came
from calibration/decision mismatch or from the learned representation.

## Changes intentionally not made

- no larger backbone or extra transformer;
- no global classification bypass;
- no validation-tuned thresholds or temperatures;
- no learning-rate or scheduler change;
- no change to the clinical proof definition;
- no change to receptive field, count truncation, or proof tolerance; and
- no stronger imbalance weights.

Those would confound the single question this audit answers. New corrected
runs must use a fresh run directory because the decision rule is part of the
checkpoint and implementation identity.
