# ORIGIN-v3 EyePACS Fold-9 Ablation Protocol

## Scope and status

This experiment is a controlled, one-fold mechanism study for the final
ORIGIN-v3 architecture. It uses EyePACS fold 9 because the fold is the requested
stress-test fold and has already completed under the full cross-validation
protocol. Consequently, this is a **retrospective ablation on a previously
observed fold**, not a new confirmatory benchmark. The ablation may support
mechanistic claims; the ten-fold ORIGIN-v3 run remains the source of the main
performance estimate.

Every variant is trained independently from the same ImageNet initialization
policy. No variant is warm-started from ORIGIN-v3. Outer split identity and
class counts are checked for protocol integrity, but outer images and
predictions are not used for optimization or checkpoint selection. The outer
fold is evaluated once, in one suite-level release job, after all selected
variants have produced audited validation-only completion markers.

## Frozen common protocol

- Dataset: EyePACS / Kaggle DR 2015, five grades.
- Fold: 9 of the existing ten-fold patient-separated protocol.
- Split sizes: 28,456 train; 3,162 inner validation; 3,508 outer test.
- Input: 640 x 640 canonical fundus image and valid-retina mask.
- Encoder: ImageNet-pretrained ConvNeXt-Tiny.
- Optimizer, learning rates, freeze schedule, augmentation, AMP policy,
  checkpoint rule, seed, batch size, and epoch budget are identical across
  variants.
- Model construction is followed by a reset to one dedicated training-stream
  seed, so variant-dependent parameter counts cannot alter shuffling, worker
  seeds, or augmentation streams.
- All pooled controls and direct-atom controls are analytically initialized to
  the nominal full-ORIGIN boundary posterior. The comparison therefore does
  not confound the tested mechanism with a uniform-versus-grade-0 starting law.
- The sensitive head path executes in FP32 for every model; normalized
  posterior arithmetic executes in FP64.
- Checkpoint selection: validation accuracy, then validation QWK, then lower
  validation loss. The outer partition never selects a checkpoint.
- Primary decision: class MAP from the normalized five-grade posterior.
- Primary metric: accuracy. Secondary metrics: QWK, MAE, balanced accuracy,
  macro F1, ECE, and expected-grade MAE.
- Uncertainty: 10,000 paired patient-cluster bootstrap replicates. The two eyes
  of one patient always remain in the same bootstrap unit.

## Registered variants and questions

| Variant | Controlled change | Question answered |
|---|---|---|
| `origin_full` | Complete ORIGIN-v3, rerun contemporaneously | Reference for every paired comparison |
| `pooled_softmax` | All four masked feature scales are globally pooled and decoded categorically | Is the spatial evidence generator better than a matched global classifier? |
| `pooled_cumulative_logit` | Same pooled representation with a shared score and ordered cumulative thresholds | Are gains explained by generic ordinal decoding alone? |
| `ledger_sequential_hazard` | Identical ORIGIN local rate ledger; replace the pure-birth matrix exponential with adjacent sequential hazards | Does the pure-birth generator add value beyond the evidence ledger? |
| `origin_simplex_direct` | Keep the same null simplex but remove reverse-cumulative prerequisite compilation | Does cumulative disease-evidence sharing matter? |
| `origin_nll_only` | Set the RPS weight to zero | Does the ordinal proper-scoring term contribute? |
| `origin_fine_only` | Retain strides 4 and 8 only | Is fine lesion evidence sufficient? |
| `origin_coarse_only` | Retain strides 16 and 32 only | Is broad context sufficient? |
| `origin_independent_sigmoid` | Replace competing cumulative simplex atoms with independent boundary sigmoids | Does conserved, coupled evidence matter? |
| `pooled_conditional` | Same pooled representation with conditional continuation probabilities | Is performance reproduced by another global ordinal factorization? |

The pooled heads are deliberately described by their implemented mathematical
form rather than branded as exact implementations of CORAL or CORN. They match
the encoder, input, multiscale feature access, objective, and training policy,
but they do not possess a replayable spatial evidence ledger. The release
artifacts mark their certificate status as not applicable instead of fabricating
an interpretability result.

## Predeclared interpretation

The main architectural claim requires `origin_full` to outperform the
registered pooled controls under the paired outer evaluation. The
cumulative-evidence claim is
tested by `origin_simplex_direct` and `origin_independent_sigmoid`. The decoder
claim is isolated by `ledger_sequential_hazard`, whose source ledger is exactly
the same computation as the full model. Fine/coarse variants quantify scale
dependence, while `origin_nll_only` isolates the RPS contribution.

No single fold or training seed establishes a population-wide superiority
claim. Report point
estimates and paired confidence intervals, including unfavorable or
inconclusive comparisons. Promote an ablation conclusion to the paper only
when it agrees with the implemented mechanism and is not based on choosing a
variant after opening the outer results.

## Execution and artifacts

The launcher creates an immutable detached worktree and a checksummed locked
protocol, then submits:

1. structural and GPU preflight;
2. a validation-only Slurm training array;
3. one fail-closed outer-release job after all array tasks terminate;
4. an after-any aggregation/audit job.

The final outputs are `FOLD9_ABLATION_RESULTS.json`, `.csv`, and `.md` under the
experiment root, plus per-image posteriors and checksummed manifests for every
variant. If any worker fails, the release refuses to evaluate any outer image.

After the original release and aggregation jobs have exited, failed or
preempted workers can be resumed without changing the signed protocol:

```bash
bash scripts/retry_origin_fold9_ablation_variants.sh \
  /absolute/path/to/experiment variant [variant ...]
```

The retry command must name every variant lacking a completion marker. It
verifies the immutable snapshot, implementation signature, split, numerical
configuration, and `last.pth` checkpoint before submitting the resume array.
The suite-level outer release remains locked until all variants pass their
validation-only audit. A resumed worker's time and peak-memory fields are
explicitly marked as resumed-segment telemetry and must not be used for
cross-variant efficiency comparisons.
