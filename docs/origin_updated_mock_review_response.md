# ORIGIN response to the updated mock review

Date: 2026-10-01

This note records the response to the second mock review of
`paper/origin_cvpr.tex`.  It is an internal decision record, not a rebuttal and
not evidence that a pending experiment succeeded.

## Overall assessment

The updated review is materially more favorable than the first review.  It
accepts that the ledger, rather than the CTMC, is the source of exact replay;
it accepts the defined sequential-hazard control, the numerical audit, the
clarified chronology, and the scoped novelty claim.  It does not identify a
verified prior work with ORIGIN's complete boundary-indexed conserved-ledger
construction.

The remaining decision is empirical: does the ordinal ledger provide a useful
audit beyond close intrinsic additive/local predictors while retaining
competitive grading?  Pending results must not be described as evidence.

## W1: IDRiD protocol and matching

The implementation already resolves most protocol questions that were omitted
from the main paper:

- all ten independently inner-selected EyePACS outer-fold checkpoints evaluate
  every IDRiD image; IDRiD is not used for training, tuning, or checkpoint
  selection;
- the image-derived dominant-field crop is applied identically to the image and
  official MA/HE/EX/SE masks; RGB is resized bilinearly and masks by nearest
  neighbor;
- a lattice cell is positive when it contains any pixel from the union lesion
  mask;
- the ranking score is the stored local rate at the true-grade-selected ordinal
  boundary;
- AP is computed within each image, checkpoint, and scale, then checkpoints and
  scales are aggregated within image before image-cluster inference; bins are
  never pooled across images for the primary estimand;
- the random deletion control is drawn within the same image and scale with 20
  deterministic repeats.

A corrective audit found two genuine v1 issues.  First, the AP helper used a
stable element-wise ranking rather than grouping exact score ties at a shared
threshold.  Exact ties could therefore make AP depend on spatial flattening
order, so the v1 AP values are provisional.  Second, the deletion helper
silently used `min(n_lesion, n_nonlesion)` only for the control.  Fourteen coarse
image-by-scale cases have more lesion-positive than non-lesion cells (mostly
stride 32, with one stride-16 case), so those controls delete fewer cells than
the guided intervention.  Fine scales are unaffected: their maximum positive
fractions are 0.308 at stride 4 and 0.407 at stride 8.  Therefore:

- the within-image aggregation design remains valid, but AP must be recomputed
  with threshold-grouped ties;
- the fine-scale replay comparison remains exactly count-matched;
- the v1 coarse/all-scale replay lift is exploratory and must not be used as a
  matched-control claim;
- a separate checksum-sealed v2 reanalysis will use
  `m=min(n_lesion,n_nonlesion)` on both sides, sampling `m` lesion and `m`
  non-lesion cells separately within every image and scale for every paired
  repeat, and will compute threshold-grouped AP.

V2 matches cell count, scale composition, conserved geometry exposure, and the
pre-atom boundary multiplier.  It does not match retinal anatomy or actual
removed rate mass; the latter is the outcome being audited.

No model retraining is needed for the corrective audit.

## W2: intrinsic comparators and concrete auditing utility

Do not launch another architecture search.  The requested comparisons are
already represented by sealed jobs:

- Gate A (`10299535` after EyePACS aggregation `10299466`) evaluates grouped
  necessity/sufficiency on out-of-fold EyePACS and APTOS evidence with multiple
  matched controls;
- Gate B (`10299570`, expanded worker array `10299569`) compares ORIGIN-CTMC,
  same-ledger sequential hazards, pooled conditional prediction, ordinal
  Additive-MIL, and sparse BagNet on controlled shortcut families, unseen cue
  conditions, internal deletion, pixel toggles, boundary selectivity,
  localization, negative controls, and clean grading retention;
- the matched grading suite (`10299854` -> `10299855` -> `10299856` ->
  `10299857`) runs the same five models over 15 dataset/fold settings and three
  seeds without a V3 checkpoint floor.

The paper may promote a useful-audit claim only if the predeclared Gate B
criteria pass.  A failed gate is a scientific result, not a reason to tune a
post-hoc V9.  If the close intrinsic controls match ORIGIN, the claim and venue
must be narrowed.

## W3: supplement and artifacts

Package job `10299968` was canceled because its immutable snapshot and input
contract predated the corrective IDRiD-v2 audit.  A replacement package must be
launched from the v2-aware commit and remain dependency-gated on IDRiD v2,
Gates A/B, and the matched-baseline release.  The replacement is designed to
bundle protocols, numerical and decoder audits, anonymous split memberships,
privacy-safe matched-baseline predictions, continuation provenance, and
aggregate intervention results.

This package alone does not embed checkpoint tensors or all privacy-safe
per-image evidence.  Unless stable external archives with hashes are supplied,
the manifest must report that independent re-inference is unavailable.  The
paper must not claim that hashes alone enable re-inference.  A separate
supplementary PDF is also required; references to a supplement in the main
paper are not sufficient.

## Immediate manuscript corrections

1. Describe the IDRiD estimand, preprocessing, checkpoint ensemble, within-image
   AP, and scale-specific controls explicitly.
2. Qualify decoder coverage on the deployed bounded-rate domain; the sampled
   inversion audit is not a proof of universal interior-simplex saturation.
3. Replace the reversed conclusion wording with “no demonstrated grading
   advantage for ORIGIN over pooled heads.”
4. Redraw Figure 1 so exact replay originates at the conserved ledger, with the
   CTMC and sequential hazard shown as declared decoder choices and the
   semigroup attached only to the CTMC.
5. Move a compact matched grading-plus-audit comparison into the main paper
   after the sealed results become terminal.

## Submission decision rule

- **Proceed with the broad CVPR auditability claim** only if Gate B passes and
  the matched suite shows acceptable grading retention.
- **Retain a narrower CVPR/MICCAI claim** if the ledger remains exact and
  spatially associated but does not outperform close intrinsic controls on the
  controlled audit.
- **Do not redesign the architecture based on the updated review.**  The review
  accepts the scoped novelty; the remaining question is whether the completed
  evidence establishes significance.
