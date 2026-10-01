# ORIGIN V3 freeze and evaluation chronology

This record separates development of the final V3 architecture from later
evaluation and retrospective analysis. Dates are repository commit dates in
Asia/Dubai time. The Git object IDs, signed run manifests, checkpoint hashes,
and split signatures—not prose recollection—are the primary provenance.

## Architecture development

| Date | Commit | Event | Evaluation visible at that point |
|---|---|---|---|
| 2026-09-08 | `49c541f52b36def5958549f685cb31efb1286808` | Initial conserved ordinal generator implementation | APTOS and EyePACS inner-validation development runs |
| 2026-09-09 | `ec1dd1c...` | Rates bounded by construction, defining the V3 parameterization | Inner validation only for the controlled V3 reruns |
| 2026-09-10 | `c267648...`, `d7215f9...`, `b0f4609b8847d862554c84b29dec59bc562d256f` | Validation-wide structural audit, FP64 conservation correction, and serializable certificates | Inner-validation audits; no full-CV aggregate existed |

Commit `b0f4609b8847d862554c84b29dec59bc562d256f` is the declared final V3
architecture freeze. Later V4--V8/PATHS branches were exploratory candidates;
none was promoted into the V3 full-CV protocol.

## Locked cross-validation

| Date | Commit/run | Event |
|---|---|---|
| 2026-09-22 | `a9e9f655cd0aa8399084a131e55bdff1796a2477` | Immutable V3 full-CV launcher, split policy, selection rule, and fail-closed aggregate protocol added |
| 2026-09-22--25 | `origin_v3_full_cv_20260922T084547Z` | Five APTOS and ten patient-grouped EyePACS folds trained with inner-validation checkpoint selection, followed by one outer evaluation per selected checkpoint |
| 2026-09-25 | aggregate artifacts | Full OOF metrics and per-image posterior exports completed |

The outer-fold results were not used to alter V3. One failed EyePACS fold-9
attempt was rerun from scratch with the same frozen code, split, seed, and
configuration. Both attempts remain in the audit trail.

## Retrospective mechanism studies

| Date | Commit | Event |
|---|---|---|
| 2026-09-26 | `290e0a0b4d13dc590f14a8bc3c264a111e87dbed` | Fold-9 matched ablation protocol added |
| 2026-09-28 | `424cf2a...` | Audited fresh recovery path added after a non-finite coarse-only run |
| 2026-10-01 | `3190cb4...` onward | Acceptance-revision intervention, shortcut, intrinsic-baseline, numerical, artifact, and IDRiD audits frozen |

The fold-9 study was designed after the full-CV run and uses an already
observed outer fold. It is therefore explicitly retrospective and cannot be
presented as confirmatory evidence or blended with the OOF benchmark estimate.
The acceptance-revision audits do not retrain or select the released V3
checkpoints. Any newly trained baseline has its own inner-selected checkpoint
and a separate release gate.

## What this chronology permits us to claim

- Per-fold checkpoint selection was nested inside each outer training pool.
- The full V3 model and hyperparameters were not selected independently inside
  all outer folds; the architecture was developed on fixed inner-validation
  development splits before full CV.
- The 10-fold EyePACS and 5-fold APTOS OOF numbers are evaluation estimates for
  the frozen V3 configuration, not evidence from fully nested architecture
  search.
- The fold-9 ablations are conditional, retrospective mechanism evidence.
- No result from the October acceptance-revision audits may be used to modify
  the frozen V3 model and then be reported on the same outer predictions as a
  clean confirmatory test.

