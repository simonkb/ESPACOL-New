# ORIGIN acceptance-revision artifact contract

The release is intended to make the paper's arithmetic, split identities,
decoder behavior, predictions, and interventions independently checkable
without redistributing the licensed source images.

## Required public artifacts

1. **Model identity**
   - source commit and dirty-tree status;
   - architecture/configuration signatures;
   - SHA-256 of each of the 15 selected checkpoints;
   - checkpoint tensors where repository hosting limits permit, otherwise a
     stable archival DOI and hashes.
2. **Split identity**
   - deterministic split-generation code and seeds;
   - one compressed membership table per fold containing hashed relative image
     identity, hashed patient cluster where available, label, and role;
   - pairwise-disjointness and complete outer-union assertions.
3. **Predictions**
   - every OOF class posterior, four ordinal tails, label, MAP decision,
     posterior mean, fold, and hashed patient cluster;
   - NLL, RPS, Brier, classwise calibration, and threshold reliability bins
     recomputable from those rows.
4. **Interventions**
   - per-image/per-boundary deletion and retention curves;
   - scale-, footprint-, count-, and evidence-mass-matched controls;
   - the exact selection masks or deterministic seeds needed to replay them;
   - checkpoint, protocol, and result hashes.
5. **Numerics and decoder contracts**
   - high-precision independent matrix-exponential and gradient comparison;
   - fixed-horizon CTMC/sequential saturation and monotonicity tests;
   - CTMC semigroup and sequential-semigroup counterexample;
   - explicit statement that exact deletion replay belongs to the additive
     ledger, not uniquely to the matrix exponential.
6. **External and synthetic audits**
   - IDRiD v2 cell-bin alignment and mask-guided internal deletion records;
   - unique lesion and non-lesion cells paired within each image, native
     evidence scale, and seeded repeat using
     $m=\min(n_{\mathrm{lesion}},n_{\mathrm{nonlesion}})$ per side;
   - the dense image--scale census, all 20 paired repeats, and exact nominal
     equality of cell count, scale composition, receptive-field geometry
     exposure, and the pre-atom boundary multiplier;
   - privacy-sanitized image-level inferential units (local paths and raw
     image identifiers removed) so aggregate estimates can be recomputed;
   - shortcut-transform definitions, hidden masks, seeds, cue-only controls,
     factorial interventions, and per-sample posteriors.
7. **Matched baselines**
   - identical split/preprocessing/optimizer policy declarations;
   - independently inner-selected checkpoints and complete per-image outputs;
   - no safety-floor checkpoint substituted for the learned result.

## Privacy and licensing

No fundus pixels are included. Membership tables use hashes rather than local
absolute paths or patient identifiers. Labels, probabilities, internal ledger
values, and deterministic procedural-mask parameters are released because they
are necessary to reproduce the reported analyses. Dataset acquisition remains
subject to the original EyePACS, APTOS, and IDRiD terms.

## Fail-closed publication rule

Every reported table must be generated from a checksummed aggregate artifact.
A missing fold, mismatched checkpoint, changed output schema, duplicate image,
split overlap, incomplete randomized control, or non-finite value invalidates
the aggregate. Partial results can be discussed as pilot evidence but cannot be
silently presented as the preregistered complete analysis.

For the revised release, the top-level assembler accepts only
`origin-idrid-semantic-statistics-manifest-v2`. The historical v1 semantic
manifest is retained as an audit trail, but it is excluded from the package
because its one-sided coarse/all-scale control can under-match dense lesion
lattices and its AP implementation did not group exact score ties. The exactly
matched v1 fine-scale deletion result remains part of the audit trail, but v1
AP and coarse/all-scale deletion estimates are excluded from the publication
bundle in favor of the corrective v2 outputs.
