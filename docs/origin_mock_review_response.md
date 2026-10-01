# ORIGIN mock-review response and revision matrix

Date: 2026-10-01  
Scope: response to the supplied borderline-reject mock review of the current
CVPR draft  
Purpose: distinguish points that can be answered from existing evidence from
points that require a revised submission and completed experiments

## Executive position

The review is substantially fair. It does not identify a mathematical error in
ORIGIN, invalidate the complete cross-validation results, or show leakage. Its
decisive criticism is an evidence gap: the submitted draft proves exact
frozen-ledger accounting, but does not yet show that the accounting yields a
useful ordinal audit beyond existing additive alternatives. We should not try
to rebut that gap with rhetoric.

The revision therefore narrows the contribution from “the CTMC uniquely makes
the model interpretable” to a **boundary-indexed, nonnegative, conserved
ordinal evidence ledger with exact grouped replay**. The CTMC is one declared
decoder for that ledger. A same-ledger sequential-hazard decoder shares the
fixed-horizon normalization, ordinal monotonicity, exclusivity, and replay
contracts. The CTMC alone adds a homogeneous, shared-clock Markov semigroup
across horizons. Cross-sectional grade labels do not demonstrate that this
cross-horizon property is practically useful, and the revision will not claim
fixed-horizon expressive superiority.

Three status labels are used below:

- **Rebuttable now:** the reviewer inferred a limitation from wording or
  missing artifacts, but committed evidence now resolves it.
- **Conceded and fixed:** the criticism is correct; the scientific claim or
  method description has been changed.
- **Pending evidence:** the criticism cannot be answered until the locked
  cluster experiment completes. No result is presumed.

## Reviewer-issue matrix

| Issue | Scientific response | Status | Verifiable evidence or output contract | Manuscript/release action |
|---|---|---|---|---|
| **W1: the intervention has not shown useful auditing value** | The 40-certificate, mostly grade-0, single-cell pilot is insufficient. Single-cell stability neither proves nor disproves grouped necessity or sufficiency. The decisive tests are validation-wide ranked deletion and retention, with count-, footprint-, coordinate-permutation-, and boundary-mass-matched controls; and a controlled ordinal-shortcut benchmark with exact pixel toggles. | **Conceded; primary evidence pending.** | Final OOF jobs: APTOS array/aggregate `10299463`/`10299464`, EyePACS array/aggregate `10299465`/`10299466`, statistics `10299467`. Required terminal artifacts are each dataset's `oof_intervention_aggregate/audit_manifest.json` and `runs/origin_acceptance/oof_statistics_v1/statistics_manifest.json`. Shortcut jobs `10299451`/`10299452` must produce `runs/origin_shortcut_aptos_20261001T115240Z/APTOS_SHORTCUT_PILOT_RESULTS.json`. | Remove the 40-case pilot as central evidence. Report grouped necessity and sufficiency by boundary, true/predicted grade, correctness, scale, cell/area/mass budget, and receptive-field class. Report raw and locally normalized effects. Promote the claim only if preregistered Gates A and B pass. |
| **W1: semantic/locality interpretation** | Spatial indexing does not imply lesion locality when theoretical receptive fields span the image. IDRiD can test association with annotated lesion bins, but it cannot convert an internal ledger intervention into pixel causality or clinical necessity. | **Conceded and scoped; descriptive audit completed, inferential analysis pending.** | Completed job `10299298`; `runs/origin_acceptance/idrid_semantics/idrid_semantic_summary.json`. Fine-bin AP was 0.18609 at prevalence 0.08154 (AP/prevalence 2.762); coarse-bin AP was 0.35186 at prevalence 0.25158 (ratio 1.648). Mask-guided **internal** deletion exceeded matched random deletion in mean expected-grade change: fine 0.01538 vs. 0.00430, coarse 0.24307 vs. 0.08728, all 0.26235 vs. 0.09135. Image-cluster intervals and multiplicity-aware summaries are reserved for `idrid_semantics/statistics/idrid_statistics_manifest.json`. | Call the result “zero-shot semantic alignment” and “mask-guided internal deletion,” never lesion segmentation, pixel intervention, biological causality, or clinical validation. Label `s16/s32` evidence contextual/global. Do not make inferential claims from the point estimates until the statistics manifest exists. |
| **W2: exact replay is not CTMC-specific** | Correct. For any deterministic decoder `D`, conservation gives `p=D(pi+sum_i l_i)` and deletion gives `p^(-S)=D(pi+sum_{i notin S} l_i)`. The architecture-level replay theorem belongs to the ledger. | **Conceded and fixed.** | The revised decoder section and executable contract distinguish the guarantees. `runs/origin_acceptance/decoder_contract_audit.json`, completed by job `10299326`, verifies both decoders' simplex/monotonicity behavior and the ledger replay attribution. | Rewrite title, abstract, contributions, theorem, and discussion around the ordinal ledger. Never attribute exclusivity, conservation, or exact replay uniquely to `exp(Q)`. |
| **W2: sequential-hazard control undefined** | It is now explicit. For the same rates, `h_k(t)=1-exp(-t lambda_k)`, `P_seq(Y>k)=prod_{j=0}^k h_j(t)`, and `p_y=(1-h_y)prod_{j<y}h_j`, with the final class equal to the product of all hazards. At `t=1`, both it and the CTMC are normalized, coordinatewise monotone ordinal maps with exact ledger replay. | **Rebuttable now after revision.** | Definition is executable in the acceptance baseline code and documented in the revised method. The decoder audit found maximum endpoint reconstruction errors of approximately `2.08e-12` for the CTMC and `1.95e-16` for the sequential map over the fixed audit, so no fixed-horizon expressiveness advantage is asserted. | Put the equations in the main paper, not only supplementary material. Include a one-row guarantee table separating ledger-wide, shared fixed-horizon, and CTMC-only properties. |
| **W2: no useful CTMC-specific distinction** | The CTMC kernel `P(t)=exp(tQ)` uses one homogeneous clock and obeys `P(t+r)=P(t)P(r)`. The natural restartable sequential construction generally does not. This is a mathematical cross-horizon contract, not evidence of biological progression or better cross-sectional grading. Both decoder families can approximate the fixed-horizon monotone simplex map in the audited setting. | **Rebuttable only as a scoped theoretical distinction; practical benefit conceded as untested.** | Completed job `10299326`: CTMC semigroup error `4.55e-15`; minimum observed sequential semigroup violation `0.01119`; simplex errors about `2.16e-12` and `1.94e-16`; monotonicity error `0`. Artifact: `runs/origin_acceptance/decoder_contract_audit.json`. | Retain the CTMC as the primary implementation, but do not sell the shared clock as a demonstrated clinical or accuracy advantage. The empirical contribution must come from the ledger audits and matched comparison. |
| **W3: one-fold/one-seed matched comparisons** | Correct. The retrospective fold-9 study is mechanism exploration, not stable comparative evidence. A matched five-model matrix is required: ORIGIN-CTMC, same-ledger hazards, pooled continuation ratio, ordinal Additive MIL, and Sparse Activations/BagNet. | **Conceded; replicated evidence pending.** | Immutable launcher `scripts/launch_origin_acceptance_baselines.sh` creates a sealed `SUBMISSION.json`; terminal outputs are `<acceptance-root>/full/outer_release/aggregate.json`, `fold_metrics.csv`, and `OUTER_RELEASE_COMPLETE.json`. The release must contain every independently inner-selected learned checkpoint; a V3 safety-floor substitution is forbidden. | Replace literature-only rankings and the single retrospective fold as the main comparison. Report paired fold/seed and patient/image-cluster uncertainty, plus training-seed spread. If ORIGIN does not retain competitive grading, narrow or reject the “without a large penalty” claim. |
| **W3: calibration evidence is only top-label ECE** | Correct. Top-label ECE alone cannot establish ordinal posterior quality. | **Conceded; analysis implementation completed, results pending.** | OOF statistics job `10299467` is required to produce `runs/origin_acceptance/oof_statistics_v1/statistics_manifest.json`. The baseline aggregate must provide NLL, RPS, multiclass Brier, four threshold Brier/ECE values, reliability bins, and 10,000-replicate cluster intervals. | Retain calibration language only if proper scores and threshold calibration support it. If calibration is applied, fit it on inner validation only and transfer it unchanged to the outer fold. |
| **W3: minority-grade failures/general accuracy language** | The reported low recalls are real limitations. High pooled accuracy on imbalanced EyePACS is not equivalent to reliable performance at every grade. | **Conceded and fixed in claim scope.** | Existing OOF confusion matrices and per-grade recall remain authoritative. The matched baseline aggregate will report balanced accuracy, macro-F1, and every grade recall alongside accuracy/QWK/MAE. | Use “competitive overall grading under the stated protocol,” not unqualified “accurate clinical grading.” Include failure strata in the intervention audit rather than hiding them. |
| **W4: apparent fold-9/protocol ambiguity** | The final bounded V3 architecture was frozen on 2026-09-10 at `b0f4609...`. The immutable full-CV launcher was added on 2026-09-22 at `a9e9f65...`; complete OOF evaluation ran September 22--25. The retrospective fold-9 ablation protocol was added on September 26 at `290e0a0...`. Fold 9 was therefore visible before the later ablation design, not used to alter V3 before full CV. Architecture search was still not fully nested and must not be described as such. | **Rebuttable now, with an explicit limitation.** | `docs/origin_freeze_chronology.md`; signed run/checkpoint/split manifests and hashes. | Replace the ambiguous sentences with the exact chronology. Call full CV an evaluation of one frozen configuration, not nested architecture-selection evidence; label fold-9 ablations retrospective throughout. |
| **W4: split/checkpoint/prediction artifacts unavailable** | The omission is a release defect, not a reason to claim reproducibility. The anonymous package must make identities and outputs independently checkable without redistributing licensed images. | **Conceded; base manifest completed, full package assembly pending.** | Completed job `10299251`: 15 checkpoints and 15 split-membership artifacts indexed; manifest checksum `aaef9abda5139bfdfa27393cf16a788d6d128babb0ad7599be0a14d4ab23b043`; root `runs/origin_acceptance/artifact_manifest/`. Required contents are specified in `docs/origin_acceptance_artifacts.md`. | Release deterministic split code plus hashed memberships, checkpoint hashes/tensors or archival links, all OOF posteriors, intervention certificates, exact decoder/control definitions, and executable reproduction. A manifest alone does not answer W1. |
| **W4: matrix-exponential implementation not independently validated** | This is now independently checked against a reference over normal and adversarial bounded rates, including gradients. | **Rebuttable now after new audit.** | Completed job `10299250`; `runs/origin_acceptance/numerical_decoder_audit.json`. Maximum absolute/relative posterior errors were `1.49e-14`/`9.41e-14`, maximum row-sum error `1.62e-14`, maximum absolute gradient error `1.33e-15`, and maximum relative gradient error `1.91e-8` (the relative statistic includes near-zero derivatives). | State Taylor degree and scaling rule precisely and include the independent audit protocol/range in the supplement. Do not substitute ledger-replay error for posterior/gradient numerical validation. |
| **Page limit** | The ten-page content draft is not submission-ready under the last verified eight-page rule. | **Conceded and fixed at production stage.** | The revision plan specifies an eight-page main body until the target-year author kit says otherwise. | Main paper: ledger theorem, one architecture figure, full grading result, replicated matched comparison, shortcut result, grouped replay result, and limitations. Move exploratory ablations, extended audits, calibration plots, provenance, and certificates to the supplement. |

## Direct answers to the reviewer's five questions

### 1. What useful auditing result exists beyond arithmetic replay?

**Response today:** one completed external analysis supplies descriptive but
limited evidence. On 81 IDRiD segmentation images, ledger bins are enriched
for annotated lesions, and deleting internally stored entries whose declared
bins overlap lesion masks changes expected grade more than matched random
deletion (point estimates above). This is a zero-shot association and an
internal intervention. It is not a pixel counterfactual, a causal claim, or a
clinical validation result.

The central response is still pending. The complete OOF audit tests ranked
deletion and retain-only sufficiency against four controls on all locked
predictions, stratified by boundary, grade, correctness, scale, and error type.
The controlled-shortcut benchmark separately tests whether the ledger can
identify a known ordinal cue's location and boundary and whether exact internal
deletion agrees with paired pixel toggles. We will answer “yes” only if the
precommitted utility gates pass; otherwise the revised paper will withdraw the
useful-spatial-explanation claim.

### 2. What is the sequential-hazard decoder, and what does it lack?

It applies `h_k(t)=1-exp(-t lambda_k)` to the identical conserved rates and
forms a continuation posterior from products of the adjacent hazards. It does
**not** lack normalization, coordinatewise ordinal monotonicity, a classifier-
bypass prohibition, or exact frozen-ledger replay. Those are shared contracts.

What it lacks is the CTMC's one-parameter homogeneous Markov transition kernel:
the CTMC satisfies `P(t+r)=P(t)P(r)` under one shared clock, whereas the natural
restartable sequential kernel fails that identity in general. The completed
contract audit verifies this distinction numerically. Because the datasets
contain one cross-sectional grade per image, we do not claim the semigroup
improves grading, calibration, progression modeling, or fixed-horizon
expressiveness.

### 3. Which outer-fold results were visible before design freeze?

No complete OOF result existed when bounded V3 was frozen on 2026-09-10. The
full-CV machinery was frozen on 2026-09-22 and produced outer results during
September 22--25 without changing V3. Fold 9 and the full aggregate were visible
when the September 26 retrospective ablation suite was designed. Consequently:

- the V3 OOF scores are evaluation of a frozen configuration;
- the architecture/hyperparameter search is not fully nested;
- the fold-9 ablations are retrospective and cannot be treated as confirmatory;
- no October audit may be used to change V3 and then be reported on the same
  OOF predictions as a clean prospective result.

The exact commit/run chronology is in `docs/origin_freeze_chronology.md`.

### 4. Are additional-fold/seed and intrinsic-baseline results available?

Not yet. The old one-fold/one-seed table remains exploratory. The new frozen
suite trains the five models named above under identical memberships,
preprocessing, augmentation, optimizer policy, checkpoint selection, and outer
release discipline. It also preserves each learned result even when it is
worse than V3, so a safety floor cannot hide a failed baseline.

This question is resolved only by the checksummed
`full/outer_release/aggregate.json` and `OUTER_RELEASE_COMPLETE.json` generated
under the baseline suite's sealed `SUBMISSION.json`. Until those exist, the
paper must show a reserved result slot rather than a claimed stable tradeoff.

### 5. Can the reported audit be inspected through artifacts?

Partly now, fully only after assembly of the anonymous supplement. The completed
artifact audit indexes all 15 frozen checkpoints and 15 split-membership files
and seals them under the checksum reported above. The numerical and decoder
contract JSON files make the implementation claims executable. The final
package must additionally contain:

- hashed split and patient-cluster membership, with disjointness checks;
- checkpoint tensors where permitted, otherwise stable archival links plus
  SHA-256;
- every OOF class posterior, tail probability, label, decision, and expected
  grade;
- every grouped intervention curve, random-control seed/mask, and protocol
  hash;
- shortcut transforms, hidden masks, factorial pixel interventions, and clean
  controls;
- exact CTMC and sequential decoder definitions; and
- one top-level fail-closed manifest.

No raw licensed fundus pixels need to be redistributed. Inspectability repairs
W4, but artifacts alone cannot repair W1 or W3.

## Guarantee-attribution table

| Property | Conserved ledger | CTMC decoder | Sequential-hazard decoder | Empirical status |
|---|:---:|:---:|:---:|---|
| Nonnegative, boundary-addressable local entries | yes | -- | -- | structural |
| Exclusive prediction path through summed ledger and prior | yes | consumes it | consumes it | structural |
| Exact grouped deletion/retention replay with frozen entries | yes | deterministic replay | deterministic replay | structural and numerically audited |
| Normalized class posterior | -- | yes | yes | structural/numerically audited |
| Coordinatewise monotone ordinal tails | -- | yes | yes | structural/numerically audited |
| Fixed-horizon saturation / broad monotone-simplex coverage | -- | yes | yes | audited; no CTMC superiority claim |
| Homogeneous shared-clock semigroup across horizons | -- | yes | no in general | CTMC-specific mathematical contract |
| Better cross-sectional accuracy or calibration | -- | unestablished | unestablished | pending matched replicated suite |
| Useful localized ordinal audit | enables test | decoder-independent in principle | decoder-independent in principle | pending Gates A/B; IDRiD provides only scoped semantic association |

## What can be said in a rebuttal now

A concise defensible response is:

> We agree that the original draft over-attributed exact replay to the CTMC
> and under-evaluated auditing utility. Exact replay follows from the conserved
> boundary-indexed ledger and any declared deterministic decoder. We now define
> the same-ledger sequential control explicitly and separate shared guarantees
> from the CTMC-only homogeneous semigroup; we make no fixed-horizon or clinical
> progression advantage claim. We also clarify that V3 was frozen before the
> full-CV run, whereas the fold-9 ablations were designed retrospectively after
> that fold was visible. Independent numerical checks reproduce the posterior
> to maximum absolute error 1.49e-14 and gradients to 1.33e-15 absolute error,
> and a checksummed artifact manifest indexes all 15 checkpoints and split
> memberships. These corrections resolve the definition, chronology, and
> verification concerns. We do not claim that they alone resolve the central
> significance concern: complete grouped-replay, shortcut, and replicated
> intrinsic-baseline experiments are required for the revised submission.

This response should not cite pending jobs as positive evidence. If a venue's
rebuttal policy prohibits new experiments, the completed October audits should
be described as clarification/supporting checks only where allowed; otherwise
they belong in a revised submission.

## Promotion rules for the revised paper

1. **Auditability claim:** promote only if both OOF deletion and retention beat
   the locked matched controls with the preregistered effect-size and interval
   gates, and the shortcut benchmark recovers boundary-specific known cues.
2. **Semantic claim:** report IDRiD only as external association plus
   mask-guided internal deletion. Require image-cluster uncertainty before
   inferential wording.
3. **Competitive-performance claim:** require the complete five-model matched
   fold/seed aggregate. Report failures and seed dispersion, not only the best
   checkpoint.
4. **Calibration claim:** require NLL, RPS, multiclass Brier, and all four
   threshold reliability analyses; top-label ECE is supplementary.
5. **CTMC claim:** restrict it to the homogeneous shared-clock semigroup. A
   clinical progression claim would require longitudinal data and is outside
   this paper.
6. **Reproducibility claim:** require a complete checksummed supplement. Any
   missing fold, mismatched hash, split overlap, incomplete random repeat, or
   non-finite output invalidates the aggregate.
7. **Venue decision:** if the utility gates fail, do not disguise exact
   accounting as useful explanation. Submit a narrower fundus-focused paper or
   develop the predeclared local/context successor on fresh evaluation data.

## Bottom line

The strongest parts of the review can be answered now: the chronology is
cleaner than the draft implied; the numerical decoder is accurate on its
declared bounded domain; the sequential decoder is now defined; and the
ledger-versus-decoder guarantee boundary is explicit. The reviewer is still
correct on the two issues that determine CVPR significance: useful auditing
behavior and replicated comparison against the closest intrinsic alternatives.
Those questions are intentionally fail-closed and remain pending until their
named aggregate manifests exist.
