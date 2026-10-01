# ORIGIN acceptance revision plan

Date: 2026-10-01  
Target: CVPR 2027 (paper registration 2026-11-10 AOE; submission 2026-11-16 AOE)  
Status: pre-registration plan; no experiment described below should be inspected before its protocol and pass/fail rule are committed.

## 1. Decision

The mock rejection is scientifically fair. It is not a rejection of the mathematics or the grading result. It is a rejection of the present significance argument:

1. the paper proves exact internal accounting but does not yet show a useful audit;
2. the same-ledger sequential-hazard control also has a normalized ordinal posterior, no classifier bypass, monotone deletion, and exact ledger replay;
3. the matched comparisons cover only one fold and one training seed; and
4. important artifacts, chronology, numerical checks, and page-limit compliance are missing from the submission package.

A rebuttal alone cannot repair items 1 and 3. The submission needs new evidence and a narrower, correct central claim.

## 2. What to rebut and what to concede

### Correct in rebuttal

- The final bounded V3 architecture was frozen in commit `b0f4609b8847d862554c84b29dec59bc562d256f` on 2026-09-10. The immutable full-CV protocol was added in `a9e9f655cd0aa8399084a131e55bdff1796a2477` on 2026-09-22. The retrospective fold-9 ablation suite was added in `290e0a0b4d13dc590f14a8bc3c264a111e87dbed` on 2026-09-26. Thus fold 9 was observed before the later ablation design, but not during development of the final V3 architecture. The paper's present wording must be corrected.
- The model's intervention is an exact *frozen-ledger internal intervention*. It was never claimed to be a causal pixel intervention. This distinction should remain prominent.

### Concede and fix

- Exact replay belongs to the conserved ledger plus a deterministic decoder, not uniquely to the CTMC.
- The sequential-hazard control has the same exclusive ledger and replay contract. It must be defined explicitly.
- Forty mostly grade-0, single-cell certificates are not evidence of useful interpretation.
- The current fold-9 study does not establish training-run generality.
- The current release is insufficient for independent split and inference verification.
- The current ten-page manuscript is over the last verified CVPR eight-content-page limit.

## 3. Revised scientific claim

The recommended cross-sectional paper is about a **boundary-indexed conserved ordinal evidence ledger**, not about the matrix exponential being uniquely interpretable.

For local nonnegative entries \(\ell_{i,k}\), priors \(\pi_k\), and any declared deterministic monotone ordinal decoder \(D\),

\[
\lambda_k=\pi_k+\sum_i\ell_{i,k},\qquad
p=D(\lambda),\qquad
p^{(-S)}=D\!\left(\pi+\sum_{i\notin S}\ell_i\right).
\]

The structural contribution is the exclusive, cumulative, boundary-specific
ledger and its exact grouped replay. Its distinction from generic additive
explanation methods must be demonstrated at the ordinal boundaries: the audit
must identify which spatial evidence advances the prediction across each
threshold and must separate evidence for different thresholds. Exact additive
accounting by itself is not a sufficient novelty claim. The CTMC and sequential
hazard are two decoder instantiations. The CTMC may be retained as the primary
implementation, but the paper must not claim that normalization, monotone
deletion, exclusivity, or exact replay require it.

The only genuinely CTMC-specific static property is a shared-clock homogeneous Markov semigroup. With one endpoint label per image, the present data do not test whether that property is useful. A CTMC-centric paper would require longitudinal observations with real time intervals; that is a different project and should not be improvised for this deadline.

Recommended working title:

> **Every Threshold Has a Place: Conserved Spatial Evidence Ledgers for Auditable Ordinal Vision**

## 4. Workstream A: validation-wide exact certificate census

Use every locked EyePACS outer-fold image, not 40 sampled validation cases. Freeze the audit protocol before opening aggregate results.

### 4.1 Cases and strata

Report by:

- true grade and predicted grade;
- correct versus erroneous prediction;
- each ordinal boundary \(Y>k\);
- scale and theoretical receptive-field class;
- nonzero predicted grades separately from grade 0.

Grade 0 must be described honestly as absence of sufficient positive severity evidence; it does not have a positive lesion witness under this nonnegative architecture.

For the primary correctly predicted positive cohort, current OOF confusion
counts imply EyePACS sample sizes of 697/3,794/308/392 for grades 1/2/3/4 and
APTOS sizes of 186/918/14/207. The APTOS grade-3 stratum is therefore
descriptive rather than independently powered. Overcalls and undercalls are
audited separately; correct grade-0 images are a negative control, not silently
discarded.

### 4.2 Exact grouped interventions

For each relevant boundary, evaluate both:

- **necessity/deletion:** remove the ranked prefix and replay the same decoder;
- **sufficiency/retention:** retain only the ranked prefix plus the fixed prior and replay the same decoder.

Budgets must be expressed in all three ways:

1. fraction of valid ledger cells;
2. nominal addressable area, using stride squared as each cell's cost; and
3. fraction of boundary evidence mass.

The clipped union area of declared receptive-field footprints is a separate
locality disclosure, not the primary budget: overlapping fields and the
image-spanning $s16/s32$ theoretical supports make it a poor substitute for
addressable ledger area.

For every selected group, compare:

- a scale-stratified uniform random group with the same cell count;
- a scale- and footprint-area-matched random group;
- a boundary-rate-mass-matched random group;
- least-evidential cells; and
- the same ranking with spatial coordinates permuted.

Use at least 50 deterministic random repeats for the complete OOF audit and
larger permutation counts for the compact summary tests. Random selection and
matching tolerances must be fixed in the protocol. At fixed area, the selected
group should have more effect than random. At fixed full rate-vector mass, the
decoder is expected by construction to reproduce the same effect; that control
therefore tests how much area random evidence needs to collect the same mass,
not whether conservation can be violated.

### 4.3 Endpoints

- change in \(P(Y>k)\), expected grade, posterior total variation, and MAP grade;
- AOPC for deletion and retention curves;
- minimal ranked-prefix budget for a MAP change;
- minimal ranked-prefix budget for \(|\Delta E[Y]|\geq0.25\);
- posterior-preservation rate under retain-only evidence;
- top-versus-random effect ratio and paired patient-cluster interval;
- evidence concentration, effective support size, scale composition, and the
  addressable area needed to obtain 50/80/90% of local-attributable effect;
- false-locality analysis: effect size as a function of receptive-field footprint.

For target tail $T_k(p)=P(Y>k)$, normalize deletion and retention by the
image's locally attributable range:

\[
D_k(S)=\frac{T_k(p_{\rm full})-T_k(p_{\rm delete\ S})}
{T_k(p_{\rm full})-T_k(p_{\rm prior})},\qquad
R_k(S)=\frac{T_k(p_{\rm keep\ S})-T_k(p_{\rm prior})}
{T_k(p_{\rm full})-T_k(p_{\rm prior})}.
\]

Always accompany normalized values with raw effects and flag near-zero
denominators.

### 4.4 Gate A

The auditability claim advances only if grouped evidence has a practically nontrivial effect and consistently exceeds matched random controls. A single significant comparison is insufficient. A reasonable preregistered gate is:

- deletion- and retention-curve AUC advantages are each at least 0.10 in
  normalized local-attributable effect units, with patient-cluster 95%
  intervals excluding zero on EyePACS and image-bootstrap intervals excluding
  zero on APTOS;
- the advantage remains positive in at least three of four adequately powered
  EyePACS positive-grade strata and at least two of APTOS grades 1, 2, and 4;
- median area-to-50%-effect is at most 10% and at most half the random value,
  or the top 10% addressable area removes/recovers at least 50% of the
  local-attributable boundary effect; and
- retain-only prefixes preserve the original prediction materially better than
  count/area-matched random prefixes.

If Gate A fails, do not call the learned maps useful explanations. The paper may still describe exact accounting, but that is unlikely to clear CVPR.

### 4.5 Machine adjudication contract

Gate A is evaluated by the checksum-sealed OOF statistics job rather than by
manual selection of favorable plots.  The adjudicator consumes the complete
curve artifact as well as the image--boundary census and emits one status per
clause: `pass`, `fail`, or `insufficient_data`.  The auditability claim is
authorized only when every clause passes.  A missing dataset, curve artifact,
method pair, required grade stratum, source commit, protocol hash, or clean
runtime provenance forces `insufficient_data` and withholds the claim.

The primary matched controls for the gate are the scale/count/stride-area
control and the scale/receptive-field-area control.  Coordinate permutation,
least-evidential selection, and boundary-mass matching remain reported
falsification diagnostics, but the mass-matched control is not used to demand
an effect difference that conservation makes impossible.  Patient-cluster
resampling is used for EyePACS and image resampling for APTOS, always preserving
the same image--boundary pairing across selection methods.  The machine report
also includes correct/error, true-grade, predicted-grade, boundary, and
TP/FP/FN strata; linear interpolation over the frozen nominal-addressable-area
grid for area-to-effect and top-area endpoints; and theoretical receptive-field
footprint disclosure.  The latter is explicitly a false-locality diagnostic,
not a pixel-localization claim.

## 5. Workstream B: controlled ordinal-shortcut benchmark

This is the most decisive new experiment because the location, boundary, and pixel-level counterfactual are known exactly without clinical annotation.

### 5.1 Construction

Create four independently randomized visual codes, one for each target
\(z_k=\mathbf1[y>k]\). The main localized family places four nonoverlapping
retinal-support codes at positions drawn deterministically per image. Active
and inactive codewords must have equal area and matched first-order
colour/luminance statistics (for example, checker phase or orientation), so
brightness and patch count cannot solve the task. During training,

\[
P(m_k=1\mid z_k=1)=0.9,\qquad
P(m_k=1\mid z_k=0)=0.1.
\]

Jitter intensity and appearance within a frozen distribution. Preserve the
clean image so toggling a codeword is a paired pixel counterfactual.
Independent marker noise is essential: perfectly nested markers would make
boundary identity unidentifiable. Per-image locations prevent the model or
auditor from succeeding from a fixed coordinate prior.

Run three predeclared shortcut families rather than relying on an easy coloured
square:

1. localized lesion-like codes with unseen test positions and appearances;
2. border/acquisition-artifact codes; and
3. diffuse colour or illumination codes whose support cannot honestly be
   described as a lesion.

These families test complementary behavior. The localized family tests spatial
and boundary identification; the border family tests acquisition shortcuts;
and the diffuse family is a deliberate negative test of locality. For each
family include missing cues, inverted cues, mutually conflicting cues,
boundary-swapped cues, and unseen locations or appearances. A model that calls
the diffuse signal a focal lesion fails the audit even if it predicts well.

Evaluate:

- correlated-marker validation;
- balanced marker groups;
- reversed-correlation groups;
- clean images; and
- a factorial paired set in which each marker is toggled while all other pixels are fixed; and
- a cue-only positive-control arm in which image--label assignments are
  permuted but a perfect valid ordinal code carries the label.

Rank ledger entries by their native target-boundary contribution. Do not choose
entries using leave-one-out effects and then score the same entries by those
effects; that circular design would overstate audit quality.

### 5.2 Models

Use identical patient splits and training policy for:

1. ORIGIN ledger + CTMC;
2. the identical ledger + sequential hazards;
3. the strongest pooled continuation-ratio model;
4. a matched ordinal Additive-MIL baseline; and
5. Sparse Activations/BagNet using the authors' released design where feasible.

Grad-CAM may be included as a post-hoc reference, not as the only baseline.

### 5.3 Metrics

- marker localization AUPRC and pointing accuracy;
- boundary identification accuracy: does the \(k\)-th evidence field select the \(k\)-th marker?;
- the full boundary-response matrix after independently toggling each marker;
- diagonal selectivity versus off-diagonal leakage;
- internal grouped-deletion effect versus matched random deletion;
- correlation between exact internal effect and paired pixel-counterfactual effect;
- worst-group accuracy and clean/reversed-correlation degradation;
- failure examples where high test accuracy is driven by the shortcut.

### 5.4 Gate B

The ledger must show a capability beyond a generic class heatmap:

- reliably identify the correct shortcut location and its ordinal boundary;
- produce a substantially diagonal boundary-response matrix;
- show a large target-versus-random intervention effect; and
- expose shortcut reliance before the clean or reversed-correlation performance failure is inspected.

Predeclare numerical gates: the aligned-to-neutral/reversed performance gap
must establish that the model actually learned the cue; blind localization
must pass a family-wise permutation test for at least three boundaries in at
least two of three seeds; discovered-region normalized deletion must exceed
scale/area-matched random deletion by at least 0.20 and twofold in median; and
the Spearman correlation between internal deletion and paired pixel-toggle
effect should be at least 0.50 (95% lower bound above 0.30), or an equivalent
predeclared quartile separation. A clean-checkpoint negative control must show
no systematic code localization.

The sequential decoder is expected to share this capability because it shares the ledger. That result supports the revised ledger contribution and should not be hidden.

## 6. Workstream C: external semantic alignment

Use IDRiD only for a separate semantic question. Its public specification includes 81 lesion-segmentation images with masks for microaneurysms, hemorrhages, hard exudates, and soft exudates, plus grades.

Report:

- lesion-mask AUPRC, pointing game, and precision at fixed evidence mass;
- lesion-overlap deletion effect versus scale/area/mass-matched non-lesion regions;
- results by lesion type and grade;
- fine fields separately from coarse context fields; and
- a random-map and intrinsic-model baseline.

Do not turn a coarse cell into a pixel-segmentation claim. The \(s16/s32\) theoretical receptive fields exceed the input size and must be labelled contextual/global evidence. Semantic alignment is complementary to internal faithfulness, not proof of it.

## 7. Workstream D: matched generality and uncertainty

Do not repeat all ten exploratory ablations. Run the smallest decisive comparison set.

### Preferred design if compute is genuinely available

- five core models (ORIGIN-CTMC, same-ledger hazard, pooled continuation ratio,
  ordinal Additive MIL, and Sparse Activations/BagNet);
- all ten patient-grouped EyePACS outer folds;
- three independent training seeds;
- the same frozen outer/inner memberships for every model and seed.

This is 150 model-fold-seed cells, of which ten ORIGIN cells already exist. If
compute is constrained, first run one seed on all ten folds, then two additional
seeds on three prospectively selected folds. Those folds may represent
predeclared low/median/high difficulty using only the already frozen ORIGIN
results, but their identities and selection rule must be committed before any
new baseline is trained. Do not choose folds after inspecting a baseline.

Report accuracy, QWK, MAP-MAE, expected-grade MAE, balanced accuracy, macro-F1, and per-grade recall. For posterior quality report NLL, RPS, Brier score, classwise ECE, and four threshold reliability diagrams/ECEs. If calibration is applied, fit it only on each inner-validation split and apply it unchanged to that fold's outer test set.

Use patient-cluster paired intervals for test-cohort uncertainty and report training-seed spread separately. Do not convert bootstrap tail fractions into p-values or claim superiority from heterogeneous literature protocols.

## 8. Workstream E: protocol, numerical, and release audit

Before submission, release within the anonymous supplementary package:

- the exact freeze chronology and commit hashes;
- anonymized split membership or deterministic split files and hashes;
- selected checkpoint tensors and SHA-256 hashes where size permits;
- every outer posterior, per-image label, and patient-cluster identifier;
- machine-readable grouped intervention certificates;
- exact definitions of both CTMC and sequential-hazard decoders;
- executable metric/bootstrap reproduction;
- failure/restart provenance; and
- a manifest verifying all files.

Validate the FP64 matrix exponential against an independent SciPy or high-precision reference over random and adversarial rate vectors in \([0,64]^4\). Test normalization, nonnegativity, absolute/relative posterior error, and finite-difference gradients. State the Taylor degree and scaling rule explicitly.

### 8.1 Fail-closed top-level package

`tools/assemble_origin_acceptance_package.py` is the final release barrier. It
validates, in place, the numerical decoder audit, decoder-contract audit,
checkpoint/split manifest, IDRiD image-cluster statistics, OOF Gate A,
controlled-shortcut Gate B, and the matched multi-fold/multi-seed outer
release. It verifies canonical and file digests, nested artifact digests,
registered protocols, complete fold/seed/model censuses, clean Git provenance,
and the privacy contract before writing anything. Missing, incomplete, dirty,
or checksum-inconsistent inputs prevent both outputs.

The exported manifest contains only schemas, digests, commit identifiers,
aggregate censuses, and gate statuses. It never embeds licensed pixels,
checkpoints, predictions, filesystem locations, or raw image/patient
identifiers. A scientifically failed Gate A or B is retained as a negative
result: the reproducibility package is complete, while
`scientific_claims_authorized` is false. The optional submission-readiness mode
requires both gates to pass. The Slurm launcher lists all seven upstream job
IDs in one `afterok` dependency and executes the assembler from a clean,
detached commit snapshot.

## 9. Architecture decision gate

Do **not** introduce another trainable architecture before Gates A and B. The review identifies missing utility evidence, not insufficient module count.

If Gates A and B pass:

- keep V3 weights and add a deterministic inference-time **Boundary Certificate Compiler**;
- export ranked necessity and sufficiency prefixes with exact replay;
- reframe the architecture around the conserved ordinal ledger;
- keep CTMC and hazard as declared decoder instantiations.

If Gate A or B fails:

- stop claiming useful spatial explanations from V3;
- do not rescue the claim with more heatmaps;
- only then train a locality/concentration-constrained successor, using a predeclared certificate loss and the shortcut benchmark as an untouched evaluation gate;
- repeat the complete matched comparison for that successor.

The first fallback should be an honest two-ledger design: fine local evidence with a bounded receptive field, plus explicitly labelled global/context evidence. Global evidence must never be drawn as a lesion-localization map. This is preferable to pretending coarse cells are local.

### 9.1 Concrete fallback if either utility gate fails: ESCROW

The proposed successor is an **Evidence-Sealed Context-Restricted Ordinal
Witness network (ESCROW)**. It is deliberately a fallback: training it before
auditing V3 would discard ten-fold evidence without first testing the capability
that V3 was designed to provide.

Partition the image into nonoverlapping $64\times64$ patches and encode every
patch independently through the full depth of a shared encoder,

\[
z_i=f_\theta(x_i),\qquad i=1,\ldots,P.
\]

There is no attention, padding, or feature exchange across patches, so every
local feature has a hard input support. A shared-weight encoder also processes
one antialiased low-resolution global view $z_G=f_\theta(D(x))$. Each local
patch emits a bounded severity-atom simplex; cumulative compilation gives

\[
a_{i,m}=Aq_{i,m},\qquad e_{i,k}=\sum_{m>k}a_{i,m},\qquad
E_k=\frac{\gamma_k}{P_{\rm valid}}\sum_{i\in V}e_{i,k}.
\]

The global branch is restricted to a discount

\[
c_k=c_{\min}+(1-c_{\min})\sigma(w_k^\top z_G+d_k),\qquad
\eta_k=-\operatorname{softplus}(\beta_k)+c_kE_k.
\]

It cannot emit positive severity evidence. Continuation probabilities
$h_k=\sigma(\eta_k)$ define the normalized posterior

\[
p_y=(1-h_y)\prod_{j<y}h_j\quad (y<K-1),\qquad
p_{K-1}=\prod_{j=0}^{K-2}h_j.
\]

For any patch set $S$, deletion and retention are exact deployed-model
computations:

\[
\eta_k^{(-S)}=\eta_k-c_k\frac{\gamma_k}{P_{\rm valid}}
\sum_{i\in S}e_{i,k},\qquad
\eta_k^{(S)}=-\operatorname{softplus}(\beta_k)+
c_k\frac{\gamma_k}{P_{\rm valid}}\sum_{i\in S}e_{i,k}.
\]

This design keeps the lesson of the completed ablation: coarse/global context
is important for accuracy, but it must be declared as context rather than drawn
as focal lesion evidence. Its scoped candidate novelty is the combination of
independently encoded bounded-support cumulative witnesses, exclusive positive
severity advancement, a context branch that can only discount those witnesses,
and exact patch-set replay. That claim still requires a final prior-art audit
and matched comparison against Additive MIL, BagNet/Sparse Activations,
FocusMIL, and ordinal MIL.

Train with categorical NLL plus RPS, a boundary-specific ranking loss between
at-risk negatives ($y=k$) and positives ($y>k$), and a small austerity penalty
that discourages the context gate from carrying the task through suppression.
Before full CV, require a three-seed APTOS gate against V3, local-only, and
unrestricted global--local fusion. Reject ESCROW if the context gate plus
shuffled/constant local evidence preserves most of the gain, if top local
patches do not beat area-matched random patches, or if the controlled ordinal
shortcut cannot be assigned to the correct boundary. Only a passing model may
advance to three predeclared EyePACS folds and then full CV.

## 10. Paper rewrite

The main paper should contain only:

1. the ledger contract and exact grouped-replay theorem;
2. one compact architecture figure;
3. complete EyePACS/APTOS grading results without split-incomparable SOTA language;
4. the four-model matched comparison;
5. one controlled ordinal-shortcut figure;
6. one grade-stratified necessity/sufficiency result; and
7. an explicit limitations paragraph.

Move the ten-arm exploratory table, numerical audits, full calibration plots, provenance, and extra certificates to the supplement. Keep the main body at eight pages until the CVPR 2027 author kit states otherwise.

Delete or rewrite the current claim that the trio of normalized ordinal posterior, exclusive spatial evidence, and exact replay is unique to the CTMC. The repository's own sequential-hazard control already satisfies that trio.

## 11. Schedule and stop/go decisions

### 2026-10-01 to 2026-10-05

- commit this protocol;
- implement grouped delete/retain audits and matched random controls;
- implement the shortcut generator and its unit tests;
- implement the independent decoder numerical audit;
- correct the chronology and decoder claims in a revision branch.

### 2026-10-06 to 2026-10-10

- run one-fold, three-seed pilots for Workstreams A and B;
- run the full existing-checkpoint certificate census;
- decide Gates A and B without changing their thresholds.

### 2026-10-11 to 2026-10-27

- run the frozen multi-fold/multi-seed core matrix;
- run IDRiD semantic evaluation;
- produce all OOF posteriors and certificates.

### 2026-10-28 to 2026-11-06

- aggregate and independently reproduce every result;
- write the eight-page paper and supplement;
- conduct an adversarial internal review against W1--W4.

### 2026-11-07 to 2026-11-16

- freeze figures/tables;
- complete OpenReview registration before 2026-11-10 AOE;
- perform anonymity, page-count, artifact, and citation audits;
- submit by 2026-11-16 AOE.

## 12. Acceptance criterion

The revised submission has a credible CVPR case only if it demonstrates all of the following:

1. competitive grading under matched, replicated evaluation;
2. a concrete ordinal audit that finds a known shortcut or meaningful failure;
3. grouped evidence that is more necessary and sufficient than carefully matched random evidence;
4. an honest comparison showing which guarantees are ledger-wide and which are decoder-specific; and
5. independently replayable artifacts.

If these do not hold, the correct action is not to inflate the claims. A fundus-focused paper with the exact ledger, complete CV, IDRiD validation, and narrower significance may still be strong for MICCAI, but it would not yet answer the present CVPR review.

## Primary references informing the plan

- Javed et al., *Additive MIL*, NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/82764461a05e933cc2fd9d312e107d12-Abstract-Conference.html
- Donteu et al., *Sparse Activations for Interpretable Disease Grading*, MIDL 2024: https://proceedings.mlr.press/v227/donteu24a.html
- Wang et al., *Image Classification with Consistent Supporting Evidence*, ML4H 2021: https://proceedings.mlr.press/v158/wang21a.html
- Dasgupta et al., *Framework for Evaluating Faithfulness of Local Explanations*, ICML 2022: https://proceedings.mlr.press/v162/dasgupta22a.html
- Boland et al., *There Are No Shortcuts to Anywhere Worth Going*, MIDL 2024: https://proceedings.mlr.press/v250/boland24a.html
- Wu et al., *On the Faithfulness of Vision Transformer Explanations*, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Wu_On_the_Faithfulness_of_Vision_Transformer_Explanations_CVPR_2024_paper.html
- IDRiD data specification: https://idrid.grand-challenge.org/Data/
- CVPR 2027 call: https://cvpr.thecvf.com/Conferences/2027/CallForPapers
