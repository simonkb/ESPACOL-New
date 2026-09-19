# PATHS-V8: Ordinal-Shell Warranted Transport (OSWT)

## Registered question

PATHS-V8 tests whether a frozen, audited ORIGIN-V3 grader can be improved by a
small structural refiner whose computation is also its explanation. The
refiner is trained only from image-level ordinal labels. It has no classifier
or pooled-feature bypass, and development uses only the inner validation split.

The experiment does **not** assume that it will improve accuracy. A prospective
APTOS gate compares it with both the static V3 source and a matched ungated
control before any EyePACS run is allowed.

## Frozen source model and ordinal shells

Every run loads one fold-specific ORIGIN-V3 checkpoint by path and SHA-256 and
freezes its encoder and generator. For the nested local cumulative atoms

\[
1\ge u_0(x)\ge u_1(x)\ge\cdots\ge u_{K-2}(x)\ge0,
\]

OSWT forms the exact exclusive grade simplex

\[
v_0=1-u_0,\qquad
v_j=u_{j-1}-u_j,\qquad
v_{K-1}=u_{K-2}.
\]

Invalid receptive-field cells receive zero mass. Valid cells satisfy
\(v_j\ge0\) and \(\sum_jv_j=1\). The stored shells are never reconstructed
after an intervention, so deleting a cell cannot create artificial Grade-0
mass through \(1-u_0\).

## Complementary ordinal-boundary ledgers

For every scale \(s\), cell \(x\), and boundary \(k\), OSWT partitions the
shell simplex **before** applying a nonlinear spatial functional:

\[
L_{s,k}(x)=\sum_{g\le k}v_{s,g}(x),\qquad
R_{s,k}(x)=\sum_{g>k}v_{s,g}(x).
\]

Thus \(L_{s,k}(x)+R_{s,k}(x)=1\) at every valid cell. Residual cumulative-atom
capacity is deliberately treated as low-side/null internal evidence; this is a
modeling assumption, not clinically validated normal anatomy or a lesion label.

For either side \(a\in\{L,R\}\), let
\(M^a_{s,k}=\sum_x a_{s,k}(x)\) and
\(r^a_{s,k}(x)=a_{s,k}(x)/M^a_{s,k}\), with zero used when the mass is zero.
At fixed probes \(z_\ell\in(0,1)\), OSWT uses

\[
\phi_z(r)=
\frac{-\log(1-(1-z)r)-(1-z)r}{-\log z-(1-z)}.
\]

One learned probe simplex \(\pi\) is shared by every grade, side, boundary,
and scale. The local stored warrant is

\[
c^a_{s,k}(x)=M^a_{s,k}\sum_\ell\pi_\ell
\phi_{z_\ell}(r^a_{s,k}(x)).
\]

Multiplication by original partition mass, rather than mean cell mass, removes
the spurious inverse-lattice-size shrinkage. Since
\(\phi_z(r)\le r^2\), each aggregate side warrant remains in \([0,1]\).
The original denominator and mass are frozen under deletion; remaining cells
are never renormalized into stronger evidence.

The native V3 boundary-scale simplex \(w_{s,k}\) is used on **both** sides of
the same cut:

\[
L_k=\sum_{s,x}w_{s,k}c^L_{s,k}(x),\qquad
R_k=\sum_{s,x}w_{s,k}c^R_{s,k}(x).
\]

Using one probe functional and one scale distribution per cut prevents a
hidden side-specific calibration route.

## Coverage-aware ordinal transport

The spatial direction at boundary \(k\) is

\[
D_k=\frac{R_k-L_k}{R_k+L_k+\tau_k}.
\]

Let \(q_k=P_{V3}(Y>k)\). The treatment uses the cut-aligned uncertainty gate

\[
G_k=4q_k(1-q_k),
\]

while the matched control uses \(G_k=1\). Parameters are explicitly bounded:

\[
0<\beta_k<\beta_{\max},\qquad
\tau_{\min}<\tau_k<\tau_{\max}.
\]

The only learned posterior update is

\[
T_k=\rho\,\beta_kG_kD_k,\qquad
\log\frac{p'_{k+1}}{p'_k}=
\log\frac{p_{k+1}}{p_k}+T_k.
\]

Prefix sums of \(T_k\), followed by one normalization, yield the unique
categorical posterior \(p'\). This posterior is factorized into continuation
probabilities for the proper risk-set NLL plus RPS objective and is the sole
source of predictions and metrics.

## Exact computation certificates

The signed probability mass crossing each ordinal cut is

\[
F_k=\sum_{j=0}^{k}(p_j-p'_j),
\]

and \(\sum_k|F_k|\) is the exact discrete ordinal Wasserstein-1 cost. A cell
intervention jointly removes its stored ORIGIN rate entries and its stored
left/right warrant entries, then reruns both structural decoders. Forward and
intervention audits verify posterior normalization, continuation
factorization, complementary partition conservation, shared scale/probe
mixtures, log-odds replay, signed-flow replay, and additive ledger deletion.

Certificate witnesses are selected by exact intervention within a
pre-specified deterministic shortlist: candidates are shortlisted from the
stored local ledger, each is replayed, and the cell causing the largest
predicted-grade margin drop is reported. The certificate records whether that
best effect is positive; promotion requires a positive effect for every
registered witness. The resulting claim is "exact best within the registered
shortlist," not global minimality.

These interventions act on encoder receptive-field ledger cells. They are not
pixel-level causal interventions, named lesion detections, or clinical proof.
The current functional measures amount and focal concentration; it does not
encode within-scale topology or anatomical relationships.

## Registered variants and training contract

1. `shell_warranted`: complementary ledgers with \(4q(1-q)\) gating.
2. `shell_warranted_ungated`: identical model and training with \(G_k=1\).
3. `identity_v3`: \(\rho=0\), returning the exact V3 output object.

Only the shared probe logits and bounded boundary parameters are optimized.
Their AdamW group uses zero weight decay because decay in raw sigmoid
coordinates would impose an unintended prior. Every invocation accepts one
fold only and is bound to that fold's checkpoint hash. Source-state tensor
hashes, implementation bytes, split signatures, checkpoint hashes, and JSON
content checksums are carried into the promotion audit.

## Prospective APTOS gate

APTOS fold 0 uses 293 inner-validation images; its outer test fold remains
locked. EyePACS fold 0 is submitted only if the risk-gated model:

- reaches at least 254/293 correct, QWK at least 0.9265, and MAE at most 0.1741;
- strictly beats both static V3 and the matched ungated control;
- does not regress in balanced accuracy or macro-F1 relative to V3;
- corrects at least one true Grade-3 case from each adjacent side, with the
  corresponding signed boundary flow, while harming no source-correct Grade-3
  case;
- reproduces the source model exactly and passes every forward, replay,
  provenance, and checksum identity, including reconstruction of the shell
  simplex and both complementary partitions from the stored cumulative atoms;
- yields a positive exact predicted-grade margin drop for every
  grade-stratified certificate sample.

Failure of any registered condition blocks EyePACS. Thresholds are not changed
after results are observed. Full cross-validation and locked-test evaluation
remain out of scope until the fold-0 mechanism gate passes.

Run the graph with `scripts/launch_paths_oswt_v8_aptos_f0.sh` after exporting
the audited APTOS and EyePACS V3 SHA-256 values.
