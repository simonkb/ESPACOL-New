# MOSAIC: minimum ordinal proofs on a retinal micro-region lattice

Status: **Core implementation complete on `mosaic-ordinal-proof`; independent
APTOS and EyePACS fold-0 decoder audits complete; raw posterior median locked
prospectively for new folds**

Date: 2026-09-03

Data constraint: **EyePACS/Kaggle DR and APTOS image-level grades only**. No
lesion masks, text encoder, concept labels, manual annotation, or clinician
study are required for development.

Working name: **MOSAIC** -- **M**inimum **O**rdinal **S**ufficient
**A**ttribution by **I**ntervention and **C**ounting.

## 1. Executive decision

Do not make the existing 300-by-300 tiles smaller and call the resulting
attention map an explanation. Replace the entire global classifier with a
fine-grid **ordinal proof circuit**:

1. a spatially bounded encoder emits cumulative severity-witness probabilities on a
   fine retinal lattice;
2. a monotone cardinality law computes every adjacent ordinal transition from
   those regional probabilities;
3. a deterministic projection finds the smallest retained micro-region set
   that preserves the transition and whose removal sufficiently weakens it;
4. only that projected proof is allowed to produce the final continuation
   probability and grade; and
5. setting a witness probability to zero is an exact internal intervention, so the
   reported effect is the actual change in the model's computation.

This directly addresses the weakness in OPTIC-C: GPA indicates that one
300-by-300 tile received more weight than another, but it does not establish
which small region was sufficient for a boundary, whether hiding that evidence
changes the decision, or whether the displayed region is merely correlated
with a separate global path.

MOSAIC is a research hypothesis, not a guarantee of 86% accuracy or venue
acceptance. The plan is designed to reject it quickly on APTOS before any full
EyePACS run.

## 2. Why the granularity must change

Microaneurysms are commonly reported as roughly 15--60 micrometres and seldom
larger than 125 micrometres. In a resized colour photograph they can occupy only
a small number of pixels. A 300-by-300 tile can therefore overlap a lesion while
remaining far too broad to explain it. Coarse Pointing Game scores are also easy
to inflate because a large tile has a high chance of touching at least one mask
pixel.

The new input is one tightly cropped fundus field resized directly to a
canonical 896-by-896 square rather than nine independently encoded
300-by-300 crops plus a global crop. A fully convolutional pass avoids tile
seams and removes the redundant global-tile evaluation. The geometric warp and
its fixed support mask are identical for every image, so the proof circuit
cannot infer a grade from image-specific padding or mask shape.

Two spatial resolutions are distinguished carefully:

- **lattice stride**: distance between neighbouring evidence centres;
- **receptive field**: image area that can influence one evidence value.

The primary engineering target is a 112-by-112 evidence lattice (nominal stride
8 pixels). The theoretical and empirical receptive field must be reported; it
will be larger than 8-by-8. An upsampled lattice is a computational
micro-region map, **not** a pixel segmentation mask. With grade labels alone,
the project can validate computational faithfulness but cannot validate that
every activated pixel is a named retinal lesion.

## 3. Prior-art boundary through 30 August 2026

The following ideas already exist and must not be claimed independently:

- grade-label-only lesion discovery for DR: LAT, CVPR 2021;
- fine sparse BagNet class-evidence maps for DR: Sparse Activations for
  Interpretable Disease Grading, MIDL 2024;
- nominal class maps as the sole DR prediction path: SoftCAM, MIDL 2026;
- intrinsic learned masks: AIM, ICCV 2025;
- learned concise sufficient subsets: SST, ICLR 2025;
- threshold-specific ordinal MIL attention: SATOMIL, WACV 2025;
- exact additive patch attribution: Additive MIL, NeurIPS 2022;
- learned visual cardinality potentials: CVPR 2015; and
- weak-label latent local ordinal states with high-order cardinality:
  MI-DORF, 2016;
- conditional continuation-ratio ordinal prediction: CORN; and
- top-ranked patch prefixes explaining a target fraction of a class logit:
  MorphoXAI, npj Digital Medicine 2026.

The defensible gap is their exact coupling:

> To our knowledge, MOSAIC is the first image-level-grade-supervised retinal
> grader in which every continuation-ratio transition is computed solely by a
> symmetric monotone regional cardinality circuit and replayed from a
> deterministic, tolerance-conditioned minimum-cardinality certificate that
> preserves the dense transition and suppresses its complement.

This is a scoped `to our knowledge` claim, not proof of absolute novelty. Do
not claim first local-evidence DR model, first intrinsic explanation, first
ordinal MIL model, first counting model, first sufficient rationale, or first
masking method.

Primary comparison anchors:

- LAT: <https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Lesion-Aware_Transformers_for_Diabetic_Retinopathy_Grading_CVPR_2021_paper.html>
- Sparse Activations: <https://proceedings.mlr.press/v227/donteu24a.html>
- SST: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/b1d4973f5d21708abef3cd6f17d842c8-Abstract-Conference.html>
- SATOMIL: <https://openaccess.thecvf.com/content/WACV2025/html/Shiku_Ordinal_Multiple-Instance_Learning_for_Ulcerative_Colitis_Severity_Estimation_with_Selective_WACV_2025_paper.html>
- Additive MIL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/82764461a05e933cc2fd9d312e107d12-Abstract-Conference.html>
- cardinality MIL: <https://openaccess.thecvf.com/content_cvpr_2015/html/Hajimirsadeghi_Visual_Recognition_by_2015_CVPR_paper.html>
- MI-DORF: <https://arxiv.org/abs/1609.01465>
- CORN: <https://doi.org/10.1007/s10044-023-01181-9>
- SIS: <https://proceedings.mlr.press/v89/carter19a.html>
- sufficiency/necessity formalization: <https://proceedings.mlr.press/v280/bharti25a.html>
- NEMt: <https://proceedings.mlr.press/v265/moller25a.html>
- selector--predictor leakage audit: <https://proceedings.mlr.press/v130/jethani21a.html>
- SoftCAM: <https://proceedings.mlr.press/v315/djoumessi26a.html>
- AIM: <https://openaccess.thecvf.com/content/ICCV2025/html/Alshami_AIM_Amending_Inherent_Interpretability_via_Self-Supervised_Masking_ICCV_2025_paper.html>
- MorphoXAI: <https://www.nature.com/articles/s41746-026-02741-z>

## 4. Exact forward pass

### 4.1 Spatially bounded evidence encoder

Let the canonicalized fundus be

\[
X_n\in\mathbb R^{3\times896\times896}.
\]

Use the early and middle blocks of ImageNet-pretrained EfficientNetV2S as a
fully convolutional local encoder. The first implementation taps the stride-8
map and uses only pointwise channel mixing after that tap:

\[
H_n=f_{\theta}^{\mathrm{local}}(X_n)
\in\mathbb R^{d\times112\times112}.
\]

The code must compute the theoretical receptive field of every candidate tap.
No self-attention, global average pooling, global token, or later spatial
mixing is permitted before witness inference. Otherwise one nominally local
cell can encode evidence from elsewhere and the certificate becomes spatially
misleading.

Three encoder variants are screened:

1. **RF-small**: stride-4 early map plus pointwise residual MLP;
2. **RF-medium**: stride-8 map plus pointwise residual MLP (default); and
3. **RF-large ceiling**: a later semantic map, included only to quantify the
   accuracy--locality trade-off.

The primary paper model is the smallest receptive field that remains
competitive. Lattice stride is never reported as receptive-field size.

Flatten the valid retinal cells into

\[
h_{n,i}\in\mathbb R^d,\qquad i=1,\ldots,P.
\]

Cells outside a fixed centered elliptical support are excluded rather than
learned as normal evidence. The mask is a compile-time function of canonical
image dimensions only; source pixels, camera field shape, and the grade never
choose valid cells. At 896 pixels and stride 8 it retains exactly 9,864 of
12,544 lattice cells for every image. It is geometry, not an annotation or an
image-derived segmentation.

### 4.2 Nested local ordinal state

A shared pointwise head emits one categorical local severity state:

\[
a_{n,i}=g_\phi(h_{n,i})\in\mathbb R^K,
\qquad
\rho_{n,i,m}=\operatorname{softmax}_m(a_{n,i}),
\quad m=0,\ldots,K-1.
\]

For boundary \(k\), define the regional witness probability

\[
\lambda_{n,i,k}
=P(L_{n,i}>k)
=\sum_{m=k+1}^{K-1}\rho_{n,i,m},
\qquad k=0,\ldots,K-2.
\]

Consequently,

\[
1\ge\lambda_{n,i,0}\ge\lambda_{n,i,1}\ge\cdots
\ge\lambda_{n,i,K-2}\ge0.
\]

This encodes cumulative **decision evidence**: a local pattern supporting a
later boundary must support all earlier boundaries. It does not claim that the
clinical lesion inventory of grade \(k+1\) literally contains every lesion of
grade \(k\).

Calibrate the local-state bias quantitatively toward state 0. With \(P\) valid
cells, choose an initial expected abnormal count \(\mu_0\in[0.1,1]\) so

\[
P\,P(L_i>0)\approx\mu_0.
\]

For four equal abnormal-state logits, the normal-state logit advantage is
approximately \(\log(4P/\mu_0)\), roughly 10--13 at this lattice size. A merely
“strong” conventional bias would let thousands of tiny background
probabilities saturate the count distribution. Grade-0 images then provide
abundant negative-bag supervision that drives abnormal probabilities down
without pixel labels.

### 4.3 Exact truncated Poisson--binomial cardinality law

For boundary \(k\), treat every valid cell as a Bernoulli regional witness:

\[
Z_{n,i,k}\sim\operatorname{Bernoulli}(\lambda_{n,i,k}),
\qquad
C_{n,k}(S)=\sum_{i\in S}Z_{n,i,k}.
\]

The model counts regional evidence events, not manually annotated lesions. The
distinction matters: one certain witness and ten weak witnesses with the same
summed probability have different Poisson--binomial count distributions.

Only thresholds \(1,\ldots,R\) are learnable, so retain exact probability
masses \(0,\ldots,R-1\) plus an overflow bucket \(R\equiv C\ge R\). Initialise

\[
D^{(0)}_0=1,\qquad D^{(0)}_r=0\quad(r=1,\ldots,R).
\]

After witness probability \(e=\lambda_{n,i,k}\), update

\[
D'_0=(1-e)D_0,
\]

\[
D'_r=(1-e)D_r+eD_{r-1},
\qquad r=1,\ldots,R-1,
\]

\[
D'_R=D_R+eD_{R-1}.
\]

This recurrence is exact for every tail up to \(R\); no probability mass is
discarded. Its arithmetic cost is \(O(PR)\), not \(O(P^2)\).

For a required count \(r\),

\[
T_r(\boldsymbol\lambda)=P(C\ge r)
=1-\sum_{j=0}^{r-1}D_j.
\]

The implementation also evaluates the complementary lower tail directly,

\[
U_r(\boldsymbol\lambda)=P(C<r)=\sum_{j=0}^{r-1}D_j,
\]

rather than forming it by subtracting a saturated continuation score from one.
For large lattices, the low tail can be far below the FP32 normal range. The
implemented recurrence therefore also carries

\[
\ell_{<R}=\log P(C<R),\qquad
\eta_j=\log P(C=j\mid C<R),\quad j=0,\ldots,R-1,
\]

and forms every stop mixture with `logsumexp` and the exact
\(\log\operatorname{softmax}(\gamma_k)\). Log witness and non-witness values
are derived directly from the local categorical `log_softmax`, rather than by
taking a logarithm after probability-space rounding. Impossible events use
true log-zero with backward-safe masked reductions. Training consumes these
log-stop values directly. This scaled lower-tail representation preserves
finite, nonzero recovery gradients for diffuse or saturated 112-by-112
evidence fields where an ordinary probability-space stop would underflow.
Thus \(T_r+U_r=1\) in exact arithmetic, while the implementation retains a
stable log representation of the small side.

Every boundary learns a global distribution over focal versus distributed
requirements:

\[
\alpha_{k,r}=\operatorname{softmax}_r(\gamma_k),
\qquad r=1,\ldots,R,
\]

\[
F_k(\boldsymbol\lambda)=
\sum_{r=1}^{R}\alpha_{k,r}T_r(\boldsymbol\lambda).
\]

Use \(R=32\) in the first screen and ablate \(R\in\{16,32,64\}\). The
response is a valid number in \([0,1]\), is symmetric in the regional
witnesses, and is coordinatewise nondecreasing because

\[
\frac{\partial F_k}{\partial\lambda_{i,k}}
=\sum_{r=1}^{R}\alpha_{k,r}P(C_{-i,k}=r-1)\ge0.
\]

Interpretation is boundary-specific:

- \(\alpha_{k,1}\) dominant: one focal witness can pass the boundary;
- mass at larger \(r\): multiple regional witness cells are required; and
- broad mass: the training labels do not identify one sharp extent rule.

The \(\alpha_k\) parameters are dataset-level and never image-conditioned. An
image-conditioned rule could become a hidden global classifier and would void
the proof theorem. \(\alpha_k\) is a latent model cardinality preference, not a
recovered clinical lesion count: witness calibration and \(\alpha_k\) can
partially compensate for one another. Interpret it only after reporting its
seed/fold stability. Adjacent cells influenced by one lesion may also count as
multiple witnesses.

A literal Python loop over all \(P\) cells would create excessive GPU kernel
launch overhead. Implement an exact block-tree recurrence:

1. partition the flattened lattice into blocks of 32 or 64 witnesses;
2. compute every block's truncated distribution in parallel;
3. merge block polynomials in a balanced tree, retaining bins \(0{:}R-1\) and
   accumulating overflow from non-negative high-count crossing terms; and
4. use block-prefix/block-suffix scans plus one within-block refinement to find
   the exact proof boundary.

The local recurrence costs \(O(PR)\); direct truncated block merging adds
approximately \(O((P/B)R^2)\) work. The balanced implementation reduces the
sequential launch depth from \(P\) to roughly \(B+\log(P/B)\) for dense
scoring. Prefix/suffix certificate search adds a scan over the \(P/B\) blocks
and one refinement of at most \(B\) cells. For that boundary block, precompute
forward and reverse within-block distributions and merge them with the outer
prefix/suffix distributions; do not divide or deconvolve a probability
polynomial. Run all count recurrences in FP32 even when the encoder uses AMP.
Clamp Bernoulli inputs to the probability interval, remove negative round-off,
and renormalize each update/merge onto the probability simplex. Compute
continuation and stop mixtures separately, retain the scaled log lower tail,
and normalize the probability pair. The truncated law is exact in real
arithmetic and stabilized in both simplex and log-tail representations;
replay and intervention equalities are asserted to a serialized numerical
tolerance rather than as bitwise identities across CPU and CUDA reductions.

### 4.4 Deterministic dual proof projection

For boundary \(k\), sort valid witness probabilities

\[
\lambda_{(1),k}\ge\lambda_{(2),k}\ge\cdots\ge\lambda_{(P),k}.
\]

Include \(m=0\). Let \(S_m\) be the top-\(m\) prefix and define retained and
complementary witness vectors

\[
\boldsymbol\lambda^+_{k,m}
=\{\lambda_{(1),k},\ldots,\lambda_{(m),k}\},
\qquad
\boldsymbol\lambda^-_{k,m}
=\{\lambda_{(m+1),k},\ldots,\lambda_{(P),k}\}.
\]

The dense reference transition is

\[
\widetilde c_k=F_k(\boldsymbol\lambda^+_{k,P}).
\]

Select the smallest prefix satisfying both retained-evidence and
removed-evidence conditions:

\[
m_k^*=\min\left\{m:
F_k(\boldsymbol\lambda^+_{k,m})\ge\widetilde c_k-\varepsilon_s,
\quad
\widetilde c_k-F_k(\boldsymbol\lambda^-_{k,m})
\ge\rho_n\max(\widetilde c_k-\varepsilon_s,0)
\right\}.
\]

Default screen values are \(\varepsilon_s=0.02\) and \(\rho_n=0.5\), followed
by a preregistered sensitivity analysis. The second inequality is called
**\(\rho_n\)-collective score necessity** or **complement suppression**, not
unqualified causal necessity. Its right-hand side becomes zero for a negligible
dense transition, allowing the empty proof. A feasible prefix always exists at
\(m=P\).

The transition used by the classifier is **not** the dense score:

\[
c_{n,k}=F_k(\boldsymbol\lambda^+_{k,m_k^*}).
\]

Thus the returned proof is the forward path. There is no pooled-feature head,
CORAL bypass, residual visual logit, text branch, or auxiliary classifier at
inference.

Top-prefix optimality is exact for the observed witness ledger. Among all
size-\(m\) subsets, the top prefix coordinatewise dominates every other sorted
retained vector, while its complement is coordinatewise dominated by every
other sorted complement. Since \(F_k\) is symmetric and coordinatewise
nondecreasing, if any size-\(m\) subset satisfies both constraints, the top
prefix does. Therefore \(m_k^*\) is the tolerance-conditioned, globally
minimum cardinality for this model-relative dual certificate.

The prefix is canonical, not necessarily the unique explanation. Tied probabilities
and equally valid substitute proofs must be reported. The full dense ledger is
part of the computation because it determines \(\widetilde c_k\), the selected
indices, and the proof size; it must be serialized so certificate replay
reproduces selection as well as the final score.

### 4.5 Exclusive continuation cascade

Interpret \(c_{n,k}\) as the conditional probability of advancing from rung
\(k\) to \(k+1\), and compute its stop partner directly as

\[
s_{n,k}=\sum_{r=1}^{R}\alpha_{k,r}U_r(\boldsymbol\lambda^+_{n,k,m_k^*}).
\]

The implementation normalizes the non-negative pair \((c_{n,k},s_{n,k})\);
in exact arithmetic \(s_{n,k}=1-c_{n,k}\). Then

\[
q_{n,k}=P(Y_n>k)=\prod_{j=0}^{k}c_{n,j}.
\]

The class probabilities are

\[
p_{n,0}=s_{n,0},
\]

\[
p_{n,y}=\left(\prod_{j=0}^{y-1}c_{n,j}\right)s_{n,y},
\qquad1\le y<K-1,
\]

\[
p_{n,K-1}=\prod_{j=0}^{K-2}c_{n,j}.
\]

This guarantees ordered cumulative probabilities and a normalized class
distribution without four independent CORAL logits. The prospectively locked
point prediction is the raw posterior median

\[
\hat y_n=\sum_{k=0}^{K-2}
\mathbf 1\!\left[q_{n,k}\ge 0.5\right].
\]

Because the cumulative probabilities are ordered, this is a valid ordinal
upper median (the `>=` convention selects the upper grade on an exact 0.5 tie)
and minimizes expected absolute grade error under the model-implied raw
cascade law. This statement does not claim that the cost-sensitive scores are
calibrated natural posteriors. The rule adds no parameter and receives no
input outside the selected proof.

There is one necessary qualification when the continuation likelihood uses
boundary outcome weights. Its population optimum is the cost-sensitive score

\[
c^*_{n,k}=\frac{w_{k,1}\pi_{n,k}}
{w_{k,1}\pi_{n,k}+w_{k,0}(1-\pi_{n,k})},
\]

not the natural continuation posterior \(\pi_{n,k}\). The decoder audit
therefore evaluates the exact, parameter-free inverse

\[
\bar c_{n,k}=\frac{w_{k,0}c_{n,k}}
{w_{k,0}c_{n,k}+w_{k,1}s_{n,k}},\qquad
\bar s_{n,k}=\frac{w_{k,1}s_{n,k}}
{w_{k,0}c_{n,k}+w_{k,1}s_{n,k}}.
\]

This analytic inverse is retained as a fixed diagnostic, not used by the
operating model. On APTOS, raw posterior median changed accuracy/MAE/QWK from
82.25/0.2150/0.9030 to 83.62/0.2116/0.8929; the small one-fold help/harm count
of 6/2 was not significant (`p=0.289`). On the independently audited EyePACS
fold, the same rule changed 81.63/0.2283/0.8093 to
83.93/0.2170/0.7999, with help/harm 95/22 (`p=5.3144e-12`). Thus accuracy and
MAE improve consistently, while QWK falls by approximately 0.01 on both
datasets. This trade-off is explicit and must be reported.

Raw posterior median is now locked prospectively for all new folds. Raw class
MAP is rejected: versus posterior median, it adds only 0.34 accuracy points on
APTOS and 0.45 on EyePACS, while worsening balanced accuracy, QWK, and MAE on
both. Analytic deweighting is rejected because it gives no consistent
cross-dataset advantage and substantially harms APTOS. The other five rules
remain audit diagnostics and may not be selected per dataset or fold.
Historical checkpoints retain their serialized rule.

Every rule receives only selected-proof scores, so the diagnostic audit adds
no image or feature bypass. The proof's sufficiency and necessity statements
remain in the raw cardinality-score domain. MOSAIC's novelty is the exclusive,
replayable proof computation, not a choice among conventional point-decision
rules. Posterior median itself is not claimed as a contribution.

## 5. Hiding evidence without creating fake images

Do **not** black out raw image squares and interpret the resulting score drop
as causal evidence. Black, grey, blur, and inpainting baselines can introduce
out-of-distribution content; mask colour and shape can themselves encode the
label.

MOSAIC can intervene at one explicit boundary-witness node:

\[
\operatorname{do}(\lambda_{n,i,k}=0).
\]

The coherent whole-region intervention instead replaces the complete local
state by normal, \(\rho_{n,i}=(1,0,\ldots,0)\), and therefore sets all four
nested boundary witnesses for that cell to zero.

For a fixed certificate, the exact direct effect of hiding selected cell \(i\)
at boundary \(k\) is

\[
\delta_{n,i,k}
=\mathbf1[i\in S_{n,k}^*]\left[
F_k(\boldsymbol\lambda^+_{n,k,m_k^*})
-F_k(\boldsymbol\lambda^+_{n,k,m_k^*}\setminus\lambda_{n,i,k})
\right].
\]

Using the count distribution of the other selected witnesses gives the
equivalent closed form

\[
\delta_{n,i,k}
=\mathbf1[i\in S_{n,k}^*]\lambda_{n,i,k}
\sum_{r=1}^{R}\alpha_{k,r}
P(C_{S_{n,k}^*\setminus i}=r-1).
\]

No backward pass, gradient approximation, surrogate mask network, new image,
or second CNN evaluation is involved. The value is the literal difference in
the circuit's boundary probability.

For the cumulative boundary \(j\ge k\), holding the other transitions fixed,

\[
q_{n,j}-q_{n,j}^{(-i,k)}
=\delta_{n,i,k}
\prod_{\substack{\ell=0\\\ell\ne k}}^{j}c_{n,\ell}.
\]

If the deterministic proof is recomputed after hiding \(i\), another region
may replace it. Therefore report two quantities:

1. **proof-conditional pivotality**: hide \(i\) while holding the proof fixed;
2. **adaptive replacement effect**: hide \(i\), recompute the proof, and report
   the end-to-end grade change.

Adaptive reprojection is not guaranteed monotone: after evidence is reduced,
the projection can expand and admit replacement regions, making its selected
score larger even though the dense and fixed-proof scores decreased. Treat
that behaviour as a redundancy diagnostic, not a contradiction.

Do not call every selected cell universally causally necessary. What is proven
is tolerance-conditioned minimum normal-baseline sufficiency and
\(\rho_n\)-collective score necessity for the fixed witness ledger.

## 6. Training objective

The primary loss is the balanced continuation likelihood:

\[
\mathcal L_{\mathrm{CCL}}(n)
=-\sum_{k<y_n}w_{k,1}\log(c_{n,k}+\epsilon)
-\mathbf1[y_n<K-1]w_{y_n,0}\log(s_{n,y_n}+\epsilon).
\]

Using the directly evaluated stop probability \(s\), rather than a floating-
point subtraction \(1-c\), preserves the recovery gradient of a confidently
wrong grade-0 example when a dense continuation rounds to one.

Weights are estimated inside each training fold over the at-risk population
for each boundary. Use effective-number smoothing and cap the rare late-boundary
weights. Do not balance only the five nominal classes: the useful imbalance
structure lives in the four at-risk transitions.

Use the same continuation likelihood on the dense transitions
\(\widetilde c_{n,k}\), denoted \(\mathcal L_{\mathrm{dense}}\). It supplies
label gradients to witnesses currently outside the hard certificate and
prevents an early random prefix from becoming self-reinforcing. This is the
same circuit before projection, not a second classifier.

The complete initial objective is deliberately small:

\[
\mathcal L
=\mathcal L_{\mathrm{CCL}}
+\eta_d\mathcal L_{\mathrm{dense}}
+\lambda_{\mathrm{stab}}\mathcal L_{\mathrm{stab}}.
\]

Here \(\mathcal L_{\mathrm{stab}}\) is Jensen--Shannon consistency between the
regional witness distributions of two geometry-identical photometric views.
Start with \(\lambda_{\mathrm{stab}}=0\); enable it only if witness maps are
unstable. No CLIP loss, concept loss, generic multi-task head, or auxiliary
global CE path is added.

Formal cached-screen training schedule:

1. initialise the encoder from EfficientNetV2S and the local-state bias toward
   normal;
2. train the head for 3--5 cached-feature epochs with the dense monotone circuit
   to avoid an unstable random hard prefix;
3. enable deterministic proof projection with \(\varepsilon_s=0\), then
   gradually relax it to 0.02 rather than imposing aggressive compression at
   the first projected epoch;
4. train only through the projected proof path; top-prefix indices and \(m^*\)
   are treated as piecewise-constant during backpropagation, while retaining
   \(\eta_d=0.1\) as a training-only dense-ledger stabilizer; and
5. fine-tune the last permitted local encoder stage only after the cached head
   passes Gate 1.

The first executable APTOS fold-0 viability pilot is explicitly a deviation:
it fine-tunes the permitted ImageNet trunk from epoch 1 with a lower encoder
learning rate, uses four dense warm-up epochs, and then applies the same
four-epoch proof-tolerance ramp. It is not reported as the formal cached Gate-1
baseline screen.

The dense warm-up and auxiliary dense-ledger term are optimization devices, not
inference bypasses. Every checkpoint evaluated as MOSAIC must use the projected
score for its reported prediction. “Proof-exclusive” means the final numerical
transition is computed only from selected witnesses; the dense ledger still
determines which witnesses are selected and is part of the replayable forward
trace.

## 7. What the explanation contains

For each predicted image, return a machine-readable certificate:

- the predicted class distribution;
- four transition probabilities \(c_k\);
- the learned extent distributions \(\alpha_k\);
- the selected cell coordinates and their known receptive-field boxes;
- the local categorical states \(\rho_i\);
- retained and complement count distributions and tail probabilities;
- sufficiency gap \(\widetilde c_k-c_k\);
- complement residual \(F_k(\boldsymbol\lambda_k^-)\);
- proof-conditional pivotality for every selected cell.

The current certificate implements the items above, together with source and
checkpoint hashes and a serialized CPU/CUDA replay tolerance. Adaptive
replacement effects and tied/substitute proof enumeration remain mandatory
analysis extensions before a full paper claim; they are not part of the first
viability pilot.

The visual overlay shows receptive-field boxes or support contours at their
native lattice resolution. It must not interpolate them into apparently
pixel-precise lesion borders.

## 8. Guarantees and non-guarantees

Unit tests can establish:

- nested local ordinal witness probabilities;
- coordinatewise monotone transition laws;
- minimum-cardinality dual proof for a fixed witness ledger;
- certificate replay of the transition used for prediction within the
  serialized numerical contract;
- analytic/direct witness-intervention agreement within that contract;
- non-increasing dense and fixed-proof severity after removing evidence;
- ordered cumulative probabilities and normalized classes; and
- no classifier bypass around the proof.

The project cannot establish from EyePACS/APTOS grades alone:

- that a witness is a particular named lesion;
- pixel-accurate segmentation;
- causal effect of physically removing pathology;
- clinical completeness of a certificate; or
- guaranteed accuracy improvement or paper acceptance.

The valid phrase is **fine-grid, computationally faithful ordinal evidence**,
not pixel-level lesion explanation.

## 9. Mandatory anti-cheating tests

The explanation can be mathematically correct yet clinically meaningless if
the local encoder learns shortcuts. Run all of the following:

1. **receptive-field audit**: alter pixels outside a cell's theoretical support
   and verify its witness probability does not change beyond numerical tolerance;
2. **metadata/mask-only test**: predict the grade from source dimensions,
   valid-cell count, or selected coordinates alone; canonical mask-count
   performance must equal the majority baseline, and any acquisition-format
   shortcut from source dimensions must be reported;
3. **support-preserving shuffle**: keep the proof locations but shuffle their
   witness values across images; accuracy must collapse;
4. **coordinate shuffle**: preserve witness values but randomize positions;
   the symmetric classifier should be invariant, while the overlay changes;
5. **label randomization**: fine-grid maps must lose stable retinal structure;
6. **weight randomization**: explanations must change when the local encoder is
   randomized;
7. **outside-fundus test**: no selected witness may lie outside the valid field;
8. **augmentation stability**: map coordinates back through known transforms
   and measure top-proof overlap and pivotality correlation; and
9. **substitute-proof audit**: report how often another equally small proof can
   replace the canonical prefix.

These tests directly address selector--predictor collusion and mask-shape label
leakage identified in prior rationale literature.

## 10. Repo-specific implementation status

The core has been implemented as a clean parallel path rather than branches
inside OPTIC-C:

- `models/mosaic.py`: local ordinal state, simplex- and log-tail-stabilized
  truncated count law, dual-proof projection, continuation cascade, and
  intervention effects;
- `models/local_efficientnet.py`: bounded spatial taps and receptive-field
  metadata;
- `utils/spatial_mask.py`: one image-independent canonical ellipse shared by
  data preprocessing and model fallback;
- `models/mosaic_model.py`: exclusive image-to-proof model;
- `losses/mosaic.py`: at-risk likelihood and stability primitive;
- `Datasets/mosaic_data.py`: tight-field crop, direct canonical-square resize,
  fixed-support preprocessing, and disjoint APTOS/EyePACS splits;
- `training/mosaic_trainer.py` and `train_mosaic.py`: end-to-end pilot training,
  validation-only architecture selection, complete resume state, and diagnostics;
- `inference/mosaic_certificate.py` and `export_mosaic_certificates.py`:
  protected, replayable fixed-proof certificates; and
- `tools/cache_spatial_features.py`: fp16 raw spatial-map cache writer keyed by
  source metadata, preprocessing, and encoder state; and
- `tools/audit_mosaic_shortcuts.py`: validation-only acquisition-format and
  fixed-mask shortcut audit that never reads outer-test images by default.

The formal cached head-screen reader and matched baseline heads, adaptive
replacement analysis, substitute-proof enumeration, and the anti-cheating
experiment drivers remain to be implemented after the viability pilot passes.

MOSAIC must not instantiate CTOT, GPA, CGPM, BiomedCLIP, PCOL, SCOLw, CORAL,
AttentionPool, or a global classifier.

## 11. Mandatory unit tests

For random, tied, empty-field, and saturated evidence tensors:

1. truncated Poisson--binomial tails match brute-force Bernoulli enumeration;
2. \(F_k(0)=0\), \(0\le F_k\le1\), and \(F_k\) is monotone;
3. local witness probabilities are nested over boundaries;
4. brute-force subset search on small \(P\le14\) agrees with the top-prefix
   minimum dual proof;
5. a proof always exists and its two inequalities hold;
6. replay from the serialized selected witnesses matches the forward transition
   within the explicit numerical contract (default absolute and relative
   tolerance \(2\times10^{-5}\));
7. analytic direct effect matches explicit \(\lambda_i=0\) intervention;
8. removing any witness cannot increase the dense or fixed-proof transition,
   later cumulative probability, or expected grade; adaptive reprojection is
   tested separately and is not assumed monotone;
9. class probabilities are non-negative and sum to one;
10. permutation of witness-value/coordinate pairs does not change prediction;
11. invalid retinal cells have exactly zero witness probability and are never selected;
12. gradients remain finite for witness probabilities near 0 and 1 and for
    saturated tails; and
13. the optimized block-tree and prefix/suffix implementation matches the
    serial \(O(PR)\) recurrence within numerical tolerance; and
14. there is no model parameter path from full-canvas pooled features to the
    output.

## 12. Fast falsification sequence

### Gate 0: deterministic code tests

Implement only the cardinality, proof, continuation, and certificate tensors.
Do not touch the backbone until every invariant passes.

Benchmark forward and backward at \(P=12{,}544\), \(R=32\), the intended batch
size, and four boundaries before building the encoder. If the exact layer is
not practical, use the preregistered stride-16 fallback
\(P\approx3{,}136\); do not silently replace the count law by a summed-rate
approximation.

Stop immediately if minimum proof or intervention equality fails.

**Current status:** local Gate 0 passes. The cluster job repeats all tests,
adds CUDA-to-CPU certificate replay, benchmarks the proof core on CUDA, and
performs a configured-batch encoder backward memory check.

### Pre-Gate-1 APTOS viability pilot

Run one seed of fold 0 end to end, with `--skip_test`, before implementing the
large matched-baseline screen. This pilot is allowed to answer only whether
the model learns non-collapsed validation performance and compresses its proof.
It does not establish superiority and is not formal Gate 1.

### Gate 1: APTOS cached-map screen

Cache spatial features from the current pretrained EfficientNet for one fixed
APTOS split. Screen head-only models using identical folds and three seeds:

1. corrected ordered CORAL + global average pooling;
2. sparse local class map + average pooling (Donteu-style collision baseline);
3. Additive MIL local logits;
4. SATOMIL-style four-boundary attention;
5. cumulative local states + max pooling;
6. CROWN's coarse nine-region Poisson-binomial circuit;
7. global CORN/continuation-ratio head;
8. fixed 90%-cumulative top-prefix extraction (MorphoXAI-style);
9. MI-DORF-inspired latent-ordinal cardinality baseline;
10. SoftCAM and, if feasible, an AIM/SST-style learned selector;
11. MOSAIC dense Poisson--binomial cardinality without proof projection; and
12. full proof-projected MOSAIC.

Primary selection metric is mean validation accuracy, matching the established
OPTIC protocol; QWK and MAE are reported secondary metrics. ReduceLROnPlateau
is driven by validation loss rather than a discrete evaluation metric. Also
report certificate size and complement residual.

Use a corrected baseline on the identical split. Promote only if MOSAIC mean
validation QWK is no more than 0.01 below that baseline, with accuracy and then
MAE as fixed tie-breakers, and the proof remains nontrivially compressed.

Stop if full MOSAIC is materially worse than the corrected baseline or if the
proof uses most of the retina on most images.

### Gate 2: APTOS locality--accuracy frontier

Partially fine-tune RF-small, RF-medium, and RF-large using the same seeds.
Measure:

- accuracy, QWK, MAE, balanced accuracy, macro-F1;
- theoretical/empirical receptive field;
- median proof area and 90th percentile;
- sufficiency gap and complement residual;
- fixed-proof and adaptive deletion curves; and
- augmentation stability.

Promote the smallest receptive field within 0.3 accuracy points and 0.01 QWK
of the best variant. If only RF-large works, the claimed micro-region
interpretability is not supported and the project stops or is reframed.

### Gate 3: two-fold EyePACS pilot

Freeze architecture and hyperparameters before EyePACS. Run two folds. Promote
to ten-fold CV only if:

- mean QWK is competitive with OPTIC-C;
- accuracy is not materially worse;
- grade-3/4 recall does not collapse;
- the same proof tolerances work without retuning per fold;
- no mask-only or receptive-field audit fails; and
- certificate sizes remain meaningfully smaller than the valid retina.

The performance ambition is more than 86% accuracy and 0.82 QWK, but it is not
a gate that can be guaranteed in advance.

### Gate 4: final experiment

Compare the frozen MOSAIC configuration with OPTIC-C and the closest
reproducible baselines. Save all out-of-fold predictions and certificates.
Report fold means, standard deviations, paired bootstrap confidence intervals,
per-grade recall, confusion matrices, runtime, parameters, FLOPs, and memory.

## 13. Required ablations

1. nominal local class maps versus nested local ordinal states;
2. max, mean, Additive MIL, and learned Poisson--binomial cardinality
   aggregation;
3. independent boundary BCE versus continuation likelihood;
4. dense cardinality score versus proof-exclusive score;
5. sufficiency-only prefix versus dual sufficiency/necessity prefix;
6. fixed \(r=1\) versus learned \(\alpha_k\);
7. \(R=16,32,64\);
8. RF-small, RF-medium, RF-large;
9. global-image bypass as a deliberately unfaithful accuracy ceiling;
10. balanced versus unbalanced at-risk likelihood; and
11. no stability loss versus photometric witness consistency.

The global-bypass ablation is useful precisely because it quantifies the price
of structural faithfulness. It cannot be called MOSAIC.

## 14. Runtime strategy

CTOT is not the main reason OPTIC-C trains for 35--40 hours. Ten
EfficientNetV2S evaluations per image dominate. MOSAIC runs one full-canvas
convolutional pass and removes the global-tile duplicate, Transformer, GPA,
CGPM, and text branches. The spatial proof head is small; sorting four maps is
minor relative to the CNN.

For rapid iteration:

1. cache fp16 spatial feature maps once on APTOS;
2. train every proof/cardinality/loss ablation on the cache;
3. fine-tune only the selected local stage for 15--25 epochs;
4. run two EyePACS folds before full CV; and
5. parallelize final folds only after preregistration.

Spatial caches are larger than 1280-D GAP caches. Projected 112-by-112 maps
with 32 fp16 channels are about 0.8 MiB per image, roughly 3 GiB for APTOS.
Cache hashes must prevent leakage across preprocessing or backbone checkpoints.

## 15. Main risks

1. Image-level grades may not identify genuinely local witness states.
2. A small receptive field may miss morphology needed for venous beading,
   IRMA, or neovascularisation and reduce accuracy.
3. A larger receptive field may recover accuracy while weakening localization.
4. Dense tiny background probabilities can accumulate; grade-0 supervision and normal
   bias initialization must prevent this.
5. Adjacent cells can describe one lesion multiple times. The learned count is
   evidence extent, not literal lesion count.
6. Conditional independence of Bernoulli witnesses is an inductive bias, not
   a biological law.
7. The proof may be large when the image contains genuinely redundant evidence.
8. Hard sorting and prefix changes can create optimization discontinuities.
9. APTOS head rankings may not survive end-to-end EyePACS fine-tuning.
10. Grade-label-only evaluation cannot establish lesion semantics or pixel
    segmentation quality.
11. The exact novelty depends on the full coupling; every component alone has
    close prior art.

## 16. Six-week project target

| Week | Deliverable |
|---|---|
| 1 | proof/cardinality layer, mathematical tests, receptive-field audit |
| 2 | APTOS spatial cache and head-only falsification |
| 3 | APTOS locality--accuracy frontier and anti-cheating tests |
| 4 | two-fold EyePACS pilot |
| 5 | selected full experiment and required ablations |
| 6 | certificate figures, statistics, and first paper draft |

This schedule assumes one working implementation path and strict stopping at
failed gates. It is not a promise that ten-fold EyePACS training will finish
inside six weeks.

## 17. Paper story if the gates pass

Working title:

> **MOSAIC: Minimum Ordinal Proofs for Structurally Faithful Disease
> Severity Grading**

Three contributions:

1. a fine-grid nested ordinal witness representation with a learned
   boundary-specific focal-versus-distributed Poisson--binomial cardinality
   law;
2. a deterministic, tolerance-conditioned minimum dual proof projection that
   is the exclusive continuation prediction path; and
3. exact proof-conditional boundary-witness intervention effects and
   replayable certificates, validated without raw-image deletion artifacts.

The central figure should show one fundus, four boundary-specific witness maps,
the top-prefix proof construction, retained/complement cardinality curves, and
the continuation product. It must not look like another attention heatmap
paper.
