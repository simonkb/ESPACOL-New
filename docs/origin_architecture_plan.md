# ORIGIN: Severity Is a Generator

**Ordinal Regional Intervention Generator Network**

Implementation target: APTOS, EyePACS DR, then BUSI

Status: architecture frozen for the first controlled experiment

## One-sentence idea

ORIGIN replaces the usual pooled classifier with a spatial field of local,
nonnegative severity-transition generators; their conserved sum produces the
entire ordinal posterior, and subtracting any reported ledger unit gives an
exact same-circuit intervention explanation.

## Forward pass

1. A multi-scale image encoder returns spatial maps at strides 4, 8, 16, and
   32; it never globally pools features for prediction.
2. Local heads produce nonnegative severity atoms
   \(a_{s,i,m}=\operatorname{softplus}(z_{s,i,m})\).
3. A level-\(m\) atom supports its boundary and all prerequisites:
   \(\rho_{s,i,k}=\sum_{m\ge k}a_{s,i,m}\).
4. Fixed geometry weights, learned positive boundary calibration, and a learned
   simplex across scales turn these into local boundary rates \(r_{s,i,k}\).
5. Every rate vector becomes an upper-bidiagonal local generator \(Q_{s,i}\),
   and the image generator is
   \(Q=Q_\varnothing+\sum_{s,i}Q_{s,i}\).
6. The only grade posterior is
   \(p=e_0^\top\exp(Q)\). The prospectively locked benchmark decision is MAP;
   posterior median and expected grade are retained as ordinal diagnostics.
   The monotonic intervention theorem applies to cumulative probabilities,
   expected grade, and posterior quantiles—not to MAP.
7. The native explanation is the local boundary-rate ledger plus exact replay
   \(p^{(-A)}=e_0^\top\exp(Q-\sum_{i\in A}Q_i)\).

## What is new—and what is not

The scoped hypothesis is that ORIGIN is the first image severity classifier to
combine all of the following:

- nonnegative generator contributions emitted by every spatial unit;
- conserved regional superposition into one adjacent-state pure-birth chain;
- an ordinal posterior computed directly by the matrix exponential; and
- ledger-unit explanations measured by exact replay of that same classifier.

The novelty is **not** CTMCs, pure-birth models, matrix exponentials, cumulative
ordinal heads, additive MIL, weakly supervised maps, noisy-OR, or deletion
tests individually. For two classes the decoder is exactly noisy-OR, so the
scientific claim begins at three severity levels.

That boundary follows a targeted primary-source audit. Cumulative and
continuation ordinal heads already exist in
[CORAL](https://arxiv.org/abs/1901.07884) and
[CORN](https://arxiv.org/abs/2111.08851); threshold-local ordinal MIL exists in
[SATOMIL](https://openaccess.thecvf.com/content/WACV2025/html/Shiku_Ordinal_Multiple-Instance_Learning_for_Ulcerative_Colitis_Severity_Estimation_with_Selective_WACV_2025_paper.html);
and exact additive regional accounting exists in
[Additive MIL](https://proceedings.neurips.cc/paper_files/paper/2022/hash/82764461a05e933cc2fd9d312e107d12-Abstract-Conference.html).
Fine-grained disease evidence is also established by
[Sparse Activations](https://proceedings.mlr.press/v227/donteu24a.html), while
[DiDiCM](https://arxiv.org/abs/2511.20263) rules out any broad claim of being
the first CTMC image classifier. No source found through 8 September 2026
combined ORIGIN's four defining operations end to end, but literature absence
cannot be guaranteed; the paper must use the scoped phrase “to our knowledge.”

## Why it is interpretable by construction

There is no hidden classification bypass. If a ledger unit is absent from the
ledger, it cannot affect the image-dependent posterior. Each entry reports:

- its input-space receptive-field footprint and scale;
- its incremental severity atoms;
- the exact rates it contributes to every ordinal boundary;
- the full posterior and the posterior after removing that entry; and
- the exact change in every cumulative boundary and expected grade.

This establishes computational faithfulness to the internal generator.
Receptive fields overlap—at strides 16 and 32 they cover the full 640-pixel
input—so a ledger deletion is not presented as a causal pixel intervention.
Lesion semantics and pixel-level causality require separate evaluation.

## Loss

\[
\mathcal L=-\log p_y+
\eta\frac1{K-1}\sum_{k=0}^{K-2}
\left(P(Y>k)-\mathbf1[y>k]\right)^2+
\lambda_b\log\!\left(1+\sum_{s,i,k}r_{s,i,k}\right).
\]

The last term is disabled by default and must be treated only as a
**total-rate magnitude penalty**. It cannot select a sparse spatial
decomposition: redistributing a fixed boundary-rate total among cells leaves
both the posterior and this penalty unchanged. The default NLL+RPS objective is
an unweighted proper scoring objective under ordinary random sampling;
empirical calibration is measured rather than assumed. Dataset imbalance is
reported through balanced accuracy, macro-F1, per-grade recall, QWK, and the
full confusion matrix rather than hidden by accuracy alone.

## First experiment gate

Use APTOS fold 0 inner validation only. Run the exact same ConvNeXt-Tiny encoder
and preprocessing with four heads: softmax, CORAL, CORN, and ORIGIN. The first
architecture gate passes only if:

- ORIGIN is within 1.0 accuracy point of the strongest matched head;
- QWK is at least 0.86 and accuracy at least 82% on the current inner split;
- all generator, nesting, monotonicity, and replay tests pass;
- invalid/background cells contribute exactly zero; and
- optional total-rate regularization does not sacrifice more than 0.5 accuracy
  points; no spatial-sparsity claim is made for this term.

After that configuration is locked, run EyePACS fold 0. The development goal is
at least 85% validation accuracy and QWK above 0.82. These are experimental
targets, not guarantees.

## Required ablations

1. ConvNeXt softmax vs CORAL vs CORN vs ORIGIN.
2. Single-scale vs the conserved multi-scale ledger.
3. Independent boundary atoms vs cumulative prerequisite atoms.
4. CTMC posterior vs logits derived from the same total rates.
5. With/without explicit null prior generator.
6. With/without total-rate magnitude regularization.
7. Posterior median vs MAP vs rounded expected grade.
8. Exact internal deletion vs pixel masking and re-encoding.

## Hard failure modes

- image-level labels identify aggregate boundary rates, not unique allocations
  across scales, cells, atoms, or the learned null prior;
- positive-only evidence cannot encode explicit counter-evidence;
- global context in a feature receptive field weakens pixel-local claims;
- higher accuracy may come from the new encoder rather than the generator,
  hence the mandatory matched-head comparison;
- grade labels may contain clinical inconsistency or label noise;
- a foundation-model claim is invalid until broad pretraining and transfer are
  demonstrated. [RETFound](https://www.nature.com/articles/s41586-023-06555-x),
  for example, used 1.6 million retinal images and multiple downstream tasks;
  ORIGIN should be called foundation-compatible until comparable evidence is
  produced.

The complete literature audit and source ledger are maintained separately in
the internal research record.
