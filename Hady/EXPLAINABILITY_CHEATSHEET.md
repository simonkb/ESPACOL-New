# Explainability — purpose cheatsheet

> Quick reference for *why* the ESPACOL explainability work exists and how it fits
> the project. Synthesized from [explainability.py](explainability.py),
> [explainability.ipynb](explainability.ipynb), the project deck
> [Explainable-Semantic-Prototype-FINAL_share.pdf](Explainable-Semantic-Prototype-FINAL_share.pdf),
> the main paper [Paper.pdf](../../Paper.pdf), and [CLAUDE.md](CLAUDE.md).

## The one-line answer

The grading model **predicts an ordinal severity grade but can't justify it**.
Explainability is a layer bolted on top of that frozen grader to make each grade
*defensible in clinical terms* — answering **WHY** (which clinical findings drove
the grade) and **WHERE** (which pixels), then proving those explanations are
trustworthy.

> *"Medical severity grading models predict labels but cannot explain why."*
> Goal: *"add semantic + spatially grounded explainability to ordinal contrastive learning."* (deck p.1–2)

## It's an add-on, not the original contribution

- The **main ESPACOL paper** ([Paper.pdf](../../Paper.pdf)) is purely about *accuracy*:
  ordinal regression + contrastive losses (PCOL, SCOLw) for imbalanced grading.
  A full-text scan found **zero** mentions of explainability / interpretability /
  saliency / XAI.
- Explainability is a **Hady-side extension** layered on the reproduced grader —
  [CLAUDE.md](CLAUDE.md) calls it *"features (demos, inference, explainability) on
  top of the ESPACOL model."*
- It runs on **frozen checkpoints** and never retrains the grader.
  Deck p.41: *"We build an explainability layer on an inherited grading model;
  we do not claim the grader."*

## The two questions it answers

| | Question | Mechanism | Why it matters |
|---|---|---|---|
| **WHY** | Which clinical findings? | Cosine similarity of the image embedding to clinical-concept vectors (BI-RADS for BUSI, ICDR for DR) via `ConceptExplainer` | Turns an opaque number into named findings, e.g. *"venous beading +0.88"* on a Severe NPDR image |
| **WHERE** | Which pixels? | Multi-scale Grad-CAM++ / `LayerCAM` fused over 4 EfficientNet stages | Plain last-layer Grad-CAM dilutes tiny lesions (microaneurysms 2–4 px) ~200×; multi-scale recovers lesion detail |
| **WHY+WHERE** | Where does *each* finding appear? | `compute_concept` — one heatmap per concept ("which pixels look like spiculated margins?") | The novel contribution; per-finding spatial evidence, not one aggregate blob |

Code anchors in [explainability.py](explainability.py): `LayerCAM` (class), `ConceptExplainer`
(class), `LayerCAM.compute_concept` (concept-guided CAM), `SanityCheck` (validation).

## How it connects back to the grader

Two wires into the *same* EfficientNet-V2S backbone:

- **Semantic (WHY):** a training-time `γ·L_IT` image-text alignment loss pulls each
  image embedding toward its grade's BioMedCLIP clinical-text description, making
  per-concept scores meaningful. The notebook's post-hoc **alignment MLP** does this
  without retraining — its **BEFORE/AFTER comparison is "the core experiment"**:
  concept scores jump from a <0.10 noise band to >0.9 for the correct class.
- **Spatial (WHERE):** Grad-CAM++ hooks the grader's own feature stages, so
  explanations derive directly from the grading model and run post-hoc on any
  checkpoint.

## Three further purposes it serves

1. **Trust / external validation.** The Adebayo `SanityCheck` confirms heatmaps are
   model-dependent (not edge detectors). The notebook adds an **external
   vision-language-model reader study** — *"can someone OUTSIDE the team see what we
   see?"* — so explanations aren't self-graded.
2. **Diagnosing failure modes.** Explainability *surfaces where the model is weak*:
   it exposes the DR **"Moderate" grade collapse** (no signature lesion, bimodal
   embeddings) and honestly reproduces the paper's **negative localisation finding**
   (pointing accuracy ~0.05, IoU ~0.13). A frozen-backbone segmentation head then
   tightens heatmaps (coherence 38% → 88%).
3. **Quantifying the trade-off.** It measures that interpretability isn't free — the
   alignment loss cost **−11.9 pp DR accuracy**, framing semantic vs. regression
   objectives as conflicting under imbalance.

## In short

Explainability exists to make an accurate-but-opaque ordinal grader *trustworthy and
clinically legible* — say which findings and where, prove the explanations are
faithful, and expose where the model fails — all as a post-hoc layer that leaves the
grader's predictions untouched.
