# Obscuration & the Faithfulness Check — a step-by-step study note

This note explains, from first principles, **how we obscure (occlude) part of an image**
and **how we use that to test whether the model's concept explanation is faithful**. It
covers the two pieces we implemented:

- **Step 1 — concept-drop** (`demo_faith_busi.py`, `demo_faith_dr.py`)
- **Step 3 — occlusion specificity / the on-vs-off-target control** (`demo_faith_control.py`)

All the logic lives in [`faithfulness.py`](faithfulness.py).

---

## 0. Setup & notation

After the dataloader, an image is a tensor `x` of shape `(1, 3, H, W)` in
**ImageNet-normalized space**: each pixel is

```
x = (p − μ) / σ          p ∈ [0,1] raw pixel,  μ,σ = ImageNet mean/std per channel
```

so `x` lives in roughly `[−2, 2]`, **not** `[0,1]`. Everything below happens in this
normalized space. One consequence we use later: the ImageNet **mean pixel maps to 0**,
so "replace a region with neutral gray" = "set it to 0".

The model maps the image to an embedding `z_it ∈ ℝ¹²⁸` (L2-normalized). Each clinical
concept phrase maps (BioMedCLIP text → recovered projection `B'`) to a unit vector
`v_c` in the **same** space. A **concept score** is a cosine similarity:

```
c_k = cos(z_it, v_c) = z_it · v_c        ∈ [−1, 1]
```

High = "the image looks like this clinical concept." This is the **WHY**. The
per-concept LayerCAM heatmap `H_c ∈ [0,1]^{H×W}` is the **WHERE** — "which pixels drive
`cos(z_it, v_c)`."

---

## 1. Obscuration (the blur occlusion) — step by step

Goal: remove the visual evidence of concept `c` from its own heatmap region, with as
few side effects as possible.

**Step 1 — Get the concept's spatial map `H_c`.**
For the top concept, the pipeline computes a Grad-CAM++-style LayerCAM whose gradient
target is `cos(z_it, v_c)`. Output is normalized to `[0,1]`; high values = pixels the
concept score depends on.

**Step 2 — Turn it into a soft mask `M`.**
```
M = clip(H_c, 0, 1)              # occlusion weight per pixel: 1 = replace, 0 = keep
M = M × tissue_mask              # (optional) never occlude the black background
M = M · 1[M ≥ threshold]         # (optional) sharpen by dropping faint values
```
We deliberately **do not renormalize** `M` — a faint map must stay a faint occlusion,
otherwise we'd amplify weak evidence into a full mask.

**Step 3 — Build a blurred copy of the whole image.**
```
x_blur = G_σ * x                 # 2-D Gaussian convolution, per channel, σ default 21 px
```
A Gaussian blur is a **low-pass filter**: each pixel becomes a Gaussian-weighted average
of its neighbors. This destroys **high-frequency** structure — edges, margins, small
lesions, texture — while preserving coarse brightness/layout. At σ=21 on a 300-px image
the local detail in a region is essentially erased.

**Step 4 — Soft-composite (the occluded image).**
```
x_occ = x ⊙ (1 − M)  +  x_blur ⊙ M        # broadcast M over the 3 channels
```
Per pixel: `M≈1` → blurred, `M≈0` → untouched, in between → a linear blend. The
concept's spatial evidence is smeared out; the rest of the image is intact. Because the
blend is convex and in normalized space, `x_occ` stays a valid, artifact-free image.

**Why blur and not gray/black?** Setting a region to a constant injects a *sharp
artificial edge* and a flat patch that the model can react to — confounding the test.
Blur removes the **information** that defines the concept without adding a new feature.
(`--occlusion gray` is available as the simpler "set to ImageNet mean = 0" baseline.)

**Why soft (not a hard binary cut)?** Using `H_c`'s continuous values removes evidence
*proportionally* to each pixel's contribution and keeps the intervention smooth — the
poster's "soft masking is default for differentiability and clean gradient flow."

Code: `occlude_image(...)` and `_gaussian_blur(...)` in [`faithfulness.py`](faithfulness.py).

---

## 2. The concept-drop check (step 1) — step by step

This is a **causal ablation of the explanation**: the heatmap claims "concept c is
*here*"; we remove what's there and check the claimed effect.

1. **Measure before.** Forward `x` → `z_it` → `c_k = cos(z_it, v_c)`.
2. **Intervene.** Build `x_occ` by blur-occluding the top concept's region (§1).
3. **Measure after.** Forward `x_occ` through the **same** model → `z_it′` → `c′_k`.
4. **Compute the drop.**
   ```
   drop   = c_k − c′_k          # positive ⇒ score fell ⇒ faithful
   L_drop = ReLU(c′_k − c_k)    # the loop's penalty; > 0 only if the score ROSE
   ```
5. **Verdict.** `PASS if drop > 0`, else `FAIL`.

**Sign-agnostic.** We compare `c_k` vs `c′_k` (a *difference*), never the sign of the
cosine. For severe/proliferative DR all cosines are negative, yet a faithful map still
makes the score *more negative* → positive `drop`. The test is about the **change under
intervention**, not the absolute value.

Code: `run_concept_drop(...)`, `concept_drop_check(...)`. Demos: `demo_faith_busi.py`,
`demo_faith_dr.py`. Batch over all classes/grades: `make_faithful_figures.py`.

> ### ⚠️ Why step 1 alone is not enough
> A drop can happen for three different reasons:
> 1. We hit the concept's **real evidence** (faithful — what we want).
> 2. Blurring **any** equal chunk perturbs the embedding a little (**generic**).
> 3. The heatmap covers **most of the image**, so blurring it trivially tanks everything.
>
> The bare drop can't tell these apart. That is what step 3 fixes.

---

## 3. Occlusion specificity (step 3) — the on-vs-off-target control

Idea: if the concept truly lives at `H_c`, then blurring the **heatmap region** should
drop the score **more** than blurring the **same-size mask placed elsewhere**.

1. **On-target.** `drop_on = c_before − c_after(blur H_c region)` (as in step 1).
2. **Off-target controls.** Make equal-area, equal-shape copies of the mask **moved
   elsewhere** — two reflections (180° and left-right) plus a few random circular
   shifts (`control_masks`). Blur each and measure its drop.
   ```
   drop_off = mean over controls of ( c_before − c_after(blur shifted mask) )
   ```
3. **Specificity.**
   ```
   specificity = drop_on − drop_off       # > 0 ⇒ the heatmap LOCATION matters
   ```
4. **Verdict.** `LOCATION-SPECIFIC` if `drop_on > drop_off`. We also report
   `mask%` = fraction of the image blurred, to flag the "trivially huge mask" case.

Moving the mask without changing its area isolates **location** as the only variable —
so `specificity > 0` is evidence that the heatmap is pointing at the right place, not
just covering enough pixels.

Code: `occlusion_specificity(...)`, `control_masks(...)`, `build_specificity_figure(...)`.
Demo: `demo_faith_control.py`.

---

## 4. What we found (BUSI + DR)

Concept-drop alone (step 1, 40 images, 5 per class/grade) gave 24 PASS / 16 FAIL, but
the **specificity control** (step 3) reframed it. The three regimes:

| group | drop on | drop off | specificity | reading |
|---|---|---|---|---|
| BUSI **benign** | +0.19 | +0.01 | **+0.19** | ✅ genuinely location-specific |
| DR **moderate** | +0.14 | −0.02 | **+0.15** | ✅ genuinely location-specific |
| BUSI normal | +0.29 | +0.11 | +0.18 | partly real, partly big mask (≈42% area) |
| DR mild | +0.15 | +0.13 | +0.02 | ⚠️ generic — blur *anywhere* drops it |
| BUSI malignant, DR No-DR/Severe/PDR | ≈0 | ≈0 | ≈0 | inert (concept ungrounded) |

**Bottom line.** Obscuring the heatmap *really* throws off the concept similarity — in
the location-specific sense that matters — in only ~2 of 8 groups cleanly. Elsewhere the
drop is either generic blur-sensitivity or there is no effect at all. The decisive signal
is `on − off`, **not** the drop alone. This is exactly why the poster's loop has three
terms (concept-drop, orthofidelity, cross-concept specificity), not one.

**Caveats.** The heatmaps are diffuse (masks cover 18–42% of the image), so for big-mask
cases the off-target control inevitably overlaps the original region and *understates*
specificity — sharper maps (a `--threshold`, or the proposed auxiliary segmentation loss)
would separate on/off more cleanly. Results above are n=3 per group: directional, not
final. Faithfulness numbers are only meaningful with `--encoder biomedclip` + the
recovered text projections.

---

## 5. How to run

```bash
source slurm/_env.sh

# Step 1 — concept-drop, single image (4-panel figure)
python Hady/demo_faith_busi.py --seed 0 --save faith_busi.png --no-show
python Hady/demo_faith_dr.py   --grade 4 --save faith_dr.png  --no-show

# Step 1 — batch across all classes/grades (figures + panels + summary.csv)
python Hady/make_faithful_figures.py --per-class 5 \
    --busi_root "$BUSI_ROOT" --dr_root "$DR_ROOT" --out Hady/faithful_figures

# Step 3 — specificity, single image (on- vs off-target figure)
python Hady/demo_faith_control.py --dataset busi --seed 0 --save ctrl_busi.png --no-show
python Hady/demo_faith_control.py --dataset dr   --grade 2 --save ctrl_dr.png  --no-show

# Step 3 — specificity table across all classes/grades
python Hady/demo_faith_control.py --dataset both --per-class 3
```

Key knobs: `--occlusion {blur,gray}`, `--blur-sigma`, `--threshold` (sharpen the mask),
`--no-tissue-mask`, `--controls N` (number of off-target shifts), `--encoder`.
