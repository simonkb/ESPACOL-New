"""
Faithfulness loop — STEP 1 (concept-drop check), shared logic.

Implements the first term of the ESPACOL "Perturbation–Consistency Faithfulness
Loop" (poster panel f) as a *demo* (measure + visualize, no training, no loss
optimisation, no backprop into weights):

    1. Get the actual prediction + per-concept cosine scores  c_k   (the WHY).
    2. Take the TOP concept's heatmap  H_c  (gamma-aligned per-concept LayerCAM).
    3. Soft-occlude the H_c region of the input image.
    4. Re-forward the occluded image through the SAME model → recompute scores c'_k.
    5. Check whether the masked concept's score actually DROPS:
           drop   = c_k - c'_k         (positive ⇒ score fell ⇒ FAITHFUL)
           L_drop = ReLU(c'_k - c_k)   (>0       ⇒ score rose ⇒ UNFAITHFUL)

This module is dataset-agnostic: it operates on tensors, an
`ExplainabilityPipeline`, and the `result` dict returned by `pipeline.explain`.
The two thin drivers `demo_faith_busi.py` / `demo_faith_dr.py` supply the
dataset-specific setup and call these helpers.
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Make the repo root (parent of Hady/) importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from explainability import _unpack_out, LayerCAM


# ─────────────────────────────────────────────────────────────────────────────
# Occlusion (soft-mask the top concept's heatmap region)
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_blur(x: torch.Tensor, sigma: float = 21.0) -> torch.Tensor:
    """
    Gaussian-blur an (N,C,H,W) tensor. Prefer torchvision; fall back to a
    separable depthwise conv so the module also runs without torchvision
    (e.g. on a login CPU).
    """
    k = int(2 * round(3 * sigma) + 1)  # odd kernel size
    try:
        from torchvision.transforms.functional import gaussian_blur as tv_blur
        return tv_blur(x, kernel_size=[k, k], sigma=[float(sigma), float(sigma)])
    except Exception:
        half = k // 2
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - half
        g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        kx = g.view(1, 1, 1, k)
        ky = g.view(1, 1, k, 1)
        c = x.shape[1]
        xb = F.pad(x, (half, half, half, half), mode="reflect")
        xb = F.conv2d(xb, kx.expand(c, 1, 1, k), groups=c)
        xb = F.conv2d(xb, ky.expand(c, 1, k, 1), groups=c)
        return xb


def occlude_image(
    x: torch.Tensor,
    heatmap: np.ndarray,
    tissue_mask=None,
    baseline: str = "blur",
    threshold: float = 0.0,
    blur_sigma: float = 21.0,
):
    """
    Soft-occlude the heatmap region of an ImageNet-normalized input.

    Parameters
    ----------
    x           : (1,3,H,W) float tensor, ImageNet-normalized (~[-2,2]), on device.
    heatmap     : (H,W) ndarray in [0,1] — the top-concept LayerCAM H_c.
    tissue_mask : (H,W) ndarray in [0,1] or None. If given, the occlusion mask is
                  multiplied by it so the black background is never occluded.
    baseline    : "blur" (default, RISE-style — replace with a blurred copy, no hard
                  edge) or "gray" (replace with the ImageNet mean, which is 0 in
                  normalized space → a neutral-gray patch).
    threshold   : zero out mask values below this before occluding (sharpen H_c).
    blur_sigma  : Gaussian sigma for the "blur" baseline.

    Returns
    -------
    (x_occ, M) : x_occ is (1,3,H,W) on the same device/dtype as x; M is the (H,W)
                 occlusion mask actually applied (in [0,1]) for plotting.

    NOTE: the occlusion mask M is used RAW (clipped, optionally × tissue mask) — it is
    deliberately NOT renormalized/stretched, so a faint H_c stays a faint occlusion.
    """
    M = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    if tissue_mask is not None:
        M = M * np.clip(np.asarray(tissue_mask, dtype=np.float32), 0.0, 1.0)
    if threshold > 0.0:
        M = np.where(M >= threshold, M, 0.0).astype(np.float32)

    M_t = torch.from_numpy(M).to(device=x.device, dtype=x.dtype).view(1, 1, *M.shape)

    if baseline == "gray":
        base = torch.zeros_like(x)           # 0 == ImageNet mean in normalized space
    else:                                    # "blur"
        base = _gaussian_blur(x, sigma=blur_sigma)

    x_occ = x * (1.0 - M_t) + base * M_t     # broadcast over the 3 channels
    return x_occ, M


# ─────────────────────────────────────────────────────────────────────────────
# Re-forward & recompute concept scores on the occluded image
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def recompute_scores(pipeline, x_occ: torch.Tensor):
    """
    Forward x_occ through the SAME model and recompute per-concept cosine scores,
    reusing the pipeline's model and concept_explainer (no new pipeline build).

    Mirrors ExplainabilityPipeline.explain() exactly (same _unpack_out, same
    z_it-vs-feature selection rule, same ConceptExplainer.scores).

    Returns
    -------
    (scores_occ, reg_score_occ):
        scores_occ    : dict {concept_name: cosine} over ALL concepts.
        reg_score_occ : float continuous regression prediction of the occluded image.
    """
    model = pipeline.layercam.model
    model.eval()
    out = model(x_occ)
    feat, z_it, pred = _unpack_out(out)

    emb_dim = pipeline.concept_explainer.embeddings.shape[-1]
    why_emb = z_it if (z_it is not None and z_it.shape[-1] == emb_dim) else feat
    scores_occ = pipeline.concept_explainer.scores(why_emb.squeeze(0))
    reg_score_occ = float(pred.detach().cpu().squeeze().item())
    return scores_occ, reg_score_occ


# ─────────────────────────────────────────────────────────────────────────────
# Concept-drop check
# ─────────────────────────────────────────────────────────────────────────────

def concept_drop_check(scores_before, scores_after, top_concept, tol: float = 0.0):
    """
    The step-1 faithfulness test for the masked (top) concept.

        drop   = c_k - c'_k          (positive ⇒ score fell ⇒ faithful)
        L_drop = ReLU(c'_k - c_k)    (the loop's penalty: >0 ⇒ it rose ⇒ unfaithful)
        PASS   = drop > tol

    SIGN-AGNOSTIC by construction (uses the difference, never sign(cosine)). This
    directly handles the DR negative-cosine regime: for severe/proliferative DR all
    cosines are negative, yet a faithful H_c still yields a positive `drop`.

    Returns a dict: {concept, c_before, c_after, drop, l_drop, passed, drops}.
    """
    c_before = float(scores_before[top_concept])
    c_after = float(scores_after[top_concept])
    drop = c_before - c_after
    l_drop = max(0.0, c_after - c_before)  # ReLU(c' - c)
    drops = {
        k: float(scores_before[k]) - float(scores_after.get(k, scores_before[k]))
        for k in scores_before
    }
    return {
        "concept": top_concept,
        "c_before": c_before,
        "c_after": c_after,
        "drop": drop,
        "l_drop": l_drop,
        "passed": drop > tol,
        "drops": drops,
    }


# ─────────────────────────────────────────────────────────────────────────────
# One-call orchestration of step 1 (shared by the demos and the batch runner)
# ─────────────────────────────────────────────────────────────────────────────

def run_concept_drop(pipeline, x0, tissue_mask=None, occlusion="blur",
                     threshold=0.0, blur_sigma=21.0):
    """
    Run the full step-1 concept-drop check on one image:
    explain → pick the top concept → occlude its H_c → re-forward → drop-check.

    Does NOT remove the pipeline's LayerCAM hooks — the caller controls hook
    lifetime so a single pipeline can be reused across many images (then call
    `pipeline.remove_hooks()` once at the end).

    Returns
    -------
    (result, x_occ, M, scores_before, scores_after, reg_after, check)
        result        : the dict from pipeline.explain(..., concept_heatmaps=True)
        x_occ         : (1,3,H,W) occluded input
        M             : (H,W) occlusion mask actually applied
        scores_before : dict {concept: cosine} before occlusion
        scores_after  : dict {concept: cosine} after occlusion
        reg_after     : float regression score of the occluded image
        check         : dict from concept_drop_check (concept/drop/l_drop/passed/...)
    """
    result = pipeline.explain(x0, concept_heatmaps=True)
    top = result["concept_scores"][0][0]
    H_c = result["concept_heatmaps"][top]
    x_occ, M = occlude_image(x0, H_c, tissue_mask=tissue_mask, baseline=occlusion,
                             threshold=threshold, blur_sigma=blur_sigma)
    scores_after, reg_after = recompute_scores(pipeline, x_occ)
    scores_before = dict(result["concept_scores"])
    check = concept_drop_check(scores_before, scores_after, top)
    return result, x_occ, M, scores_before, scores_after, reg_after, check


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — occlusion SPECIFICITY (on-target vs off-target control)
# ─────────────────────────────────────────────────────────────────────────────

def control_masks(M, rng, n_roll=3):
    """
    Equal-area, equal-shape copies of mask `M` placed ELSEWHERE — the off-target
    controls. Two reflections (180° and left-right) plus `n_roll` random circular
    shifts. Moving the mask without changing its area isolates LOCATION as the only
    variable, so comparing on- vs off-target drop tests whether the heatmap's
    location (not just its size) is what carries the concept.
    """
    outs = [np.flip(M, (0, 1)).copy(), np.fliplr(M).copy()]
    H, W = M.shape
    for _ in range(n_roll):
        dy = rng.randint(H // 4, 3 * H // 4)
        dx = rng.randint(W // 4, 3 * W // 4)
        outs.append(np.roll(M, (dy, dx), (0, 1)).copy())
    return outs


def occlusion_specificity(pipeline, x0, tissue_mask=None, occlusion="blur",
                          blur_sigma=21.0, n_roll=3, rng=None, tol=0.0):
    """
    Step-3 control: does occluding the HEATMAP region drop the top concept MORE than
    occluding an equal-size mask placed elsewhere? A bare concept-drop (step 1) can
    be faithful localization, generic blur-sensitivity, or a trivially huge mask;
    this separates them.

        drop_on     = c_before - c_after(blur H_c region)
        drop_ctrl   = mean over control masks of  c_before - c_after(blur shifted mask)
        specificity = drop_on - drop_ctrl       (> 0 ⇒ the heatmap LOCATION matters)

    Does NOT remove the pipeline's LayerCAM hooks (caller controls lifetime).

    Returns a dict:
        top, c_before, c_on, drop_on, drop_ctrl, ctrl_drops, specificity,
        mask_frac (fraction of the image blurred), passed (drop_on > drop_ctrl+tol),
        reg_before, scores_before, and for plotting: M (on-target mask), x_on,
        M_ctrl0 / x_ctrl0 (the first control mask + its occluded image).
    """
    if rng is None:
        rng = random.Random(0)

    scores_before, reg_before = recompute_scores(pipeline, x0)
    top = max(scores_before, key=scores_before.get)
    vec = pipeline.concept_explainer.get_vector(top)
    all_v = pipeline.concept_explainer.embeddings
    H_c = pipeline.layercam.compute_concept(
        x0, vec, all_concept_vectors=all_v, temperature=10.0
    )

    M_on = np.clip(H_c.astype(np.float32), 0.0, 1.0)
    if tissue_mask is not None:
        M_on = M_on * np.clip(tissue_mask.astype(np.float32), 0.0, 1.0)

    # on-target: pass the mask directly (tissue already folded in)
    x_on, _ = occlude_image(x0, M_on, tissue_mask=None, baseline=occlusion, blur_sigma=blur_sigma)
    s_on, _ = recompute_scores(pipeline, x_on)
    drop_on = float(scores_before[top] - s_on[top])

    # off-target controls: same mask, moved
    ctrl_drops, M_ctrl0, x_ctrl0 = [], None, None
    for i, Mc in enumerate(control_masks(M_on, rng, n_roll=n_roll)):
        x_c, _ = occlude_image(x0, Mc, tissue_mask=None, baseline=occlusion, blur_sigma=blur_sigma)
        s_c, _ = recompute_scores(pipeline, x_c)
        ctrl_drops.append(float(scores_before[top] - s_c[top]))
        if i == 0:
            M_ctrl0, x_ctrl0 = Mc, x_c

    drop_ctrl = float(np.mean(ctrl_drops))
    specificity = drop_on - drop_ctrl
    return {
        "top": top,
        "c_before": float(scores_before[top]),
        "c_on": float(s_on[top]),
        "drop_on": drop_on,
        "drop_ctrl": drop_ctrl,
        "ctrl_drops": ctrl_drops,
        "specificity": float(specificity),
        "mask_frac": float(M_on.mean()),
        "passed": bool(drop_on > drop_ctrl + tol),
        "reg_before": float(reg_before),
        "scores_before": scores_before,
        "M": M_on, "x_on": x_on, "M_ctrl0": M_ctrl0, "x_ctrl0": x_ctrl0,
    }


def print_specificity_summary(spec, true_name, pred_before, encoder):
    verdict = ("LOCATION-SPECIFIC (faithful)" if spec["passed"] and spec["specificity"] > 0
               else "NOT location-specific")
    print("-" * 60)
    print(f"Top concept     : {spec['top']}")
    print(f"True / pred     : {true_name} / {pred_before}")
    print(f"c_before        : {spec['c_before']:+.4f}")
    print(f"drop on-target  : {spec['drop_on']:+.4f}   (blur the H_c region)")
    print(f"drop off-target : {spec['drop_ctrl']:+.4f}   (same mask moved elsewhere, mean of {len(spec['ctrl_drops'])})")
    print(f"specificity     : {spec['specificity']:+.4f}   (on - off; >0 ⇒ location matters)")
    print(f"mask fraction   : {100*spec['mask_frac']:.1f}% of the image blurred")
    print(f"VERDICT         : {verdict}")
    if encoder != "biomedclip":
        print("WARNING: --encoder random ⇒ concept vectors are placeholders; specificity is meaningless.")
    print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_mask(heatmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Suppress out-of-tissue activations, then renormalize to [0,1] (matches the
    explain demos' _apply_mask). Used ONLY for the WHERE panel, not the occlusion."""
    hm = heatmap * mask
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm


def build_faithfulness_figure(
    plt,
    image_np: np.ndarray,
    occ_np: np.ndarray,
    M: np.ndarray,
    top_concept: str,
    scores_before: dict,
    scores_after: dict,
    check: dict,
    mask: np.ndarray,
    true_name: str,
    pred_before: str,
    pred_after: str,
    dataset_name: str,
    baseline_word: str,
):
    """
    One row, 4 panels (mirrors the explain demos' figure style):

        Original | WHERE H_c (top concept) | Occluded input | before-vs-after bars

    The before/after grouped bar chart shows every concept's cosine score; the top
    (masked) concept's label is highlighted. The title carries drop, L_drop, PASS/FAIL.
    """
    import matplotlib.gridspec as gridspec
    import textwrap

    verdict = "PASS (faithful)" if check["passed"] else "FAIL (unfaithful)"

    fig = plt.figure(figsize=(18, 5.2))
    gs = gridspec.GridSpec(1, 4, wspace=0.5)

    # [0] Original ────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow((image_np * 255).astype(np.uint8))
    ax0.axis("off")
    ax0.set_title(f"Original\n(true: {true_name} | pred: {pred_before})")

    # [1] WHERE H_c of the top concept ────────────────────────────────────────
    where = _apply_mask(M, mask)
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(LayerCAM.overlay(image_np, where))
    ax1.axis("off")
    ax1.set_title(f"WHERE: {top_concept}\n(H_c — masked region)")

    # [2] Occluded input ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow((occ_np * 255).astype(np.uint8))
    ax2.axis("off")
    ax2.set_title(f"Occluded ({baseline_word})\npred: {pred_after}")

    # [3] before-vs-after concept scores ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    names = [c for c, _ in sorted(scores_before.items(), key=lambda kv: kv[1], reverse=True)]
    names_r = names[::-1]                      # highest score plotted at the top
    before_r = [scores_before[n] for n in names_r]
    after_r = [scores_after[n] for n in names_r]
    y = np.arange(len(names_r))
    h = 0.4
    ax3.barh(y + h / 2, before_r, height=h, color="steelblue", label="before (c_k)")
    ax3.barh(y - h / 2, after_r, height=h, color="orange", label="after (c'_k)")
    ax3.set_yticks(y)
    ax3.set_yticklabels([textwrap.fill(n, 18) for n in names_r], fontsize=8)
    for tick_label, name in zip(ax3.get_yticklabels(), names_r):
        if name == top_concept:
            tick_label.set_fontweight("bold")
            tick_label.set_color("crimson")
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_xlim(-1, 1)
    ax3.set_xlabel("Cosine similarity")
    ax3.legend(fontsize=8, loc="lower right")
    ax3.set_title(
        f"Concept scores  |  drop={check['drop']:+.3f}  L_drop={check['l_drop']:.3f}\n{verdict}"
    )

    fig.suptitle(
        f"ESPACOL faithfulness — concept-drop (step 1)  |  {dataset_name}  |  "
        f"top: {top_concept}  |  {verdict}",
        fontsize=12, fontweight="bold",
    )
    return fig


def build_specificity_figure(plt, image_np, on_np, off_np, spec,
                             true_name, pred_before, dataset_name, baseline_word):
    """
    Step-3 figure: Original | WHERE H_c | On-target occluded | Off-target occluded.

    Same-size mask, two placements: on the heatmap (panel 3) vs moved elsewhere
    (panel 4). If only the on-target occlusion drops the concept, the heatmap
    location is what carries it (faithful). Titles carry the drops + specificity.
    """
    import matplotlib.gridspec as gridspec

    top = spec["top"]
    verdict = ("LOCATION-SPECIFIC ✓" if spec["passed"] and spec["specificity"] > 0
               else "not location-specific ✗")

    M_vis = spec["M"] / (spec["M"].max() + 1e-8)  # stretch faint maps for display only

    fig = plt.figure(figsize=(18, 5.2))
    gs = gridspec.GridSpec(1, 4, wspace=0.18)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow((image_np * 255).astype(np.uint8)); ax0.axis("off")
    ax0.set_title(f"Original\n(true: {true_name} | pred: {pred_before})")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(LayerCAM.overlay(image_np, M_vis)); ax1.axis("off")
    ax1.set_title(f"WHERE: {top}\n(H_c — the masked region)")

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow((on_np * 255).astype(np.uint8)); ax2.axis("off")
    ax2.set_title(f"ON-target occluded ({baseline_word})\n"
                  f"c {spec['c_before']:+.2f}→{spec['c_on']:+.2f}  drop {spec['drop_on']:+.3f}")

    ax3 = fig.add_subplot(gs[3])
    ax3.imshow((off_np * 255).astype(np.uint8)); ax3.axis("off")
    ax3.set_title(f"OFF-target (mask moved)\n"
                  f"mean drop {spec['drop_ctrl']:+.3f}  ({len(spec['ctrl_drops'])} controls)")

    fig.suptitle(
        f"ESPACOL faithfulness — occlusion specificity (step 3)  |  {dataset_name}  |  "
        f"top: {top}  |  specificity = {spec['drop_on']:+.3f} − {spec['drop_ctrl']:+.3f} "
        f"= {spec['specificity']:+.3f}  →  {verdict}",
        fontsize=12, fontweight="bold",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_faith_summary(check, pred_before, pred_after, reg_before, reg_after, encoder):
    verdict = "PASS (faithful)" if check["passed"] else "FAIL (unfaithful)"
    print("-" * 60)
    print(f"Top concept   : {check['concept']}")
    print(f"c_k  (before) : {check['c_before']:+.4f}")
    print(f"c'_k (after)  : {check['c_after']:+.4f}")
    print(f"drop = c-c'   : {check['drop']:+.4f}   (positive ⇒ score fell ⇒ faithful)")
    print(f"L_drop=ReLU   : {check['l_drop']:.4f}   (ReLU(c'-c); >0 ⇒ unfaithful)")
    print(f"Grade pred    : {pred_before} -> {pred_after}  (reg {reg_before:.3f} -> {reg_after:.3f})")
    print(f"VERDICT       : {verdict}")
    if encoder != "biomedclip":
        print("WARNING: --encoder random ⇒ WHY concept vectors are placeholders; "
              "the drop is NOT meaningful. Use --encoder biomedclip with the recovered "
              "text projection for an interpretable faithfulness check.")
    print("-" * 60)
