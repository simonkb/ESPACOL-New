# Obscuration Results — what the concept-drop test actually showed (step by step)

This note reports **the results we got** from the obscuration (concept-drop, step 1) run, group by
group, with the real numbers. For *how* the obscuration and the check work (the blur, the math),
see the method note [`OBSCURATION.md`](OBSCURATION.md). Here we just read off and interpret the data.

All numbers come from [`faithful_figures/summary.csv`](faithful_figures/summary.csv) (40 images).

---

## 1. What was run

- **40 images**: 5 each of BUSI {normal, benign, malignant} and DR {grade 0..4}.
- **Occlusion**: `blur` (σ=21), restricted to tissue/fundus, applied to the **top concept's**
  heatmap `H_c`.
- **Encoder**: `biomedclip` + the recovered text projections (so the WHY scores are gamma-aligned).
- **Seed**: 0 (reproducible sampling).

For each image we recorded **two** things, before and after obscuration:

1. **WHY — the concept score** `c = cos(z_it, v_c)` for the masked top concept.
2. **Grade — the model's prediction** `reg` (continuous severity → rounded to a class).

> Headline: on the bare concept-score drop, **24 PASS / 16 FAIL**. But that single number hides
> almost everything — the interesting result is *which* groups respond, and the fact that **the
> grade prediction collapses in lockstep with the concept score** exactly where the heatmap is real.

---

## 2. The two effects we measured

```
drop          = c_before − c_after        # WHY: positive ⇒ concept score fell ⇒ PASS
grade shift   = reg_before → reg_after     # did the predicted severity change?
```

`PASS/FAIL` is defined on `drop` only. The **grade shift** is a second, independent readout — and it
turns out to be the more clinically meaningful one.

---

## 3. Results group by group

### BUSI — normal · top concept "no mass" · mean drop ≈ **+0.37**

| image | c → c′ | drop | reg → reg′ | grade | verdict |
|---|---|---|---|---|---|
| normal (86)  | 0.513 → 0.514 | −0.002 |  0.03 → −0.01 | Normal → Normal  | FAIL |
| normal (127) | 0.514 → 0.023 | **+0.492** | −0.02 → 1.04 | Normal → **benign** | PASS |
| normal (121) | 0.515 → 0.149 | **+0.367** | −0.02 → 0.66 | Normal → **benign** | PASS |
| normal (85)  | 0.513 → 0.036 | **+0.477** |  0.01 → 0.89 | Normal → **benign** | PASS |
| normal (131) | 0.515 → 0.002 | **+0.513** | −0.02 → 1.12 | Normal → **benign** | PASS |

4 of 5 show **huge** drops, and the grade flips **Normal → benign**. Read carefully, though: this is
likely the *opposite* mechanism from "removing evidence." Blurring a large patch of homogeneous
tissue **synthesizes a smooth, circumscribed blob** — which is exactly what a benign mass looks
like. So the model isn't losing "no-mass" evidence so much as the **blur is manufacturing a mass**.
The drop is real but its *cause* is an artifact (see the big-mask caveat in §7).

### BUSI — benign · top concept "oval shape" · mean drop ≈ +0.11 (one case carries it)

| image | c → c′ | drop | reg → reg′ | grade | verdict |
|---|---|---|---|---|---|
| benign (327) | 0.495 → 0.493 | +0.002 | 0.98 → 0.99 | Benign → Benign | PASS |
| benign (134) | 0.499 → 0.499 | +0.001 | 1.00 → 0.97 | Benign → Benign | PASS |
| benign (347) | 0.485 → −0.090 | **+0.575** | 0.98 → 1.79 | Benign → **malignant** | PASS |
| benign (76)  | 0.497 → 0.498 | −0.001 | 1.01 → 1.01 | Benign → Benign | FAIL |
| benign (205) | 0.494 → 0.505 | −0.011 | 0.96 → 0.88 | Benign → Benign | FAIL |

One genuinely responsive case (benign 347: drop **0.575**, grade flips to malignant — blurring the
margin destroys the "oval" evidence); the other four barely move. The group "mean" is misleading —
it's one strong case among four inert ones.

### BUSI — malignant · top concept "spiculated margins" · mean drop ≈ **0.000** (inert)

All five sit at `c ≈ 0.079` and **do not move** (drops 0.0000–0.0005); the grade stays Malignant.
The concept score is already near zero *and* the heatmap is diffuse, so blurring it changes nothing.
The PASS verdicts here are trivial (drops ~0.0004) — not evidence of faithfulness.

### DR — grade 0 (No DR) · top concept "cotton wool spots" · mean drop ≈ **0.002** (inert)

`c ≈ 0.557` → essentially unchanged; grade stays No DR. Note the top concept is a *lesion* ("cotton
wool spots") on a healthy retina — DR has **no "normal retina" anchor**, so a No-DR image is forced
to name a lesion it doesn't have. Nothing to remove → nothing drops.

### DR — grade 1 (Mild NPDR) · heterogeneous · mean drop ≈ +0.02 (messy)

| image | top concept | c → c′ | drop | reg → reg′ | verdict |
|---|---|---|---|---|---|
| 18185 | hard exudates    | 0.569 → 0.325 | **+0.244** | 1.98 → 0.01 (Mod→No DR) | PASS |
| 18405 | cotton wool spots| 0.557 → 0.556 | +0.001 | 0.01 → 0.01 | PASS |
| 27304 | cotton wool spots| 0.765 → 0.556 | **+0.208** | 1.28 → 0.00 (Mild→No DR) | PASS |
| 22360 | microaneurysms   | 0.112 → 0.416 | **−0.305** | 1.00 → 0.43 | **FAIL** |
| 39111 | microaneurysms   | 0.106 → 0.143 | −0.037 | 1.00 → 0.99 | FAIL |

The most mixed group: two strong PASS where the lesion concept fires and blurring it collapses the
grade, one **anti-faithful** case (22360, see §5), and two near-inert. The different top concepts per
image reflect Mild-NPDR's weak, scattered signal.

### DR — grade 2 (Moderate NPDR) · top concept "hard exudates" · mean drop ≈ **+0.18** (strongest)

| image | c → c′ | drop | reg → reg′ | grade | verdict |
|---|---|---|---|---|---|
| 35283 | 0.567 → 0.391 | **+0.176** | 2.00 → 0.02 | Moderate → **No DR** | PASS |
| 32284 | 0.567 → 0.572 | −0.004 | 1.99 → 1.99 | Moderate → Moderate | FAIL |
| 41796 | 0.572 → 0.328 | **+0.244** | 2.00 → 0.01 | Moderate → **No DR** | PASS |
| 25784 | 0.567 → 0.313 | **+0.254** | 1.99 → 0.01 | Moderate → **No DR** | PASS |
| 155   | 0.567 → 0.332 | **+0.235** | 1.99 → 0.00 | Moderate → **No DR** | PASS |

The cleanest, most consistent result. In 4/5, blurring the hard-exudate region drops the concept
score by ~0.18–0.25 **and** collapses the grade **Moderate → No DR** (`reg ≈ 2.0 → ~0.0`). This is
the clearest "remove the lesion, lose the diagnosis" behavior in the whole run.

### DR — grade 3 (Severe NPDR) · "dot and blot hemorrhages" · mean drop ≈ **0.000** (inert)

All cosines flat at **≈ −0.31**; drops in `[−0.002, +0.001]`; grade stays Severe (`reg ≈ 3.0`). The
negative cosine is expected (severe DR sits far from the concept-text cluster — see `OBSCURATION.md`
caveat), but the point here is it **doesn't move**: the heatmap carries no removable evidence.

### DR — grade 4 (PDR) · "preretinal hemorrhage" · mean drop ≈ **0.000** (totally inert)

Cosines flat at **≈ −0.20**, drops `≈ ±0.0001`, grade stays PDR (`reg ≈ 4.0`). Occlusion does
essentially nothing — the most inert group in the set.

---

## 4. The headline finding: concept-drop and grade-collapse move together

The most important pattern is the **coupling** between the two readouts:

- **Where occlusion removes real lesion evidence** (DR grade 2 hard exudates; DR grade 1 cases 18185
  & 27304), **both** the WHY concept score **and** the predicted grade collapse together
  (Moderate/Mild → No DR, `reg ≈ 2 → 0`).
- **Where the heatmap is inert** (BUSI malignant, DR No-DR / Severe / PDR), **neither** moves — the
  concept score and the grade both sit still.

That co-movement is the strongest evidence in the run that, *in the responsive groups*, the heatmap
is genuinely pointing at the diagnostically-relevant pixels: erasing them costs the model both its
concept match and its grade.

**The one exception to watch:** BUSI normal flips Normal→benign, but by the *opposite* mechanism —
the blur **adds** a mass-like blob rather than removing evidence. So a grade change is not
automatically "faithful"; you have to ask whether obscuration *removed* a feature or *created* one.

---

## 5. The clearest unfaithful case

**DR grade 1, `22360_left.jpeg`** (top concept *microaneurysms*):

```
c:   0.112 → 0.416     drop = −0.305     L_drop = 0.305   (the loop's penalty fires)
reg: 0.997 → 0.425
```

Blurring the region the heatmap claims holds the microaneurysms made the image score **higher** on
microaneurysms — the concept got *stronger* when its own evidence was removed. That is the textbook
"unfaithful" signal: the WHERE map and the WHY score actively disagree.

---

## 6. Three regimes — and why the bare drop isn't the verdict

Sorting the 8 groups by behavior:

| regime | groups | what's happening |
|---|---|---|
| **genuinely responsive** | DR moderate (hard exudates), BUSI benign (1 case) | removing the lesion drops WHY *and* grade |
| **drop, but suspect** | BUSI normal | drop is real but the blur *manufactures* a mass |
| **inert** | BUSI malignant, DR No-DR / Severe / PDR | diffuse / ungrounded heatmap → nothing removable |
| **anti-faithful** | DR mild (22360) | score rises when evidence removed |

A drop alone can't separate "hit the real evidence" from "blur anywhere perturbs the embedding" from
"the mask covers half the image." That is precisely what the **step-3 specificity control** (on- vs
off-target blur) is for. Per [`OBSCURATION.md`](OBSCURATION.md) §4, that control finds only ~2/8
groups (**BUSI benign, DR moderate**) are cleanly *location*-specific — the rest are generic
blur-sensitivity or big-mask artifacts. **The decisive signal is `drop_on − drop_off`, not the drop
alone.** Run it with [`demo_faith_control.py`](demo_faith_control.py).

---

## 7. Caveats (read before quoting any number)

- **Diffuse heatmaps.** The concept maps cover ~18–42% of the image. Big masks both (a) make the blur
  *create* structure (BUSI normal) and (b) make the off-target control overlap the original region,
  understating specificity. Sharper maps (`--threshold`, or the proposed auxiliary segmentation loss)
  would separate on/off cleanly.
- **n = 5 per group.** These are directional, not final; group "means" are easily swung by one case
  (BUSI benign).
- **Encoder.** Numbers are only meaningful with `--encoder biomedclip` + the recovered text
  projections; the random encoder makes the drop meaningless.
- **DR negative cosines** are expected for severe/PDR; the check is sign-agnostic (uses the *change*),
  so a flat negative cosine correctly reads as "inert," not "faithful."
- **Execution.** Regenerate via SLURM (`sbatch` / `srun`), never on a login node — see the policy in
  [`CLAUDE.md`](CLAUDE.md).

---

## 8. Representative figures

Each case below has a combined 4-panel figure plus standalone `original` / `occluded` / `where`
panels under [`faithful_figures/`](faithful_figures/):

- **Responsive (DR moderate, grade collapse):** `dr_grade2_idx20446` —
  [original](faithful_figures/panels/dr_grade2_idx20446_original.png) ·
  [occluded](faithful_figures/panels/dr_grade2_idx20446_occluded.png) ·
  [where](faithful_figures/panels/dr_grade2_idx20446_where.png)
- **Suspect drop (BUSI normal → benign, blur makes a blob):** `busi_normal_idx3` —
  [original](faithful_figures/panels/busi_normal_idx3_original.png) ·
  [occluded](faithful_figures/panels/busi_normal_idx3_occluded.png) ·
  [where](faithful_figures/panels/busi_normal_idx3_where.png)
- **Anti-faithful (microaneurysms score rises):** `dr_grade1_idx17754` —
  [original](faithful_figures/panels/dr_grade1_idx17754_original.png) ·
  [occluded](faithful_figures/panels/dr_grade1_idx17754_occluded.png) ·
  [where](faithful_figures/panels/dr_grade1_idx17754_where.png)
- **Inert (PDR, nothing moves):** `dr_grade4_idx34179` —
  [original](faithful_figures/panels/dr_grade4_idx34179_original.png) ·
  [occluded](faithful_figures/panels/dr_grade4_idx34179_occluded.png) ·
  [where](faithful_figures/panels/dr_grade4_idx34179_where.png)

---

### Bottom line

Obscuring the heatmap **really does throw off the model — but only in ~2 of 8 groups cleanly** (DR
moderate, BUSI benign), where both the concept score and the predicted grade collapse together.
Elsewhere the drop is a blur artifact (BUSI normal), generic sensitivity, simply absent (malignant,
No-DR, severe, PDR), or — in one case — backwards (microaneurysms). The honest verdict is the
co-movement of WHY-score and grade under a *location-specific* occlusion, not the raw drop.
