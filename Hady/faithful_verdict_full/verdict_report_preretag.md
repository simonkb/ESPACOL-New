# Faithfulness verdict — full dataset, train vs held-out

- **Images**: 35906 total  |  all per group, 8 groups  |  seed 0
- **Occlusion**: blur (σ=21.0), 6 off-target controls/image  |  encoder: biomedclip
- **Tests**: paired one-sided Wilcoxon; specificity p-values BH-FDR corrected within each partition; α=0.05.
- **Partitions**: each image is tagged by whether it was in the checkpoint's k-fold TRAINING set or its HELD-OUT fold (BUSI fold 4, DR fold 2). The held-out table is the data-leak-free verdict; train vs held-out measures in-sample (memorisation) inflation.

Verdict rule: **FAITHFUL** = specificity significant & median > 0.02; **GENERIC** = drop real but specificity not (mask>0.35 ⇒ big-mask); **INERT** = |median drop|<0.02 & drop n.s.; **ANTI-FAITHFUL** = median drop_on<0.


## Overall (all images)

| group | n | drop_on med [IQR] | drop_off | specificity med (95% CI) | p_drop | p_spec(FDR) | mask% | grade-Δ% | verdict |
|---|--:|---|--:|---|--:|--:|--:|--:|---|
| BUSI:normal | 133 | +0.475 [+0.21,+0.54] | +0.075 | +0.338 (+0.26,+0.37) | <.001 | <.001 | 40% | 71% | **FAITHFUL** |
| BUSI:benign | 437 | +0.008 [+0.00,+0.10] | +0.000 | +0.007 (+0.01,+0.01) | <.001 | <.001 | 23% | 26% | **WEAK/INCONCLUSIVE** |
| BUSI:malignant | 210 | +0.001 [+0.00,+0.00] | +0.000 | +0.000 (+0.00,+0.00) | <.001 | <.001 | 30% | 16% | **WEAK/INCONCLUSIVE** |
| DR:grade0 (No DR) | 25810 | -0.000 [-0.00,+0.00] | -0.000 | -0.000 (-0.00,-0.00) | 1.000 | 1.000 | 27% | 2% | **INERT** |
| DR:grade1 (Mild NPDR) | 2443 | -0.067 [-0.37,-0.01] | -0.010 | -0.040 (-0.06,-0.03) | 1.000 | 1.000 | 17% | 44% | **ANTI-FAITHFUL** |
| DR:grade2 (Moderate NPDR) | 5292 | -0.001 [-0.00,+0.24] | -0.001 | -0.000 (-0.00,+0.00) | <.001 | <.001 | 23% | 40% | **WEAK/INCONCLUSIVE** |
| DR:grade3 (Severe NPDR) | 873 | -0.000 [-0.00,+0.00] | -0.001 | +0.000 (+0.00,+0.00) | 1.000 | <.001 | 16% | 4% | **INERT** |
| DR:grade4 (PDR) | 708 | -0.000 [-0.00,+0.00] | -0.000 | +0.000 (-0.00,+0.00) | 0.900 | 0.437 | 8% | 1% | **INERT** |

## Train-only (in-sample)

| group | n | drop_on med [IQR] | drop_off | specificity med (95% CI) | p_drop | p_spec(FDR) | mask% | grade-Δ% | verdict |
|---|--:|---|--:|---|--:|--:|--:|--:|---|
| BUSI:normal | 107 | +0.473 [+0.03,+0.54] | +0.073 | +0.298 (+0.24,+0.36) | <.001 | <.001 | 40% | 69% | **FAITHFUL** |
| BUSI:benign | 350 | +0.007 [+0.00,+0.10] | +0.000 | +0.006 (+0.00,+0.01) | <.001 | <.001 | 23% | 25% | **WEAK/INCONCLUSIVE** |
| BUSI:malignant | 168 | +0.001 [+0.00,+0.01] | +0.000 | +0.000 (+0.00,+0.00) | <.001 | <.001 | 29% | 18% | **WEAK/INCONCLUSIVE** |
| DR:grade0 (No DR) | 23229 | -0.000 [-0.00,+0.00] | -0.000 | -0.000 (-0.00,-0.00) | 1.000 | 1.000 | 28% | 1% | **INERT** |
| DR:grade1 (Mild NPDR) | 2199 | -0.100 [-0.37,-0.01] | -0.012 | -0.075 (-0.11,-0.06) | 1.000 | 1.000 | 17% | 47% | **ANTI-FAITHFUL** |
| DR:grade2 (Moderate NPDR) | 4767 | -0.001 [-0.00,+0.24] | -0.001 | -0.000 (-0.00,+0.00) | <.001 | <.001 | 23% | 41% | **WEAK/INCONCLUSIVE** |
| DR:grade3 (Severe NPDR) | 786 | -0.000 [-0.00,+0.00] | -0.001 | +0.000 (+0.00,+0.00) | 1.000 | <.001 | 15% | 1% | **INERT** |
| DR:grade4 (PDR) | 629 | -0.000 [-0.00,+0.00] | -0.000 | +0.000 (-0.00,+0.00) | 0.887 | 0.351 | 8% | 0% | **INERT** |

## Held-out only (no data leak)

| group | n | drop_on med [IQR] | drop_off | specificity med (95% CI) | p_drop | p_spec(FDR) | mask% | grade-Δ% | verdict |
|---|--:|---|--:|---|--:|--:|--:|--:|---|
| BUSI:normal | 26 | +0.477 [+0.42,+0.82] | +0.083 | +0.369 (+0.21,+0.43) | <.001 | <.001 | 41% | 81% | **FAITHFUL** |
| BUSI:benign | 87 | +0.013 [+0.00,+0.20] | +0.001 | +0.012 (+0.01,+0.02) | <.001 | <.001 | 23% | 29% | **WEAK/INCONCLUSIVE** |
| BUSI:malignant | 42 | +0.001 [+0.00,+0.00] | +0.000 | +0.000 (-0.00,+0.00) | <.001 | 0.012 | 30% | 10% | **WEAK/INCONCLUSIVE** |
| DR:grade0 (No DR) | 2581 | -0.000 [-0.00,+0.00] | -0.000 | -0.000 (-0.00,-0.00) | 1.000 | 1.000 | 27% | 5% | **INERT** |
| DR:grade1 (Mild NPDR) | 244 | -0.000 [-0.00,+0.00] | -0.000 | -0.000 (-0.00,+0.00) | 0.985 | 1.000 | 25% | 22% | **INERT** |
| DR:grade2 (Moderate NPDR) | 525 | -0.000 [-0.00,+0.24] | -0.001 | -0.000 (-0.00,+0.00) | <.001 | 0.008 | 22% | 34% | **WEAK/INCONCLUSIVE** |
| DR:grade3 (Severe NPDR) | 87 | -0.001 [-0.00,+0.11] | -0.001 | +0.001 (-0.00,+0.01) | 0.690 | 0.008 | 21% | 33% | **INERT** |
| DR:grade4 (PDR) | 79 | -0.000 [-0.00,+0.00] | +0.000 | +0.000 (-0.00,+0.00) | 0.504 | 0.700 | 11% | 11% | **INERT** |

**Bottom line.** FAITHFUL groups — overall: 1, train-only: 1, held-out-only: 1 (of 8). **If held-out ≈ train, running the verdict on training images did NOT inflate faithfulness** (the data-leak concern is empirically dead); a drop from train to held-out quantifies in-sample/memorisation inflation. The decisive column is `specificity med (95% CI)` with a significant `p_spec(FDR)` — not the bare `drop_on`. `grade-Δ%` is the fraction of images whose predicted grade changed under the on-target occlusion (the WHY↔grade coupling).

**Caveats.** Numbers require `--encoder biomedclip` + recovered bridges. Diffuse heatmaps (high mask%) let the off-target control overlap the real region, *understating* specificity — sharpen with `--threshold` or the proposed seg loss. DR severe/PDR cosines are negative; the test is sign-agnostic (uses the change).
