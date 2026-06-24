# Faithfulness verdict — scaled, controlled, statistically tested

- **Images**: 800 total  |  up to **100 per group**, 8 groups  |  seed 0
- **Occlusion**: blur (σ=21.0), 6 off-target controls/image  |  encoder: biomedclip
- **Tests**: paired one-sided Wilcoxon; specificity p-values BH-FDR corrected across 8 groups; α=0.05.

Verdict rule: **FAITHFUL** = specificity significant & median > 0.02; **GENERIC** = drop real but specificity not (mask>0.35 ⇒ big-mask); **INERT** = |median drop|<0.02 & drop n.s.; **ANTI-FAITHFUL** = median drop_on<0.

| group | n | drop_on med [IQR] | drop_off | specificity med (95% CI) | p_drop | p_spec(FDR) | mask% | grade-Δ% | verdict |
|---|--:|---|--:|---|--:|--:|--:|--:|---|
| BUSI:normal | 100 | +0.475 [+0.26,+0.57] | +0.081 | +0.332 (+0.25,+0.38) | <.001 | <.001 | 40% | 74% | **FAITHFUL** |
| BUSI:benign | 100 | +0.007 [+0.00,+0.05] | +0.000 | +0.006 (+0.00,+0.01) | <.001 | <.001 | 22% | 25% | **WEAK/INCONCLUSIVE** |
| BUSI:malignant | 100 | +0.001 [+0.00,+0.03] | +0.000 | +0.000 (+0.00,+0.00) | <.001 | <.001 | 30% | 22% | **WEAK/INCONCLUSIVE** |
| DR:grade0 (No DR) | 100 | -0.000 [-0.00,+0.00] | -0.000 | -0.000 (-0.00,+0.00) | 0.760 | 0.651 | 28% | 2% | **INERT** |
| DR:grade1 (Mild NPDR) | 100 | -0.052 [-0.37,-0.01] | -0.012 | -0.034 (-0.12,-0.01) | 1.000 | 1.000 | 18% | 42% | **ANTI-FAITHFUL** |
| DR:grade2 (Moderate NPDR) | 100 | -0.001 [-0.00,+0.24] | -0.001 | -0.000 (-0.00,+0.04) | 0.003 | <.001 | 22% | 40% | **WEAK/INCONCLUSIVE** |
| DR:grade3 (Severe NPDR) | 100 | -0.001 [-0.00,+0.00] | -0.001 | +0.001 (+0.00,+0.00) | 1.000 | <.001 | 17% | 8% | **INERT** |
| DR:grade4 (PDR) | 100 | -0.000 [-0.00,+0.00] | -0.000 | +0.000 (-0.00,+0.00) | 0.647 | 0.294 | 8% | 1% | **INERT** |

**Bottom line.** 1 of 8 groups are statistically location-specific (FAITHFUL). The decisive column is `specificity med (95% CI)` with a significant `p_spec(FDR)` — not the bare `drop_on`. `grade-Δ%` is the fraction of images whose predicted grade changed under the on-target occlusion (the WHY↔grade coupling).

**Caveats.** Numbers require `--encoder biomedclip` + recovered bridges. Diffuse heatmaps (high mask%) let the off-target control overlap the real region, *understating* specificity — sharpen with `--threshold` or the proposed seg loss. DR severe/PDR cosines are negative; the test is sign-agnostic (uses the change).
