"""
5-Minute Diagnosis: Check if ordinal loss is broken on DR
Compare BUSI vs DR loss component magnitudes at epoch 24 (before curriculum)
"""

import pandas as pd

# Data from actual logs
data = {
    'Dataset': ['BUSI', 'DR'],
    'Epoch': [24, 24],
    'pcol': [0.548, 2.790],
    'scolw': [3.941, 4.404],
    'rmse': [0.109, 0.264],
    'it': [0.254, 1.632],
    'pic': [0.673, 0.634],
    'cons': [0.086, 0.043],
    'val_acc': [87.26, 74.53]
}

df = pd.DataFrame(data)

print("="*80)
print("LOSS COMPONENT ANALYSIS (Epoch 24 - Before Curriculum)")
print("="*80)
print()
print(df.to_string(index=False))
print()

# Calculate ratios
print("="*80)
print("DR / BUSI RATIOS (should all be ~1.0 if balanced)")
print("="*80)
print()

components = ['pcol', 'scolw', 'rmse', 'it', 'pic', 'cons']
for comp in components:
    ratio = df.loc[df['Dataset']=='DR', comp].values[0] / df.loc[df['Dataset']=='BUSI', comp].values[0]
    status = "⚠️ PROBLEM" if ratio > 2.0 else "✅ OK"
    print(f"  {comp:8s}: {ratio:.2f}x   {status}")

print()
print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

it_ratio = df.loc[df['Dataset']=='DR', 'it'].values[0] / df.loc[df['Dataset']=='BUSI', 'it'].values[0]
pcol_ratio = df.loc[df['Dataset']=='DR', 'pcol'].values[0] / df.loc[df['Dataset']=='BUSI', 'pcol'].values[0]

print(f"Ordinal Threshold Loss (it): {it_ratio:.1f}x higher on DR")
print(f"Concept Loss (pcol): {pcol_ratio:.1f}x higher on DR")
print()

if it_ratio > 5.0:
    print("🔴 CRITICAL: Ordinal thresholds are completely misaligned on 5-class DR")
    print("   The spine was designed for 3-class BUSI, not 5-class DR")
    print("   Solution: Reduce eta (ordinal weight) or redesign ordinal component")
    print()
    print("ACTION: Try eta=0.05 instead of eta=0.1 for DR training")
else:
    print("✅ Ordinal loss is reasonable")

if pcol_ratio > 3.0:
    print("🔴 Concept learning is 3x harder on DR")
    print("   Either concepts are misaligned or text encoder needs better training")
else:
    print("✅ Concept learning is similar")

print()
print("="*80)
print("NEXT STEP: Edit configs/config.py line with DRConfig.eta")
print("="*80)
print("""
Current:
    eta: float = 0.1

Change to:
    eta: float = 0.05

Then run:
python train_dr.py --dr_root Datasets/DR --folds 0 --use_concept_spine --epochs 20 --run_dir runs/dr_debug_eta_reduce

Check if val_acc improves in first 5 epochs compared to Simon's 74.53%
""")