### Step 1: image-level POD

| Task | Image-POD alone, best r (model) | v5 best | v5 best + image-POD r=50 | Paired Δ [95% CI], one-sided p |
|---|---|---|---|---|
| OCTID CSR/DR/MH (500x750) | 0.696 [0.646, 0.755] (r=100, LR) | 0.872 [0.833, 0.911] | 0.881 [0.855, 0.924] | +0.015 [-0.010, +0.041], p = 0.141 |
| OCTDL 7-class | 0.439 [0.377, 0.479] (r=25, LR) | 0.669 [0.605, 0.723] | 0.664 [0.606, 0.720] | -0.002 [-0.015, +0.010], p = 0.637 |
| OCTDL 6 diseases | 0.423 [0.357, 0.469] (r=10, LR) | 0.653 [0.589, 0.722] | 0.634 [0.558, 0.690] | -0.029 [-0.061, -0.005], p = 0.992 |
| Kermany 4-class | 0.537 [0.509, 0.558] (r=50, LightGBM) | 0.695 [0.671, 0.718] | 0.698 [0.678, 0.724] | +0.006 [-0.004, +0.015], p = 0.123 |
| Kermany CNV/DME/DRUSEN | 0.611 [0.578, 0.635] (r=25, LightGBM) | 0.731 [0.706, 0.765] | 0.731 [0.708, 0.765] | +0.001 [-0.013, +0.015], p = 0.456 |
| Kermany -> OCTDL (NORMAL/DME/AMD) | 0.546 [0.508, 0.584] (r=200, LR) | 0.709 [0.673, 0.743] | 0.693 [0.657, 0.730] | -0.017 [-0.036, +0.002], p = 0.963 |

*Source: `results/v6/image_pod_<task>.json`.*

### Step 2: POD residual anomaly score

| NORMAL vs ABNORMAL | AUROC r=10 | AUROC r=25 [95% CI] | AUROC r=50 |
|---|---|---|---|
| OCTDL 7-class | 0.665 | 0.696 [0.659, 0.731] | 0.717 |
| Kermany 4-class | 0.660 | 0.700 [0.673, 0.732] | 0.720 |
| Kermany healthy -> OCTDL (NORMAL vs ABNORMAL) | 0.659 | 0.691 [0.656, 0.724] | 0.708 |

*Source: `results/v6/residual_auroc.json`.*

| Task | v5 best + e(x) | Paired Δ vs v5 best |
|---|---|---|
| OCTID CSR/DR/MH (500x750) | 0.875 [0.846, 0.920] | +0.011 [+0.000, +0.027], p = 0.049 |
| OCTDL 7-class | 0.666 [0.607, 0.724] | +0.000 [-0.007, +0.008], p = 0.476 |
| OCTDL 6 diseases | 0.649 [0.584, 0.716] | -0.005 [-0.014, +0.004], p = 0.859 |
| Kermany 4-class | 0.699 [0.673, 0.721] | +0.003 [-0.004, +0.010], p = 0.202 |
| Kermany CNV/DME/DRUSEN | 0.729 [0.704, 0.762] | -0.003 [-0.010, +0.005], p = 0.746 |
| Kermany -> OCTDL (NORMAL/DME/AMD) | 0.704 [0.667, 0.739] | -0.005 [-0.018, +0.006], p = 0.821 |

*Source: `results/v6/residual_clf.json`.*

### Step 3: hybrid (frozen deep features + physics) vs fine-tuned CNN

| Task | Hybrid DINOv2-S | Hybrid ResNet-18 | DINOv2-S only | ResNet-18 only | CNN (fine-tuned) | Hybrid DINOv2-S − CNN |
|---|---|---|---|---|---|---|
| OCTID CSR/DR/MH (500x750) | 0.899 [0.864, 0.933] | 0.904 [0.865, 0.932] | 0.865 [0.831, 0.908] | 0.908 [0.884, 0.948] | 0.935 [0.905, 0.961] | -0.036 [-0.078, +0.004], p = 0.965 |
| OCTDL 7-class | 0.688 [0.632, 0.748] | 0.708 [0.657, 0.764] | 0.600 [0.541, 0.649] | 0.671 [0.603, 0.709] | n/a | n/a |
| OCTDL 6 diseases | 0.669 [0.602, 0.736] | 0.682 [0.616, 0.741] | 0.581 [0.520, 0.655] | 0.649 [0.585, 0.704] | n/a | n/a |
| Kermany 4-class | 0.734 [0.715, 0.759] | 0.730 [0.704, 0.750] | 0.683 [0.668, 0.711] | 0.695 [0.670, 0.715] | 0.867 [0.851, 0.885] | -0.130 [-0.152, -0.108], p = 1.000 |
| Kermany CNV/DME/DRUSEN | 0.763 [0.740, 0.791] | 0.759 [0.735, 0.789] | 0.722 [0.691, 0.742] | 0.718 [0.701, 0.751] | n/a | n/a |
| Kermany -> OCTDL (NORMAL/DME/AMD) | 0.729 [0.694, 0.766] | 0.753 [0.719, 0.789] | 0.683 [0.647, 0.717] | 0.737 [0.702, 0.770] | 0.925 [0.908, 0.940] | -0.197 [-0.232, -0.159], p = 1.000 |

Grouped permutation importance of the DINOv2-S hybrid on OCTID CSR/DR/MH (drop in balanced accuracy):

| Rank | Group | Importance |
|---|---|---|
| 1 | physics: NDT (pulse-echo) | 0.0647 |
| 2 | DINOv2-S embedding (PCA 64) | 0.0563 |
| 3 | physics: disease-specific (thickness, fluid, gap, RPE) | 0.0398 |
| 4 | physics: texture: rpe band | 0.0347 |
| 5 | physics: texture: outer band | 0.0217 |
| 6 | physics: texture: inner band | 0.0197 |
| 7 | image-POD (50 modes) | 0.0111 |
| 8 | physics: texture: choroid band | 0.0087 |
| 9 | physics: boundary fractal | 0.0069 |
| 10 | POD residual e(x) | 0.0000 |

*Source: `results/v6/hybrid_<task>.json`, `cnn_<task>.json`, `hybrid_importance.json`.*

### Step 4: Grad-CAM vs physics band importance (OCTID CSR/DR/MH)

| Band | CAM share CSR | CAM share DR | CAM share MH | physics recall drop CSR | physics recall drop DR | physics recall drop MH |
|---|---|---|---|---|---|---|
| inner | 0.120 | 0.122 | 0.137 | 0.116 | 0.029 | -0.010 |
| outer | 0.148 | 0.169 | 0.168 | 0.020 | 0.121 | 0.008 |
| rpe | 0.081 | 0.082 | 0.081 | 0.104 | 0.072 | 0.010 |
| choroid | 0.251 | 0.275 | 0.233 | 0.016 | 0.066 | -0.002 |

Spearman ρ over the 12 band × class cells: -0.22 (p = 0.48). *Source: `results/v6/gradcam_octid_dis750.json`.*

### Step 5: cross-scanner robustness (Kermany → OCTDL)

| Model | Calibration | TTA | Balanced acc | Recall NORMAL / DME / AMD |
|---|---|---|---|---|
| v5 best (soft vote) | none | no TTA | 0.709 | 0.79 / 0.55 / 0.78 |
| v5 best (soft vote) | none | TTA | 0.670 | 0.66 / 0.50 / 0.85 |
| v5 best (soft vote) | healthy-ref k=30 | no TTA | 0.682 ± 0.015 | 0.72 / 0.75 / 0.57 |
| v5 best (soft vote) | healthy-ref k=30 | TTA | 0.705 ± 0.018 | 0.75 / 0.74 / 0.62 |
| v6 hybrid (DINOv2-S + physics) | none | no TTA | 0.729 | 0.81 / 0.58 / 0.79 |
| v6 hybrid (DINOv2-S + physics) | none | TTA | 0.729 | 0.73 / 0.63 / 0.83 |
| v6 hybrid (DINOv2-S + physics) | healthy-ref k=30 | no TTA | 0.701 ± 0.015 | 0.77 / 0.70 / 0.63 |
| v6 hybrid (DINOv2-S + physics) | healthy-ref k=30 | TTA | 0.719 ± 0.015 | 0.77 / 0.73 / 0.66 |

Paired tests: v6 hybrid (DINOv2-S + physics) | none | no TTA vs v5 best (soft vote) | none | no TTA: +0.020 [-0.004, +0.046], p = 0.061; v5 best (soft vote) | none | TTA vs no TTA: -0.039 [-0.060, -0.021], p = 1.000; v6 hybrid (DINOv2-S + physics) | none | TTA vs no TTA: +0.000 [-0.017, +0.019], p = 0.501; v6 hybrid (DINOv2-S + physics) | none | no TTA vs CNN: -0.197 [-0.232, -0.159], p = 1.000

*Source: `results/v6/cross.json`.*

### Step 6: conformal triage on the v6 hybrid

| Task | Target | Coverage per class | Mean set size (v5) | Singleton rate (v5) | Singleton accuracy (v5) |
|---|---|---|---|---|---|
| OCTDL triage NORMAL vs ABNORMAL | 0.95 | ABNORMAL 0.951, NORMAL 0.961 | 1.06 (1.08) | 0.94 (0.92) | 0.951 (0.952) |
| OCTDL triage NORMAL vs ABNORMAL | 0.90 | ABNORMAL 0.901, NORMAL 0.914 | 0.97 (0.98) | 0.96 (0.96) | 0.939 (0.931) |
| Kermany triage NORMAL vs ABNORMAL | 0.95 | ABNORMAL 0.942, NORMAL 0.938 | 1.18 (1.27) | 0.82 (0.73) | 0.928 (0.921) |
| Kermany triage NORMAL vs ABNORMAL | 0.90 | ABNORMAL 0.891, NORMAL 0.889 | 1.04 (1.09) | 0.95 (0.91) | 0.893 (0.878) |
| OCTID CSR/DR/MH (500x750) | 0.95 | CSR 1.000, DR 0.961, MH 0.931 | 2.03 (2.10) | 0.19 (0.18) | 0.934 (0.948) |
| OCTID CSR/DR/MH (500x750) | 0.90 | CSR 0.926, DR 0.936, MH 0.863 | 1.11 (1.25) | 0.86 (0.75) | 0.913 (0.910) |
| Kermany CNV/DME/DRUSEN | 0.95 | CNV 0.947, DME 0.956, DRUSEN 0.947 | 1.75 (1.82) | 0.39 (0.33) | 0.908 (0.899) |
| Kermany CNV/DME/DRUSEN | 0.90 | CNV 0.905, DME 0.900, DRUSEN 0.900 | 1.47 (1.53) | 0.57 (0.52) | 0.859 (0.841) |

*Source: `results/v6/conformal.json` (v5 values copied from `results/v5/conformal.json`).*
