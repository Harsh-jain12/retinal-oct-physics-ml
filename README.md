# Multi-Parametric Retinal Diagnostics: reproduction

A from-scratch reproduction of *"Multi-Parametric Retinal Diagnostics: A Deterministic Physics-Based Engine for Tomographic Classification"* (Banerjee, Sarkar, Munshi, IIT Gandhinagar). The pipeline:

- OCTID dataset, 517 scans
- artifact-aware 750×500 normalization and a 15×10 grid of 50×50 cells
- per-grid KT-1 slope (S) and NMSID Hurst coefficient (H), giving 300 features
- class-balanced Random Forest on a stratified 80/20 split

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
python download_data.py           # OCTID from Kaggle -> data/OCTID (no login needed)
```

## Run

| Step | Command | Output |
|---|---|---|
| Feature extraction (about 3.5 min) | `python extract_features.py` | `results/features.csv` |
| Paper results and Figs 3–7 | `python train_eval.py` | `results/fig*.png`, `results/results.json` |
| Robustness and confound checks | `python sanity_checks.py` | `results/sanity_checks.json` |
| Resampling-harmonized variant | `python harmonized_experiment.py` | console |

## Code layout

- `retina/preprocess.py`: Sec. 2. White-wedge mask (I ≥ 240), tight bbox crop, symmetric 3:2 crop, cubic resize to 750×500, residual white set to black, 150 micro-grids (grid id = row·15 + col).
- `retina/features.py`: Sec. 3.
  - KT-1: Gaussian σ ∈ {3, 2, 1, 0.1}, robust 99th-percentile N_max, slope of 1/N_max against |W''(0)| ∝ σ².
  - Hurst: NMSID over all pixel pairs at ⌊distance⌋ = k, for k = 1…25, then the log-log slope.
- RF settings: 300 trees, max_depth 12, min_samples_leaf 2, `class_weight="balanced"`, `random_state=42`.

### Choices the paper leaves unspecified

| Item | Choice made here |
|---|---|
| Intensity scale | [0, 1] |
| W''(0) | Proportional to σ² (Gaussian window) |
| NMSID scale range | k = 1…25 (half a grid) |
| Black background grids | N_max floored at 0.05; H = 0 when NMSID is 0 |
| Phase-space plot (Fig. 4) | Per-scan mean H against mean S |

## Results

### Paper vs. this reproduction

| Metric | Paper | This repo |
|---|---|---|
| Test set | 104 (CSR 21 / DR 22 / MH 20 / N 41) | identical |
| NORMAL precision / recall | 1.00 / 1.00 | **1.00 / 1.00** (41/41) |
| Overall accuracy (80/20 split) | 71% | **68.3%** |
| Overall accuracy (5-fold × 10 CV) | n/a | **72.9 ± 3.4%** (range 65–79%) |
| Top-15 features that are Hurst | 15/15 | **15/15** |
| Location of top-15 grids | rows 7–8 (grids 106–133) | **rows 7–8** (grids 106–133) |

The paper's numbers reproduce.

### Caveat: the 100% NORMAL triage is a dataset confound

In OCTID, the NORMAL scans and the disease scans were acquired on different scanners/protocols.

| Original size | CSR | DR | MH | NORMAL |
|---|---|---|---|---|
| 500×750 | 68 | 102 | 102 | 0 |
| 586×879 | 34 | 5 | 0 | 0 |
| 496×512, 512×512, 496×768, 496×1024, 496×1536 | 0 | 0 | 0 | 206 |

NORMAL files are also all `.jpg`, while disease files are all `.jpeg`.

What `sanity_checks.py` and `harmonized_experiment.py` found:

1. **Image size alone** separates NORMAL from disease with **100%** accuracy.
2. **Resampling drives H.** Sec. 2.1 cubic-upsamples NORMAL scans by about 1.32× on average, and disease scans by about 1.0×. Upsampling smooths texture and raises H.
   - Spearman correlation between upsampling factor and mean H: **0.82**.
3. **Resampling flips the model.** Shrinking the 63 held-out disease scans to 512 px wide and running the identical pipeline makes the trained model call **63/63 of them NORMAL** (before: 0/63).
4. **The physics features add nothing over plain statistics.** A baseline using only each grid's mean and standard deviation of intensity scores **77.7%** CV accuracy, with NORMAL P/R of 0.98/0.99.
5. **Harmonizing resampling isn't enough.** Giving every scan the same resampling still leaves NORMAL 100% separable, because of other device signatures (speckle, field of view, compression).

So OCTID cannot validate healthy-vs-disease triage. The defensible signal is the disease-only 3-class problem (CSR / DR / MH), which reaches **56.9%** CV accuracy (chance is 33%).

## Part 2: improved pipeline (v2)

### What v2 adds

**Preprocessing** (`retina/layers.py`):
- Physical axial resampling to 3.87 µm/px when the scanner spacing is known.
- Overlay masking applied to every image of a dataset regardless of class, so the mask can't leak the label.
- Classical RPE/ILM tracing and **retina flattening**.
- Anatomical bands (inner retina, outer retina, RPE, choroid) × fovea-centred sectors.

**Features** (`retina/features_v2.py`):
- **Per band × sector texture:** Hurst H plus its fit quality, NMSID curve shape, lacunarity, entropy, KT-1, and **structure-tensor coherence**.
- **Disease-specific:** thickness profile and foveal dip, gap width (macular hole), fluid area (vitreous-relative threshold, opened mask), hyper-reflective foci, RPE elevation, left/right asymmetry, segmentation QC.

**Evaluation** (`retina/evaluation.py`):
- Patient-grouped repeated stratified CV with patient-level bootstrap 95% CIs.
- Models: paper RF, LightGBM, logistic regression, two-stage (normal vs. abnormal, then which disease).
- Grouped permutation importance.

### Run (v2)

```bash
python make_manifest.py octid
python extract_v2.py manifests/octid.csv          # ~4 min; paper + entropy + v2 features
python evaluate_v2.py results/v2/octid_features.csv
python analysis_v2.py results/v2/octid_features.csv
python controls_octid.py
```

### OCTID results

Patient-grouped 5-fold CV × 3 repeats. Full table: `results/v2/octid_model_grid.md`.

| Task (balanced accuracy, 95% CI) | Paper features + paper RF | Best v2 |
|---|---|---|
| 4-class (scanner-confounded) | 0.667 [0.62, 0.70] | **0.848** [0.80, 0.86] (band summary + disease, LightGBM) |
| CSR / DR / MH | 0.554 [0.52, 0.63] | **0.832** [0.79, 0.87] (band summary + disease, logistic regression) |
| CSR / DR / MH, 500×750 scans only | 0.504 [0.48, 0.59] | **0.836** [0.76, 0.85] (band summary + disease, logistic regression) |

The last row is the cleanest test: every image has the same original size. On it, interpretable layer-wise features with a *linear* model reach 0.84, against 0.50 for the paper's grid features (chance is 0.33).

### What the controls showed (`controls_octid.py`, `analysis_v2.py`)

**Location claim ("the model found the choroid"):**
- **The RPE isn't in grid rows 7–8.** In ~95% of scans in every class it lies in grid rows 3–6 (`octid_rpe_position.png`). Rows 7–8 are deep choroid, sclera and the noise floor below it.
- **Row 8 mostly detects the scanner.** Its importance is +0.198 balanced accuracy with NORMAL included, but only +0.069 among the diseases.
- **Grid location adds nothing.** Feeding the same values sorted, with no positions, scores as well or better (4-class 0.761, disease-only 0.614).

**Overlay icon:**
- Disease scans carry a scan-direction icon in the lower-left corner, which falls in grid rows 7–9, columns 0–2.
- Three paper top-15 grids sit there. Masking the corner in all images removes them from the top 15 and lowers disease-only accuracy from 0.553 to 0.504.

**Entropy:**
- Hurst–entropy correlation is moderate (median |ρ| 0.50 across paper grids, 0.35 across v2 regions).
- H's real redundancy is with intensity std (VIF 24.6) and first-scale roughness (VIF 13).
- Adding entropy back *improves* the paper features: disease-only 0.587 → 0.657 with gradient boosting.

**Importance measures:**
- Gini and held-out permutation importance share only **1 of 15** top features.
- On held-out data only 42 of the 300 paper features carry any signal.

**Most useful feature families for CSR / DR / MH** (grouped permutation importance):
- structure-tensor **coherence** (+0.20)
- the inner-retina band (+0.16)
- the outer-retina band (+0.09)
- the RPE band (+0.08)
- Hurst H (+0.01)

This supports the paper's own future-work direction (anisotropic coherence), not its fractal-roughness claim.

### Single-scanner datasets: OCTDL and Kermany

These datasets have patient IDs, so the CV is truly patient-grouped. Full tables: `results/v2/octdl_model_grid.md`, `results/v2/kermany_model_grid.md`.

| Dataset and task (balanced accuracy, 95% CI) | Paper features (best model) | Best v2 |
|---|---|---|
| OCTDL (Optovue), 7 classes, 2,064 scans / 820 patients | 0.464 | **0.661** [0.60, 0.72] (band summary + disease, logistic regression) |
| OCTDL, 6 diseases only | 0.445 | **0.648** [0.58, 0.71] (same) |
| Kermany (Spectralis), 4 classes, 2,401 scans / 864 patients | 0.479 | **0.683** [0.66, 0.71] (v2 all, two-stage LightGBM) |
| Kermany, 3 diseases only | 0.596 | **0.727** [0.70, 0.75] (band summary + disease, LightGBM) |

On single-scanner data, with no scanner shortcut available, the paper's grid features sit just above chance. The layer-wise v2 features add about 0.13 to 0.20 balanced accuracy.

### Cross-scanner test (`cross_scanner.py`)

Train on Kermany (labels harmonised to NORMAL / DME / AMD; CNV and drusen count as AMD), then test without retraining:
- on OCTDL, a different scanner, using the same three labels (1,710 scans);
- on OCTID as normal vs. abnormal.

| Features | Model | Within Kermany | Kermany → OCTDL | OCTDL healthy scans called healthy |
|---|---|---|---|---|
| Paper (300 grid H+S) | RF (paper config) | 0.535 | 0.374 | 0 / 332 |
| Paper (300 grid H+S) | LightGBM | 0.557 | 0.378 | 2 / 332 |
| v2 band summary + disease | LightGBM | 0.742 | **0.638** | 211 / 332 |
| v2 all | LightGBM | 0.744 | 0.640 | 200 / 332 |

- **Paper features don't transfer.** On a new scanner they label almost every scan AMD and find essentially no healthy eyes.
- **v2 transfers partly.** It loses about 0.10 against within-scanner accuracy but stays well above chance (0.33).
- **OCTID check (normal vs. abnormal):** Kermany-trained v2 gets 0.68–0.72 balanced accuracy, against 0.67–0.68 for paper features. The paper features reach that score by calling nearly everything abnormal (normal recall 0.34).
- **Caveat:** OCTID's diseases (CSR, MH) aren't among Kermany's classes, so this is only a rough check.

### CNN baseline (`cnn_baseline.py`)

Setup:
- ImageNet-pretrained ResNet-18, fine-tuned on CPU at 224×224.
- The same patient-grouped folds as above.
- 6–8 epochs, no tuning.

| Task (balanced accuracy) | Paper features | Best v2 (interpretable) | ResNet-18 |
|---|---|---|---|
| OCTID, CSR / DR / MH, 500×750 only (272 scans) | 0.504 | 0.836 | **0.940** |
| Kermany, 4 classes (patient-grouped 5-fold) | 0.479 | 0.683 | **0.864** |
| Kermany → OCTDL (different scanner, NORMAL / DME / AMD) | 0.378 | 0.640 | **0.932** |

**The CNN wins clearly, including across scanners.**
- An honest framing of the physics approach is *interpretable and training-light*, not *more accurate*.
- The v2 features close about half the gap between the paper features and a CNN.
- **Next step:** use them alongside a CNN, for example as an explanation layer or in a hybrid model, rather than in place of one.

### Reproduce the new-dataset results

C: had only ~15 GB free, and the Kermany Kaggle archive is 11.6 GB. So only a 600-per-class patient subset is extracted from the zip; the archive can then be deleted.

```bash
python download_data.py octdl && python make_manifest.py octdl
python download_data.py kermany            # or, on low disk, after the zip downloads:
python download_data.py kermany-subset ~/.cache/kagglehub/datasets/paultimothymooney/kermany2018/2.archive
python make_manifest.py kermany --per-class 600
python extract_v2.py manifests/octdl.csv && python extract_v2.py manifests/kermany.csv
python evaluate_v2.py results/v2/octdl_features.csv --repeats 2
python evaluate_v2.py results/v2/kermany_features.csv --repeats 2
python cross_scanner.py --train results/v2/kermany_features.csv     --test results/v2/octdl_features.csv --octid results/v2/octid_features.csv --repeats 2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python cnn_baseline.py manifests/octid.csv --task disease-only-750 --folds 5 --epochs 8
python cnn_baseline.py manifests/kermany.csv --external manifests/octdl.csv --harmonise --epochs 6
python cnn_baseline.py manifests/kermany.csv --task all --folds 5 --epochs 6
```

## Part 3: NDT-inspired features

OCT is optical pulse-echo testing:
- each image column is an A-scan;
- the RPE is the "back-wall echo";
- fluid, holes and exudates are "defect echoes".

`retina/features_ndt.py` adds 56 `ndt_*` features on top of v2, computed on v2's flattened scan.

These are display images (log-compressed JPEGs), so every quantity is **relative**, normalised to references inside the same image: the vitreous level and the RPE echo level. Raw reference levels are kept only as `ndtqc_*` QC columns and never enter a model.

| Family | What it measures | NDT analogue |
|---|---|---|
| `att_*`, `trans_*` | Depth slope of the log-display A-scan per band; light reaching the choroid | Attenuation coefficient (Beer–Lambert, dB/mm) |
| `ic_*` | Michelson contrast of the ILM and RPE echoes against neighbouring layers | Reflection coefficient / impedance mismatch |
| `hm_*` | Defect width and fluid-pocket size at half-maximum between the vitreous and local tissue | 6 dB drop flaw sizing |
| `sh_*`, `hypertrans` | Signal loss in the choroid under a column (overall and under bright foci) | Shadowing behind a strong reflector |
| `cnr_*` | Retina vs. vitreous, RPE vs. choroid, fluid pocket vs. surround | SNR / CNR / detectability |
| `spk_*` | Nakagami m and speckle contrast after removing layer structure | Grain-noise statistics |

**Fluid is defined as anechoic.** A pocket must be darker than halfway between the vitreous and the local *darkest tissue* level (25th percentile of the retina column). An earlier median-based reference also flagged the normal outer nuclear layer; the QC figure caught this.

```bash
python extract_ndt.py manifests/octid.csv      # also octdl, kermany -> results/v3/<name>_features.csv
python ndt_checks.py                           # QC overlays + scanner-robustness check
```

### First check: class signal vs. scanner dependence

`results/v3/ndt_scanner_vs_signal.png` compares two quantities per feature family:
- **Class signal:** Kruskal–Wallis ε² across the 4 Kermany classes.
- **Scanner shift:** |Cohen's d| between *healthy* Kermany and *healthy* OCTDL eyes. Same anatomy on a different scanner, so any difference is a scanner signature.

Findings:
- **Half-max sizing is the best NDT family:** high signal and the lowest scanner shift. For example, fluid height has ε² 0.20 and |d| 0.05.
- **Also low-shift:** attenuation and shadowing.
- **Strongest but scanner-dependent:** RPE-vs-choroid contrast, choroid transmission and retina-vs-vitreous CNR (ε² ≈ 0.22 but |d| ≈ 1.5). These are exactly the kind of features that broke the original paper, so the cross-scanner test will decide whether they stay.
- **Paper grid features have both the lowest signal and a moderate scanner shift.**

### Step 2: does NDT add to v2?

`evaluate_v2.py` runs on `results/v3/*_features.csv`; outputs are `results/v3/*_model_grid.md`, `cross_scanner.json`, `ndt_importance.*` and `ndt_paired_test.json`.

**Best balanced accuracy per feature set** (patient-grouped CV, best of RF / LightGBM / logistic regression / two-stage):

| Task | v2 band summary + disease | + NDT | + NDT (scanner-robust) | NDT only |
|---|---|---|---|---|
| OCTID CSR / DR / MH, 500×750 | 0.822 | **0.855** | 0.849 | 0.764 |
| OCTDL 7-class | 0.661 | **0.689** | 0.678 | 0.616 |
| OCTDL 6 diseases | 0.648 | **0.685** | **0.685** | 0.614 |
| Kermany 4-class | 0.683 | 0.687 | **0.688** | 0.620 |
| Kermany CNV / DME / drusen | 0.727 | 0.723 | **0.735** | 0.681 |
| Kermany → OCTDL (cross-scanner) | 0.669 | 0.677 | **0.680** | 0.542 |

**Paired test** (`ndt_paired_test.py`): same model and same folds, with vs. without NDT, over 3 repeats.
- The mean difference is positive in **22 of 24** model × task comparisons. Typical size is **+0.01 to +0.03**.
- The patient-level bootstrap 95% CI excludes zero in 10 of 24.
- The bootstrap is on repeat 0's out-of-fold predictions. There is no multiple-comparison correction.
- Largest consistent gains:
  - OCTID diseases with LightGBM: +0.03 in all 3 repeats.
  - OCTDL 7-class with LightGBM: +0.03, CI [+0.02, +0.06].
  - OCTDL diseases with logistic regression: +0.03.
- **Kermany gains little** (≤ +0.02).

**Grouped permutation importance** (`ndt_importance.png`):
- The **v2 bands** (outer, inner, RPE) remain the strongest families everywhere.
- **NDT attenuation** ranks 3rd on OCTDL (+0.105).
- **NDT interface contrast** ranks 3rd on Kermany (+0.049).
- **Shadowing and SNR/CNR** contribute almost nothing once v2 is present.
- **Half-max sizing** has strong single-feature signal but largely overlaps v2's thickness and fluid features.

**Verdict:** NDT features are a small, mostly consistent addition (+1 to +3 points), not a step change.
- Attenuation and interface contrast carry information v2 lacked.
- Dropping the 8 scanner-dependent NDT features costs nothing and sometimes helps, including across scanners.
- The CNN gap remains: Kermany → OCTDL is 0.68 here vs. 0.93 for ResNet-18.


### Step 3: probability of detection (MAPOD) for the fluid detectors

In ultrasonic NDT, an inspection procedure is qualified by inserting flaws of known size and measuring the probability of detection (POD) against size. The standard method is hit/miss analysis (MIL-HDBK-1823A), which reports **a90**: the flaw size found 90% of the time. `mapod.py` applies the same idea to the two fluid detectors:

- **Lesions:** fully anechoic fluid pockets, filled with a patch of the scan's own vitreous (keeping its speckle and noise floor), are inserted into **60 healthy scans per scanner** (Kermany/Spectralis and OCTDL/Optovue). Three types:
  - an intraretinal cyst, round in µm (assuming a lateral spacing of 6 mm / 512 columns);
  - an intraretinal cyst, round on screen, which is how cysts look in these display images;
  - a subretinal fluid lens on top of the RPE, 8× wider than tall.
- **Sizes:** 14 heights from 8 to 160 µm, 2 random placements each, for about 1,600 insertions per lesion type and scanner.
- **Detectors:** the NDT half-max anechoic rule (`features_ndt.fluid_mask`) vs. v2's fixed-fraction dark rule (`features_v2.v2_fluid_mask`).
- **Hit:** at least 30% of the lesion is newly flagged.
- **False call:** any fluid component flagged on the clean healthy scan.
- **Fit:** a probit on ln(a). a90/95 is the 95th percentile of a90 over a patient bootstrap.

**At default thresholds** (`mapod_pod.png`), in µm (a90/95 in brackets):

| Scanner | Lesion | NDT half-max | v2 fixed-fraction | NDT − v2 |
|---|---|---|---|---|
| Spectralis | cyst, round in µm | 139 (149) | 133 (138) | +6 |
| Spectralis | cyst, round on screen | **56** (62) | 61 (65) | −5 |
| Spectralis | subretinal fluid | **52** (59) | 57 (64) | −5 |
| Optovue | cyst, round in µm | 128 (139) | 123 (128) | +5 |
| Optovue | cyst, round on screen | **45** (47) | 50 (52) | −5 |
| Optovue | subretinal fluid | **54** (60) | 55 (58) | −1 |

Healthy scans with at least one false call: **NDT 30% vs. v2 15%** on Spectralis, **47% vs. 32%** on Optovue.

At default thresholds, NDT finds pockets about 5 µm smaller, but it also calls fluid on more healthy eyes. A detector that calls more will always detect more, so these numbers alone don't show NDT is better.

**At a matched false-call rate** (`mapod_sweep.py` → `mapod_tradeoff.png`): each detector's threshold fraction is swept (6 values each) and a90 is plotted against the false-call rate.

| a90 (µm) at matched false-call rate | 10% | 20% | 30% |
|---|---|---|---|
| Spectralis cyst: NDT / v2 | 77 / 78 | 60 / 60 | 54 / 54 |
| Spectralis SRF: NDT / v2 | **63** / 71 | 52 / 54 | 48 / 49 |
| Optovue cyst: NDT / v2 | 64 / **61** | **55** / 58 | **51** / 55 |
| Optovue SRF: NDT / v2 | **52** / 56 | **43** / 54 | **42** / 52 |

What this shows:
- **On Spectralis the two trade-off curves almost overlap.** The exception is subretinal fluid at the strictest setting (10% false calls: 63 vs. 71 µm). On these clean, high-contrast scans the threshold rule matters much less than where it is set.
- **On Optovue, NDT half-max is better at equal false calls**, by about 4–11 µm for subretinal fluid and up to 4 µm for cysts. Optovue has a different speckle and contrast profile, and the half-max reference adapts to each scan's own darkest tissue, which is the reason the 6 dB rule is used in ultrasonic sizing.
- **The shipped defaults sit at different operating points.** NDT's 0.4 runs at about twice v2's false-call rate. If false calls matter more than sensitivity, NDT at about 0.3 matches v2's false-call rate with similar or better POD.
- **Lateral resolution limits small cysts.** If cysts are round in µm, they are only a few columns wide, and both detectors' lateral smoothing and 5×5 opening remove them: a90 rises to about 130 µm. The real value lies between the "round in µm" and "round on screen" rows, depending on the true lateral spacing.
- **Sizing** (`mapod_sizing.png`): half-max sizes subretinal fluid almost exactly (median measured/true height 0.97–0.98) vs. 0.92 for v2, and screen-round cysts at 0.89–0.91 vs. 0.83–0.85. Cysts that are round in µm are undersized by both (about 0.74).
- **A failure mode of local referencing:** a very large subretinal lens (≥ 150 µm) on Optovue is missed 7% of the time by NDT, because the pocket fills the column and drags down the "darkest tissue" reference. v2 never misses these.
- **Limits of the simulation:**
  - Lesions are painted onto the flattened canvas, so segmentation is taken as given.
  - Tissue is not displaced the way real fluid displaces it.
  - Lesions are fully anechoic; real cysts can contain some reflectivity.
  - Read POD here as a relative comparison of detectors, not a clinical detection limit.

```bash
python mapod.py          # ~3 min: POD curves, sizing, examples   -> results/v3/mapod*.png, mapod.json
python mapod_sweep.py    # ~8 min: threshold sweep, matched FCR   -> results/v3/mapod_tradeoff.png, mapod_sweep.json
```


## Part 4: fractal dimension, validated and applied to retinal "coastlines"

Sources:
- **Athe, Shakya, Munshi (2013)**, *Characterization of multiphase flow in bubble columns using KT-1 signature and fractal dimension*, Flow Meas. Instrum. 33:122–137. This is where the paper's KT-1 + NMSID Hurst method comes from (same group).
- **Classic fractals:** the Sierpiński carpet, Brownian (fBm) fractals, and Richardson's coastline measurement (Britain ≈ 1.25; the Koch curve, often quoted as the coastline example, is exactly 1.262).

**Formulas used** (`retina/fractal.py`):

| Quantity | Relation | Used for |
|---|---|---|
| Box counting | N(ε) ∝ ε^−D | Sets (Sierpiński carpet) |
| fBm surface (image as a landscape) | E\|I(p₂) − I(p₁)\| ∝ \|p₂ − p₁\|^H, **D = 3 − H** | The paper's intensity Hurst |
| fBm profile (a curve) | E\|z(x+d) − z(x)\| ∝ d^H, **D = 2 − H** | Layer boundaries |
| Divider / ruler (coastline) | L(r) ∝ r^(1−D) | Curves (Koch) |

**D = 3 − H of the image intensity is just a relabelling** of the Hurst exponent the paper already uses, so on its own it adds no information. The new use is the **fractal dimension of the retina's layer boundaries**: the ILM (inner surface) and the RPE are treated as coastlines.

### Step 1: validate the estimators on known answers (`fractal_validation.py` → `results/v4/fractal_validation.png`)

Following the paper's Sec. 6.3:

| Test shape | True | Estimated |
|---|---|---|
| Sierpiński carpet, box counting | 1.8928 | **1.8928** |
| Koch curve, divider (coastline) method | 1.2619 | **1.267** (box counting 1.293) |
| Plane, 2-D NMSID | H = 1 | 0.965 |
| Exact fBm profiles, 1-D NMSID, H = 0.1 … 0.9 | H | within ±0.01 |
| fBm surfaces, 2-D NMSID (256 px) | 0.1 / 0.5 / 0.9 | 0.24 / 0.50 / 0.77 |
| Paper's estimator, 50×50 grid, k = 1–25 | 0.1 / 0.5 / 0.9 | 0.22 / 0.47 / 0.75 |
| … after the paper's 1.32× cubic upsampling | 0.1 / 0.5 / 0.9 | 0.26 / 0.51 / 0.76 |

What this shows:
- **The boundary (1-D) estimator is accurate.**
- **The 2-D intensity estimator compresses the scale:** low H reads high and high H reads low. Part of this is likely the approximate spectral surface generator.
- **Upsampling raises H by up to +0.04,** a direct, controlled confirmation of the Part 1 confound (healthy OCTID scans are upsampled more, which inflates their H).

### Step 2: retinal boundaries as coastlines (`extract_fractal.py`, `fractal_eval.py` → `results/v4/`)

**Method:**
- The ILM is re-located as the strongest dark-to-bright step, and the RPE as the brightest row, within ±4 rows of the v2 trace.
- This uses sub-pixel fitting and no lateral median filter, which would erase the small-scale roughness being measured.
- Heights are in µm.
- **14 `frac_*` features per scan:**
  - D = 2 − H from 1-D NMSID over lags of 1–16 columns, whole scan and central ±110 columns, with its fit r²;
  - roughness amplitude (mean step between neighbouring columns);
  - the divider dimension and its fit r².

**By class** (`fractal_by_class.png`), the patterns are anatomically sensible:
- **Epiretinal membrane has the most wrinkled ILM:** median D 1.16 vs. 1.10 for healthy eyes on OCTDL.
- **CNV, DME and drusen also raise ILM D** on Kermany (1.15–1.16 vs. 1.11).
- **AMD, CNV, RVO and RAO make the RPE much rougher:** amplitude 4–5 µm/column vs. 2.7 µm for healthy eyes.
- **Healthy RPE has the *highest* D.** A flat, healthy RPE has only noise-like jitter (H small, so D high). Disease adds real, persistent bumps, which raise H and lower D. Read **D** as the *type* of roughness and **amplitude** as its *size*.
- **They barely depend on the scanner:**
  - healthy Kermany vs. healthy OCTDL |d| = 0.01–0.58; most RPE features are below 0.4;
  - compare the strongest NDT features, at about 1.5;
  - the best single feature, central RPE D, has ε² 0.17 on OCTDL and 0.11 on Kermany, with |d| = 0.24.
- **The divider dimension stays close to 1.00 for every class.** At ruler sizes that fit the image, these nearly flat, self-affine curves look straight, so the coastline-ruler method is uninformative here; use the variogram D instead.

**Accuracy:**

| Task (balanced accuracy) | Fractal boundary features only (best) | v2 + disease + NDT robust | + fractal |
|---|---|---|---|
| OCTID CSR / DR / MH, 500×750 | 0.672 (chance 0.33) | 0.847–0.853 | 0.838–0.868 |
| OCTDL 7-class | 0.377 (chance 0.14) | 0.641–0.682 | 0.644–0.682 |
| Kermany 4-class | 0.464 (chance 0.25) | 0.676–0.685 | 0.677–0.680 |
| Kermany CNV / DME / drusen | 0.531 (chance 0.33) | 0.707–0.730 | 0.708–0.726 |

**Paired test** (`fractal_paired_test.json`, same folds, 3 repeats, patient bootstrap):
- **12 model × task comparisons; differences all lie between −0.015 and +0.020.**
- No confidence interval excludes zero.
- The only consistent gain is OCTID diseases with logistic regression: +0.020 in all 3 repeats, CI [−0.006, +0.044].

**Verdict:**
- **The boundary fractal dimension is a real, scanner-robust and interpretable signal**:
  - 14 numbers alone reach 2–3× chance;
  - the patterns match the anatomy (ERM wrinkles the ILM; AMD and CNV roughen the RPE).
- **It adds no accuracy on top of v2 + NDT:**
  - v2's RPE-elevation, thickness and band features already capture the same roughness;
  - as with D = 3 − H, a fractal relabel of an existing measurement doesn't add information.
- **Its value is explanatory:** a single, physically meaningful number per boundary that a clinician can read.

```bash
python fractal_validation.py                                   # ~1 min
python extract_fractal.py manifests/octid.csv                  # also octdl, kermany (~1 min each)
python fractal_eval.py                                         # ~8 min
```


## Known limitations

- **No patient IDs in OCTID.** Each image is its own group, so two scans of the same eye could land in train and test. The Kermany and OCTDL results use real patient IDs.
- **No pixel spacing for OCTID.** Its axial spacing is unknown (exported screenshots), so it is resampled by aspect ratio only.
- **Imperfect segmentation.** The classical RPE/ILM tracer fails on some severe cases (large cysts, extreme tilt). Per-scan QC features record this.

- **Kermany subset.** 2,401 of ~84k scans (at most 3 per patient), because of disk space.
- **No hyperparameter tuning** for either the CNN or v2. Both could improve.
- **Lateral spacing assumed.** The fractal divider dimension and the MAPOD cyst shapes assume 11.7 um per column (6 mm / 512), which the images do not record.
- **Synthetic lesions (MAPOD).** These are painted, fully anechoic pockets on the flattened canvas, with an assumed lateral spacing. The POD numbers compare detectors; they are not clinical detection limits.

## Citation

- Kermany D. et al. *Identifying medical diagnoses and treatable diseases by image-based deep learning.* Cell 172(5), 2018. Data: https://data.mendeley.com/datasets/rscbjbr9sj
- Kulyabin M. et al. *OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods.* Scientific Data 11, 2024 (arXiv:2312.08255). Data: https://zenodo.org/records/10839556
- Gholami P., Roy P., Parthasarathy M.K., Lakshminarayanan V. *OCTID: Optical Coherence Tomography Image Database*. arXiv:1812.07056.
- Athe P., Shakya S., Munshi P., Luke A., Mewes D. *Characterization of multiphase flow in bubble columns using KT-1 signature and fractal dimension.* Flow Measurement and Instrumentation 33, 122–137, 2013. doi:10.1016/j.flowmeasinst.2013.05.005
- Mandelbrot B.B. *The Fractal Geometry of Nature.* W.H. Freeman, 1982.
