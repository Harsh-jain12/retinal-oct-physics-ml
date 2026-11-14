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

## Citation

- Kermany D. et al. *Identifying medical diagnoses and treatable diseases by image-based deep learning.* Cell 172(5), 2018. Data: https://data.mendeley.com/datasets/rscbjbr9sj
- Kulyabin M. et al. *OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods.* Scientific Data 11, 2024 (arXiv:2312.08255). Data: https://zenodo.org/records/10839556
- Gholami P., Roy P., Parthasarathy M.K., Lakshminarayanan V. *OCTID: Optical Coherence Tomography Image Database*. arXiv:1812.07056.
- Athe P., Shakya S., Munshi P., Luke A., Mewes D. *Characterization of multiphase flow in bubble columns using KT-1 signature and fractal dimension.* Flow Measurement and Instrumentation 33, 122–137, 2013. doi:10.1016/j.flowmeasinst.2013.05.005
- Mandelbrot B.B. *The Fractal Geometry of Nature.* W.H. Freeman, 1982.
