"""v6 / Step 6: conformal triage on the best v6 model (same protocol as v5_conformal.py).

Class-conditional split conformal (retina/conformal.py) around the v6 hybrid (DINOv2-S + physics).
Calibration patients are split off *inside* each outer training fold; the hybrid's POD / PCA /
residual are fitted on the proper-training part only (HybridClassifier.fit), so nothing sees the
calibration or test scans. Same tasks, alphas and seeds (0, 1) as v5, so rows compare 1:1.

For triage the clinically relevant number is the ABNORMAL coverage: the fraction of diseased eyes
whose prediction set contains ABNORMAL (i.e. that are *not* cleared as healthy). It must be >= 1 - alpha.

  python v6_conformal.py [--quick]
Writes results/v6/conformal.json
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import warnings

import numpy as np

from retina.conformal import conformal_cv, summarize
from retina.hybrid import HybridClassifier
from retina.v6run import dump, task_data

warnings.filterwarnings("ignore")
OUT = "results/v6"
ALPHAS = (0.05, 0.10)
SEEDS = (0, 1)
SPEC = dict(use=("feat", "pod", "resid", "emb_dino"))
TASKS = {  # name: (v6 task, binary triage?)
    "OCTDL triage NORMAL vs ABNORMAL": ("OCTDL 7-class", True),
    "Kermany triage NORMAL vs ABNORMAL": ("Kermany 4-class", True),
    "OCTID CSR/DR/MH (500x750)": ("OCTID CSR/DR/MH (500x750)", False),
    "Kermany CNV/DME/DRUSEN": ("Kermany CNV/DME/DRUSEN", False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 outer folds")
    a = ap.parse_args()
    seeds = (0,) if a.quick else SEEDS
    v5 = {(r["task"], r["alpha"]): r for r in json.load(open("results/v5/conformal.json"))}
    res = []
    for tname, (task, binary) in TASKS.items():
        data = task_data(task, ("feat", "tile", "emb_dino"), a.quick)
        y = np.where(data["y"] == "NORMAL", "NORMAL", "ABNORMAL") if binary else data["y"]
        kw = dict(SPEC)
        if data["ref_tiles"] is not None:
            kw["ref_tiles"] = data["ref_tiles"]
        fn = lambda s: HybridClassifier(data["layout"], seed=s, **kw)
        for al in ALPHAS:
            runs = []
            for s in seeds:
                S, _, classes = conformal_cv(fn, data["X"], y, data["groups"], alpha=al, seed=s,
                                             n_splits=data["n_splits"])
                runs.append(summarize(S, y, classes))
            agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if k != "coverage_per_class"}
            agg["coverage_per_class"] = {c: float(np.mean([r["coverage_per_class"][c] for r in runs]))
                                         for c in runs[0]["coverage_per_class"]}
            row = dict(task=tname, alpha=al, target=1 - al, n=len(y), model="v6 hybrid (DINOv2-S + physics)", **agg)
            ref = v5.get((tname, al))
            if ref:
                row["v5"] = {k: ref[k] for k in ("coverage", "mean_set_size", "singleton_rate", "singleton_accuracy",
                                                 "coverage_per_class")}
            res.append(row)
            pc = ", ".join(f"{c} {v:.3f}" for c, v in row["coverage_per_class"].items())
            print(f"{tname:36s} alpha={al:.2f} coverage {row['coverage']:.3f} ({pc}) set {row['mean_set_size']:.2f} "
                  f"single {row['singleton_rate']:.2f} (acc {row['singleton_accuracy']:.3f})"
                  + (f" | v5 set {ref['mean_set_size']:.2f} single {ref['singleton_rate']:.2f}" if ref else ""), flush=True)
    dump(res, f"{OUT}/conformal.json")


if __name__ == "__main__":
    main()
