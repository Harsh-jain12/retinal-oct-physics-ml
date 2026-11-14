"""v6 / Step 3: hybrid model = frozen deep embeddings + physics, compared with the fine-tuned CNN.

Hybrid input (all fitted inside the training fold):
    [PCA-64 of a frozen embedding | v5 best features (171) | image-POD r=50 | POD residual e(x), r=25]
    -> soft vote (LR + LightGBM + RBF-SVM)
Two embeddings: ResNet-18 (ImageNet) and DINOv2-S. The DINOv2-S hybrid is the pre-registered
"v6 hybrid" (chosen before seeing results); the ResNet-18 variant and the embedding-only model are
reported alongside. Paired tests on identical folds vs v5 best and vs the fine-tuned CNN (seed-0 folds).

Explainability check: grouped permutation importance of the DINOv2 hybrid on OCTID CSR/DR/MH with
groups = the 4 v5 feature families split by anatomical band where possible, image-POD, e(x),
embedding. Acceptance: the hybrid matches the CNN within the CI *and* physics groups stay in the top 10.

  python v6_hybrid.py [--task slug|all|cross] [--quick]
Writes results/v6/hybrid_<task>.json, hybrid_importance.json, oof_<task>_hybrid_*.csv
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from retina.hybrid import HybridClassifier
from retina.v6data import CROSS, SLUG, TASKS
from retina.v6eval import argmax_labels, folds, paired, save_oof
from retina.v6run import cross_data, cross_fit_predict, cross_summary, dump, run_models, task_data

warnings.filterwarnings("ignore")
OUT = "results/v6"
PHYS = ("feat", "pod", "resid")
SPECS = {
    "hybrid DINOv2-S": dict(use=PHYS + ("emb_dino",)),
    "hybrid ResNet-18": dict(use=PHYS + ("emb_resnet",)),
    "DINOv2-S embedding only": dict(use=("emb_dino",)),
    "ResNet-18 embedding only": dict(use=("emb_resnet",)),
}
BLOCKS = ("feat", "tile", "emb_resnet", "emb_dino")


def oof_or_none(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def within(task, seeds, quick):
    data = task_data(task, BLOCKS, quick)
    res = {"task": task, "n": len(data["y"]), "models": run_models(data, SPECS, seeds, OUT, save=tuple(SPECS))}
    y, g = data["y"], data["groups"]
    head = data["preds"]["hybrid DINOv2-S"][0]
    res["paired"] = {}
    for ref, f in (("v5 best (soft vote)", f"{OUT}/oof_{data['slug']}_v5_best_soft_vote.csv"),
                   ("CNN (fine-tuned ResNet-18)", f"{OUT}/oof_{data['slug']}_cnn.csv")):
        b = oof_or_none(f)
        if b is not None and not quick:
            assert (b.path.values == data["paths"]).all()
            res["paired"][f"hybrid DINOv2-S vs {ref}"] = paired(y, b.pred.values, head, g)
            res["paired"][f"hybrid ResNet-18 vs {ref}"] = paired(y, b.pred.values, data["preds"]["hybrid ResNet-18"][0], g)
            print(f"  paired vs {ref}: {res['paired'][f'hybrid DINOv2-S vs {ref}']}", flush=True)
    dump(res, f"{OUT}/hybrid_{data['slug']}.json")
    return data


def cross():
    data = cross_data(BLOCKS)
    res = {"task": CROSS, "models": {}, "paired": {}}
    preds = {}
    for name, kw in SPECS.items():
        _, P, classes = cross_fit_predict(data, kw)
        preds[name] = argmax_labels(P, classes)
        res["models"][name] = cross_summary(data["yte"], preds[name], data["gte"])
        save_oof(f"{OUT}/oof_{SLUG[CROSS]}_{name_slug(name)}.csv", data["paths"], data["yte"], P, classes)
        print(f"  {CROSS} {name:28s} bacc {res['models'][name]['bacc']:.3f}", flush=True)
    for ref, f in (("v5 best (soft vote)", f"{OUT}/oof_{SLUG[CROSS]}_v5_best_soft_vote.csv"),
                   ("CNN (fine-tuned ResNet-18)", f"{OUT}/oof_{SLUG[CROSS]}_cnn.csv")):
        b = oof_or_none(f)
        if b is not None:
            res["paired"][f"hybrid DINOv2-S vs {ref}"] = paired(data["yte"], b.pred.values, preds["hybrid DINOv2-S"], data["gte"])
    dump(res, f"{OUT}/hybrid_{SLUG[CROSS]}.json")


def name_slug(n):
    from retina.v6run import model_slug
    return model_slug(n)


def feature_groups(cols):
    """Physics feature families (v5 set) -> column indices in the 'feat' block."""
    fam = {}
    for i, c in enumerate(cols):
        if c.startswith("band_"):
            b = c.split("_")[1]
            fam.setdefault(f"texture: {b} band", []).append(i)
        elif c.startswith("ndt_"):
            fam.setdefault("NDT (pulse-echo)", []).append(i)
        elif c.startswith("frac_"):
            fam.setdefault("boundary fractal", []).append(i)
        else:
            fam.setdefault("disease-specific (thickness, fluid, gap, RPE)", []).append(i)
    return fam


def importance(task="OCTID CSR/DR/MH (500x750)", seed=0, n_rep=5, n_splits=5):
    """Grouped permutation importance of the DINOv2 hybrid, permuting groups of the *model input Z*
    (after the in-fold POD/PCA) in held-out folds; same idea as retina.evaluation.grouped_permutation_importance."""
    from retina.v6data import feat_matrix
    data = task_data(task, BLOCKS, n_splits == 2)
    y, g, X = data["y"], data["groups"], data["X"]
    _, cols = feat_matrix(data["frame"])
    fam = feature_groups(cols)
    rng = np.random.default_rng(seed)
    drops = {}
    for tr, te in folds(y, g, seed, n_splits):
        kw = dict(SPECS["hybrid DINOv2-S"])
        if data["ref_tiles"] is not None:
            kw["ref_tiles"] = data["ref_tiles"]
        m = HybridClassifier(data["layout"], seed=seed, **kw).fit(X[tr], y[tr])
        Z = m.transform_z(X[te])
        base = balanced_accuracy_score(y[te], m.clf_.predict(Z))
        zg = m.z_groups()
        groups = {f"physics: {k}": [zg["feat"][i] for i in v] for k, v in fam.items()}
        groups["image-POD (50 modes)"] = zg["pod"]
        groups["POD residual e(x)"] = zg["resid"]
        groups["DINOv2-S embedding (PCA 64)"] = zg["emb_dino"]
        for name, idx in groups.items():
            for _ in range(n_rep):
                Zp = Z.copy()
                Zp[:, idx] = Z[rng.permutation(len(te))][:, idx]
                drops.setdefault(name, []).append(base - balanced_accuracy_score(y[te], m.clf_.predict(Zp)))
    imp = pd.Series({k: float(np.mean(v)) for k, v in drops.items()}).sort_values(ascending=False)
    out = {"task": task, "importance": imp.to_dict(), "rank": list(imp.index),
           "physics_in_top10": [k for k in imp.index[:10] if k.startswith("physics") or k in ("image-POD (50 modes)", "POD residual e(x)")]}
    print(imp.round(4).to_string(), flush=True)
    dump(out, f"{OUT}/hybrid_importance.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", help="task slug, 'cross', 'importance' or 'all'")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = (0,) if a.quick else (0, 1, 2)
    for t in TASKS:
        if a.task in ("all", SLUG[t]):
            within(t, seeds, a.quick)
    if a.task in ("all", "cross"):
        cross()
    if a.task in ("all", "importance"):
        importance(n_splits=2 if a.quick else 5)


if __name__ == "__main__":
    main()
