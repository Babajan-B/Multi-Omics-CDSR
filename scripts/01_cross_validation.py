"""Repeated stratified 5-fold cross-validation (3 repeats) with fold-level results.

  * fold-level AUCs for the Nadeau-Bengio corrected resampled t-test
  * a subtype (OncotreePrimaryDisease) baseline next to the lineage baseline
  * curated drug-specific biomarkers forced into the RNA and early-fusion sets
  * alternative response definitions (tertile extremes; lower quartile vs rest)

Usage: python 01_cross_validation.py            (all drugs, parallel)
"""
import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load, prep, fit_predict, late_fusion, fname, DRUGS, OUT)

K, R, SEED = 5, 3, 2026
MAIN = ["Tissue", "Subtype", "RNA", "RNA_cur", "Early", "Early_cur", "Late"]
ALT = ["Tissue", "RNA", "Early"]


def labels_for(df, scheme):
    v = df["LN_IC50"].values
    if scheme == "median":
        return df.index.values, (v < np.median(v)).astype(int)
    if scheme == "tertile":
        lo, hi = np.quantile(v, [1 / 3, 2 / 3])
        keep = (v <= lo) | (v >= hi)
        return df.index.values[keep], (v[keep] <= lo).astype(int)
    if scheme == "quartile":
        return df.index.values, (v < np.quantile(v, 0.25)).astype(int)
    raise ValueError(scheme)


def run_drug(drug):
    expr, cn, mut, lin, labels = load()
    d = labels[labels.drug == drug].groupby("ModelID")["LN_IC50"].mean().to_frame()
    rows, oof = [], {}
    for scheme in ("median", "tertile", "quartile"):
        ids, y = labels_for(d, scheme)
        models = MAIN if scheme == "median" else ALT
        P = {m: np.full((R, len(ids)), np.nan) for m in models}
        for r in range(R):
            skf = StratifiedKFold(K, shuffle=True, random_state=SEED + r)
            for f, (itr, ite) in enumerate(skf.split(ids, y)):
                tr, te = ids[itr], ids[ite]
                seed = SEED + 100 * r + f
                feats = prep(tr, te, expr, cn, mut, lin, drug, curated=False)
                if scheme == "median":
                    fc = prep(tr, te, expr, cn, mut, lin, drug, curated=True)
                    feats["RNA_cur"], feats["Early_cur"] = fc["RNA"], fc["Early"]
                for m in models:
                    if m == "Late":
                        p = late_fusion(feats, y[itr], seed)
                    else:
                        p = fit_predict(feats[m][0], y[itr], feats[m][1], seed)
                    P[m][r, ite] = p
                    rows.append(dict(drug=drug, scheme=scheme, repeat=r, fold=f, model=m,
                                     n_train=len(itr), n_test=len(ite),
                                     auc=roc_auc_score(y[ite], p)))
        oof[scheme] = dict(ids=ids, y=y, **{m: P[m] for m in models})
    np.savez(os.path.join(OUT, "cv", fname(drug) + ".npz"),
             **{f"{s}__{k}": v for s, dct in oof.items() for k, v in dct.items()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cv", fname(drug) + "_folds.csv"), index=False)
    return drug


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "cv"), exist_ok=True)
    todo = sys.argv[1:] or DRUGS
    done = Parallel(n_jobs=min(12, len(todo)), verbose=5)(delayed(run_drug)(d) for d in todo)
    print("done", done)
