"""External validation on PRISM (secondary screen 20Q2) from the raw screen data.

  * tissue-only, expression, curated-expression, early-fusion and late-fusion models
  * three definitions of PRISM sensitivity:
      ext_median   - median of the evaluation cohort (original definition)
      fixed_overlap- median PRISM AUC of the GDSC2-overlap lines, i.e. a
                     threshold fixed without any evaluation-cohort data
      prism_all    - median over every STR-passed PRISM line for the drug
  * threshold-free Spearman correlation with PRISM AUC
"""
import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load, prep, fit_predict, late_fusion, fname, DRUGS, OUT, RAW,
                    delong_ci, boot_auc)

PRISM_NAME = {"Docetaxel": "docetaxel", "Paclitaxel": "paclitaxel", "Cisplatin": "cisplatin",
              "Gemcitabine": "gemcitabine", "5-Fluorouracil": "5-fluorouracil",
              "Irinotecan": "irinotecan", "Nutlin-3a (-)": "nutlin-3", "Dabrafenib": "dabrafenib",
              "Trametinib": "trametinib", "Lapatinib": "lapatinib", "Erlotinib": "erlotinib",
              "Alpelisib": "alpelisib"}
SEED = 7


def prism_table():
    cache = os.path.join(OUT, "prism_auc_per_line.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    p = pd.read_csv(os.path.join(RAW, "PRISM_secondary_dose_response.csv"),
                    usecols=["depmap_id", "name", "auc", "passed_str_profiling", "screen_id"])
    p = p[p.name.isin(PRISM_NAME.values()) & (p.passed_str_profiling.astype(str).str.upper() == "TRUE")]
    p = p.dropna(subset=["auc"]).groupby(["name", "depmap_id"], as_index=False).auc.mean()
    p.to_csv(cache, index=False)
    return p


def run(drug):
    from sklearn.metrics import roc_auc_score
    expr, cn, mut, lin, labels = load()
    pr = prism_table()
    pr = pr[pr.name == PRISM_NAME[drug]].set_index("depmap_id").auc
    g = labels[labels.drug == drug].groupby("ModelID")["LN_IC50"].mean()
    tr = g.index.values
    ytr = (g.values < np.median(g.values)).astype(int)
    overlap = [i for i in tr if i in pr.index]
    ext = np.array([i for i in pr.index if i in expr.index and i not in set(tr)])
    thresholds = dict(ext_median=pr.loc[ext].median(), fixed_overlap=pr.loc[overlap].median(),
                      prism_all=pr.median())
    feats = prep(tr, ext, expr, cn, mut, lin, drug, curated=False)
    fc = prep(tr, ext, expr, cn, mut, lin, drug, curated=True)
    feats["RNA_cur"] = fc["RNA"]
    preds = {m: fit_predict(feats[m][0], ytr, feats[m][1], SEED) for m in ("Tissue", "RNA", "RNA_cur", "Early")}
    preds["Late"] = late_fusion(feats, ytr, SEED)
    conc = stats.spearmanr(g.loc[overlap], pr.loc[overlap])[0]
    rows = []
    for thr_name, thr in thresholds.items():
        y = (pr.loc[ext].values < thr).astype(int)
        for m, p in preds.items():
            lo, hi = boot_auc(y, p, B=2000, seed=SEED)
            rows.append(dict(drug=drug, threshold=thr_name, threshold_value=thr, model=m,
                             n_train=len(tr), n_external=len(ext), n_overlap=len(overlap),
                             frac_sensitive=y.mean(), auc=roc_auc_score(y, p), lo=lo, hi=hi,
                             spearman_vs_neg_prism_auc=stats.spearmanr(p, -pr.loc[ext].values)[0],
                             concordance_rho=conc))
    np.savez(os.path.join(OUT, "external", fname(drug) + ".npz"), ids=ext,
             prism_auc=pr.loc[ext].values, **preds)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "external"), exist_ok=True)
    prism_table()
    nj = int(os.environ.get("NJ", "3"))
    out = Parallel(n_jobs=nj, verbose=5)(delayed(run)(d) for d in DRUGS)
    pd.concat(out).to_csv(os.path.join(OUT, "external_validation.csv"), index=False)
    print(pd.concat(out).query("threshold=='ext_median'")[["drug", "model", "n_external", "auc"]])
