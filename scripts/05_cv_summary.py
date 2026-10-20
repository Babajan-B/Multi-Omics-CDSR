"""Summaries of the cross-validation run (01_cross_validation.py).

  * reproduction of the original AUCs with new random partitions
  * corrected resampled t-test (fold level) next to DeLong (averaged OOF)
  * curated-biomarker models, subtype baseline
  * alternative response thresholds and stability of the drug ranking
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, PUB, DRUGS, fname, delong_ci, delong_paired, corrected_t, holm

orig = pd.read_csv(os.path.join(PUB, "Table_S1_internal_performance.csv")).pivot(index="drug", columns="model", values="auc")

COMPS = {"median": [("Early", "RNA"), ("Late", "RNA"), ("RNA", "Tissue"), ("Subtype", "Tissue"),
                    ("RNA", "Subtype"), ("RNA_cur", "RNA"), ("Early_cur", "Early")],
         "tertile": [("RNA", "Tissue"), ("Early", "RNA")],
         "quartile": [("RNA", "Tissue"), ("Early", "RNA")]}

perf, comp = [], []
for dr in DRUGS:
    z = np.load(os.path.join(OUT, "cv", fname(dr) + ".npz"), allow_pickle=True)
    folds = pd.read_csv(os.path.join(OUT, "cv", fname(dr) + "_folds.csv"))
    for scheme, pairs in COMPS.items():
        y = z[f"{scheme}__y"]
        models = sorted({m for p in pairs for m in p})
        P = {m: np.nanmean(z[f"{scheme}__{m}"], axis=0) for m in models}
        for m in models:
            a, lo, hi = delong_ci(y, P[m])
            perf.append(dict(drug=dr, scheme=scheme, model=m, n=len(y), auc=a, lo=lo, hi=hi))
        f = folds[folds.scheme == scheme]
        for a_, b_ in pairs:
            d, lo, hi, p, _ = delong_paired(y, P[a_], P[b_])
            fa = f[f.model == a_].set_index(["repeat", "fold"]).auc
            fb = f[f.model == b_].set_index(["repeat", "fold"]).auc
            nt = f[f.model == a_].set_index(["repeat", "fold"])
            md, mlo, mhi, mp = corrected_t((fa - fb).values, nt.n_train.mean(), nt.n_test.mean())
            comp.append(dict(drug=dr, scheme=scheme, comparison=f"{a_} vs {b_}", delta=d, lo=lo, hi=hi,
                             p_delong=p, fold_mean_delta=md, ct_lo=mlo, ct_hi=mhi, p_corrected_t=mp))
perf = pd.DataFrame(perf)
comp = pd.DataFrame(comp)
for key, g in comp.groupby(["scheme", "comparison"]):
    comp.loc[g.index, "p_delong_holm"] = holm(g.p_delong.values)
    comp.loc[g.index, "p_ct_holm"] = holm(g.p_corrected_t.values)
perf.to_csv(os.path.join(OUT, "cv_performance.csv"), index=False)
comp.to_csv(os.path.join(OUT, "cv_comparisons.csv"), index=False)

res = {}
m = perf[perf.scheme == "median"].pivot(index="drug", columns="model", values="auc")
o = orig.loc[m.index]
res["reproduction_max_abs_diff"] = {k: float((m[k] - o[k]).abs().max()) for k in ("Tissue", "RNA", "Early", "Late")}
res["reproduction_median_abs_diff"] = {k: float((m[k] - o[k]).abs().median()) for k in ("Tissue", "RNA", "Early", "Late")}
for (scheme, c), g in comp.groupby(["scheme", "comparison"]):
    res[f"{scheme}|{c}"] = dict(median_delta=float(g.delta.median()), range=[float(g.delta.min()), float(g.delta.max())],
                                n_pos=int((g.delta > 0).sum()), n_delong_nominal=int((g.p_delong < 0.05).sum()),
                                n_delong_holm=int((g.p_delong_holm < 0.05).sum()),
                                n_ct_nominal=int((g.p_corrected_t < 0.05).sum()),
                                n_ct_holm=int((g.p_ct_holm < 0.05).sum()),
                                wilcoxon_p=float(stats.wilcoxon(g.delta).pvalue))
# ranking stability of RNA AUC across thresholds
piv = perf[perf.model == "RNA"].pivot(index="drug", columns="scheme", values="auc")
res["rank_stability_rna"] = {f"median_vs_{s}": float(stats.spearmanr(piv["median"], piv[s])[0]) for s in ("tertile", "quartile")}
res["rna_auc_by_scheme_median"] = {s: float(piv[s].median()) for s in piv.columns}
res["rna_auc_by_scheme_range"] = {s: [float(piv[s].min()), float(piv[s].max())] for s in piv.columns}
json.dump(res, open(os.path.join(OUT, "cv_summary.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
