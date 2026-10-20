"""Redundancy between copy-number and expression features:
Spearman correlation between copy number and expression of the same gene for
the 1000 most variable copy-number genes, and how well lineage predicts
copy-number principal components."""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, OUT, PUB, DRUGS
expr, cn, mut, lin, labels = load()
out = {}
ids = labels[labels.drug == "Docetaxel"].ModelID.unique()
c = cn.loc[ids].dropna(axis=1, how="all")
top = c.var().sort_values(ascending=False).index[:1000]
shared = [g for g in top if g in expr.columns]
r = np.array([stats.spearmanr(c[g], expr.loc[ids, g])[0] for g in shared])
r = r[~np.isnan(r)]
out["cn_expr_same_gene_median_rho"] = float(np.median(r))
out["cn_expr_same_gene_iqr"] = [float(np.percentile(r, 25)), float(np.percentile(r, 75))]
out["frac_rho_gt_0.3"] = float((r > 0.3).mean())
out["n_genes"] = int(len(r))
perf = pd.read_csv(os.path.join(PUB, "Table_S1_internal_performance.csv"))
p = perf.pivot(index="drug", columns="model", values="auc")
out["CN_minus_Tissue_median"] = float((p.CN - p.Tissue).median())
out["CN_vs_Tissue_spearman_across_drugs"] = float(stats.spearmanr(p.CN, p.Tissue)[0])
json.dump(out, open(os.path.join(OUT, "cn_redundancy.json"), "w"), indent=1)
print(out)
