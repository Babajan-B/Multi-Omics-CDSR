"""Statistics computed from the outputs of the primary analysis (DATA_DIR).

  a) pooled (random-effects) fusion gain, across-drug Wilcoxon test,
     equivalence bounds and minimum detectable differences
  b) paired DeLong differences within each cross-validation repeat
  c) paired bootstrap test of regression fusion gains
  d) robustness of the association between screen concordance and external AUC
  e) per-lineage LOCO uncertainty and minimum-lineage-size sensitivity
  f) copy-number missingness
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, OUT, DRUGS, CYTOTOXIC, fname, delong_paired, holm, load

rng = np.random.default_rng(2026)
res = {}

# ------------------------------------------------------------------ a) fusion
dl = pd.read_csv(os.path.join(V2, "summary_delong.csv"))
rows = []
for comp in ("Early vs RNA", "Late vs RNA"):
    t = dl[dl.comparison == comp].set_index("drug").loc[DRUGS]
    d = t.delta_auc.values
    se = np.abs(d / t.z.values)
    w = 1 / se**2
    fixed = np.sum(w * d) / w.sum()
    Q = np.sum(w * (d - fixed) ** 2)
    tau2 = max(0.0, (Q - 11) / (w.sum() - (w**2).sum() / w.sum()))
    wr = 1 / (se**2 + tau2)
    pooled = np.sum(wr * d) / wr.sum()
    pse = np.sqrt(1 / wr.sum())
    I2 = max(0.0, (Q - 11) / Q) if Q > 0 else 0.0
    wil = stats.wilcoxon(d)
    z90 = stats.norm.ppf(0.95)
    lo90, hi90 = d - z90 * se, d + z90 * se
    # minimum detectable difference at 80% power
    mdd_raw = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se
    mdd_holm = (stats.norm.ppf(1 - 0.025 / 12) + stats.norm.ppf(0.8)) * se
    res[comp] = dict(pooled=pooled, lo=pooled - 1.96 * pse, hi=pooled + 1.96 * pse,
                     p=2 * stats.norm.sf(abs(pooled / pse)), tau2=tau2, I2=I2,
                     n_positive=int((d > 0).sum()), median=float(np.median(d)),
                     wilcoxon_W=float(wil.statistic), wilcoxon_p=float(wil.pvalue),
                     max_upper95=float((d + 1.96 * se).max()),
                     all_90ci_within_0p05=bool(((lo90 > -0.05) & (hi90 < 0.05)).all()),
                     n_90ci_within_0p03=int(((lo90 > -0.03) & (hi90 < 0.03)).sum()),
                     mdd_raw_median=float(np.median(mdd_raw)), mdd_holm_median=float(np.median(mdd_holm)),
                     mdd_holm_range=[float(mdd_holm.min()), float(mdd_holm.max())])
    for i, dr in enumerate(DRUGS):
        rows.append(dict(drug=dr, comparison=comp, delta=d[i], se=se[i],
                         ci95_lo=d[i] - 1.96 * se[i], ci95_hi=d[i] + 1.96 * se[i],
                         ci90_lo=lo90[i], ci90_hi=hi90[i], mdd80_unadj=mdd_raw[i], mdd80_holm=mdd_holm[i]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "a_fusion_equivalence_power.csv"), index=False)

# ------------------------------------------------- b) per-repeat paired DeLong
rows = []
for dr in DRUGS:
    z = np.load(os.path.join(V2, "cv", fname(dr) + ".npz"), allow_pickle=True)
    y = z["y"]
    for a, b in (("Early", "RNA"), ("Late", "RNA"), ("RNA", "Tissue")):
        for r in range(z["oof_" + a].shape[0]):
            dd, lo, hi, p, _ = delong_paired(y, z["oof_" + a][r], z["oof_" + b][r])
            rows.append(dict(drug=dr, comparison=f"{a} vs {b}", repeat=r, delta=dd, lo=lo, hi=hi, p=p))
rep = pd.DataFrame(rows)
rep.to_csv(os.path.join(OUT, "b_per_repeat_delong.csv"), index=False)
agg = rep.groupby(["comparison", "drug"]).agg(dmin=("delta", "min"), dmax=("delta", "max"),
                                              pmin=("p", "min"), pmax=("p", "max")).reset_index()
agg.to_csv(os.path.join(OUT, "b_per_repeat_summary.csv"), index=False)

# --------------------------------------------- c) regression fusion bootstrap
rows = []
for dr in DRUGS:
    z = np.load(os.path.join(V2, "cv", fname(dr) + ".npz"), allow_pickle=True)
    yc, r1, r2 = z["y_cont"], z["reg_RNA"], z["reg_Early"]
    s1, s2 = stats.spearmanr(r1, yc)[0], stats.spearmanr(r2, yc)[0]
    B = []
    n = len(yc)
    for _ in range(2000):
        i = rng.integers(0, n, n)
        B.append(stats.spearmanr(r2[i], yc[i])[0] - stats.spearmanr(r1[i], yc[i])[0])
    B = np.array(B)
    p = min(1.0, 2 * min((B <= 0).mean(), (B >= 0).mean()))
    rows.append(dict(drug=dr, cls="Cytotoxic" if dr in CYTOTOXIC else "Targeted",
                     rho_RNA=s1, rho_Early=s2, delta=s2 - s1,
                     lo=np.percentile(B, 2.5), hi=np.percentile(B, 97.5), p=max(p, 1 / 2000)))
reg = pd.DataFrame(rows)
reg["p_holm"] = holm(reg.p.values)
reg.to_csv(os.path.join(OUT, "c_regression_fusion.csv"), index=False)
w = stats.wilcoxon(reg.delta)
res["regression"] = dict(n_positive=int((reg.delta > 0).sum()), median=float(reg.delta.median()),
                         wilcoxon_p=float(w.pvalue), n_holm_sig=int((reg.p_holm < 0.05).sum()),
                         n_nominal=int((reg.p < 0.05).sum()))

# ------------------------------------------------- d) concordance robustness
ext = pd.read_csv(os.path.join(V2, "summary_external.csv"))
perf = pd.read_csv(os.path.join(V2, "summary_performance.csv"))
conc = {}
for m in ("RNA", "Early"):
    e = ext[ext.model == m].set_index("drug").loc[DRUGS]
    x, y = e.label_concordance_rho.values, e.auc.values
    internal = perf[perf.model == m].set_index("drug").loc[DRUGS].auc.values
    r0, p0 = stats.spearmanr(x, y)
    bs = []
    for _ in range(10000):
        i = rng.integers(0, 12, 12)
        if len(np.unique(i)) < 4:
            continue
        bs.append(stats.spearmanr(x[i], y[i])[0])
    loo = [stats.spearmanr(np.delete(x, k), np.delete(y, k)) for k in range(12)]
    rxy, rxz, ryz = r0, stats.spearmanr(x, internal)[0], stats.spearmanr(y, internal)[0]
    partial = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    tp = partial * np.sqrt(9 / (1 - partial**2))
    conc[m] = dict(rho=r0, p=p0, boot_lo=float(np.nanpercentile(bs, 2.5)), boot_hi=float(np.nanpercentile(bs, 97.5)),
                   loo_min=float(min(l[0] for l in loo)), loo_max=float(max(l[0] for l in loo)),
                   loo_pmax=float(max(l[1] for l in loo)),
                   internal_vs_external=float(ryz), internal_vs_concordance=float(rxz),
                   partial=float(partial), partial_p=float(2 * stats.t.sf(abs(tp), 9)))
res["concordance"] = conc

# ------------------------------------------------------------------ e) LOCO
mv = pd.read_csv(os.path.join(V2, "summary_mixed_vs_loco.csv"))
lo = pd.read_csv(os.path.join(V2, "summary_loco.csv"))
lr = lo[lo.model == "RNA"]
by_lin = lr.groupby("lineage").agg(pairs=("drug", "size"), n_min=("n_test", "min"), n_max=("n_test", "max"),
                                   median_auc=("auc_loco", "median"),
                                   median_ci_width=("loco_hi", "median")).reset_index()
lr = lr.assign(width=lr.loco_hi - lr.loco_lo, ci_has_half=(lr.loco_lo <= 0.5) & (lr.loco_hi >= 0.5),
               below=(lr.loco_hi < 0.5), above=(lr.loco_lo > 0.5))
by_lin = lr.groupby("lineage").agg(pairs=("drug", "size"), n_min=("n_test", "min"), n_max=("n_test", "max"),
                                   median_auc=("auc_loco", "median"), q1=("auc_loco", lambda s: s.quantile(.25)),
                                   q3=("auc_loco", lambda s: s.quantile(.75)), median_ci_width=("width", "median"),
                                   ci_above_half=("above", "sum"), ci_below_half=("below", "sum")).reset_index()
by_lin.sort_values("n_max", ascending=False).to_csv(os.path.join(OUT, "e_loco_by_lineage.csv"), index=False)
by_drug = lr.groupby("drug").agg(pairs=("drug", "size"), median_auc=("auc_loco", "median"),
                                 median_ci_width=("width", "median"), ci_above_half=("above", "sum"),
                                 ci_below_half=("below", "sum")).reset_index()
by_drug.to_csv(os.path.join(OUT, "e_loco_by_drug.csv"), index=False)
thr = []
for m in ("RNA", "Early"):
    t = mv[mv.model == m]
    for nmin in (15, 20, 25, 30):
        s = t[t.n >= nmin]
        thr.append(dict(model=m, min_lines=nmin, pairs=len(s), lineages=s.lineage.nunique(),
                        lineage_list="; ".join(sorted(s.lineage.unique())),
                        median_gap=s.gap.median(), mixed_higher=int((s.gap > 0).sum()),
                        wilcoxon_p=stats.wilcoxon(s.auc_mixed, s.auc_loco).pvalue,
                        loco_below_half=int((s.auc_loco < 0.5).sum())))
pd.DataFrame(thr).to_csv(os.path.join(OUT, "e_loco_threshold_sensitivity.csv"), index=False)
res["loco"] = dict(pairs=int(len(lr)), ci_width_median=float(lr.width.median()),
                   ci_width_range=[float(lr.width.min()), float(lr.width.max())],
                   ci_contains_half=int(lr.ci_has_half.sum()), ci_above=int(lr.above.sum()),
                   ci_below=int(lr.below.sum()),
                   width_small=float(lr[lr.n_test < 25].width.median()),
                   width_large=float(lr[lr.n_test >= 40].width.median()))

# ------------------------------------------------------------ f) CN missing
expr, cn, mut, lin, labels = load()
ids = labels.ModelID.unique()
c = cn.loc[ids]
allmiss = c.isna().all()
res["cn_missing"] = dict(genes=int(c.shape[1]), all_missing_genes=int(allmiss.sum()),
                         missing_fraction_retained=float(c.loc[:, ~allmiss].isna().values.mean()),
                         n_missing_retained=int(c.loc[:, ~allmiss].isna().values.sum()))

# lineage composition (haematological lines excluded from LOCO)
li = lin.loc[ids, "OncotreeLineage"].value_counts()
res["lineages"] = li.to_dict()

json.dump(res, open(os.path.join(OUT, "summary_stats.json"), "w"), indent=2, default=float)
print(json.dumps(res, indent=1, default=float))
