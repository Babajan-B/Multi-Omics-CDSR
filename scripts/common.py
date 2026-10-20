"""Shared configuration, data loading, fold-internal preprocessing, models and
statistics used by every analysis script.

Paths can be overridden with environment variables:
  DATA_DIR  processed matrices and outputs of the primary analysis (default: data/processed)
  RAW_DIR   raw downloads, needed only for the PRISM rebuild       (default: data/raw)
  PUB_DIR   published supplementary tables S1-S7                    (default: data/published_tables)
  OUT_DIR   where results are written                               (default: results)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.environ.get("DATA_DIR", os.path.join(REPO, "data", "processed"))
RAW = os.environ.get("RAW_DIR", os.path.join(REPO, "data", "raw"))
PUB = os.environ.get("PUB_DIR", os.path.join(REPO, "data", "published_tables"))
OUT = os.environ.get("OUT_DIR", os.path.join(REPO, "results"))
os.makedirs(OUT, exist_ok=True)

DRUGS = ["Docetaxel", "Paclitaxel", "Cisplatin", "Gemcitabine", "5-Fluorouracil",
         "Irinotecan", "Nutlin-3a (-)", "Dabrafenib", "Trametinib", "Lapatinib",
         "Erlotinib", "Alpelisib"]
CYTOTOXIC = DRUGS[:6]

# Drug-specific curated biomarkers, chosen from the mechanism of action before
# running the analysis. Expression of every listed gene and copy number of the
# listed amplification drivers are forced into the feature set when the
# unsupervised filter does not select them.
CURATED_RNA = {
    "Docetaxel": ["ABCB1", "ABCG2", "ABCC1", "TUBB3", "MAPT", "BCL2L1"],
    "Paclitaxel": ["ABCB1", "ABCG2", "ABCC1", "TUBB3", "MAPT", "BCL2L1"],
    "Cisplatin": ["SLFN11", "ERCC1", "GSTP1", "ABCC2", "MLH1", "ATP7B"],
    "Gemcitabine": ["SLFN11", "DCK", "RRM1", "RRM2", "SLC29A1", "CDA"],
    "5-Fluorouracil": ["SLFN11", "TYMS", "DPYD", "TYMP", "UMPS", "MTHFR"],
    "Irinotecan": ["SLFN11", "TOP1", "UGT1A1", "ABCG2", "CES2", "ABCB1"],
    "Nutlin-3a (-)": ["MDM2", "CDKN1A", "TP53", "MDM4", "BAX", "ZMAT3"],
    "Dabrafenib": ["BRAF", "DUSP6", "ETV4", "ETV5", "SPRY2", "SPRY4"],
    "Trametinib": ["DUSP6", "ETV4", "ETV5", "SPRY2", "SPRY4", "FOSL1"],
    "Lapatinib": ["ERBB2", "EGFR", "ERBB3", "GRB7", "ESR1", "PTEN"],
    "Erlotinib": ["EGFR", "ERBB3", "AREG", "EREG", "ERBB2", "MET"],
    "Alpelisib": ["PIK3CA", "PTEN", "ESR1", "ERBB2", "ERBB3", "PIK3R1"],
}
CURATED_CN = {
    "Lapatinib": ["ERBB2", "EGFR"], "Erlotinib": ["EGFR", "ERBB2", "MET"],
    "Nutlin-3a (-)": ["MDM2", "MDM4"], "Alpelisib": ["PIK3CA", "ERBB2"],
    "Dabrafenib": ["BRAF"], "Trametinib": ["BRAF", "KRAS"],
}

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.5, min_child_weight=3, tree_method="hist",
                  eval_metric="logloss", n_jobs=1)


def load():
    expr = pd.read_pickle(os.path.join(V2, "expr.pkl"))
    cn = pd.read_pickle(os.path.join(V2, "cn.pkl"))
    mut = pd.read_pickle(os.path.join(V2, "mut.pkl"))
    lin = pd.read_pickle(os.path.join(V2, "lineage.pkl"))
    labels = pd.read_csv(os.path.join(V2, "labels.csv"))
    return expr, cn, mut, lin, labels


def fname(drug):
    return drug.replace(" (-)", "_-").replace(" ", "_")


# ---------------------------------------------------------------- features
def prep(tr, te, expr, cn, mut, lin, drug=None, curated=False):
    """All selection, imputation and scaling fitted on the training ids only."""
    out = {}
    e_tr, e_te = expr.loc[tr], expr.loc[te]
    a = e_tr.values
    mad = pd.Series(np.median(np.abs(a - np.median(a, axis=0)), axis=0), index=expr.columns)
    genes = list(mad.sort_values(ascending=False, kind="stable").index[:1000])
    if curated:
        genes += [g for g in CURATED_RNA[drug] if g in expr.columns and g not in genes]
    mu, sd = e_tr[genes].mean(), e_tr[genes].std().replace(0, 1)
    out["RNA"] = ((e_tr[genes] - mu) / sd, (e_te[genes] - mu) / sd)

    c_tr, c_te = cn.loc[tr], cn.loc[te]
    keep = c_tr.columns[c_tr.notna().any().values]
    med = pd.Series(np.nanmedian(c_tr[keep].values, axis=0), index=keep)
    c_tr, c_te = c_tr[keep].fillna(med), c_te[keep].fillna(med)
    var = pd.Series(c_tr.values.var(axis=0, ddof=1), index=keep)
    cgenes = list(var.sort_values(ascending=False, kind="stable").index[:1000])
    if curated:
        cgenes += [g for g in CURATED_CN.get(drug, []) if g in c_tr.columns and g not in cgenes]
    mu, sd = c_tr[cgenes].mean(), c_tr[cgenes].std().replace(0, 1)
    out["CN"] = ((c_tr[cgenes] - mu) / sd, (c_te[cgenes] - mu) / sd)

    m_tr, m_te = mut.loc[tr], mut.loc[te]
    mk = m_tr.columns[m_tr.values.mean(axis=0) >= 0.02]
    out["MUT"] = (m_tr[mk].astype(float), m_te[mk].astype(float))

    out["Early"] = tuple(pd.concat([out[k][i].add_prefix(k + ":") for k in ("RNA", "MUT", "CN")], axis=1)
                         for i in (0, 1))
    for col, name in (("OncotreeLineage", "Tissue"), ("OncotreePrimaryDisease", "Subtype")):
        cats = sorted(lin.loc[tr, col].dropna().unique())
        f = lambda ids: pd.get_dummies(pd.Categorical(lin.loc[ids, col], categories=cats)).astype(float).set_index(pd.Index(ids))
        out[name] = (f(tr), f(te))
    return out


def xgb(seed):
    return XGBClassifier(random_state=seed, **XGB_PARAMS)


def fit_predict(Xtr, ytr, Xte, seed):
    m = xgb(seed).fit(Xtr.values, ytr)
    return m.predict_proba(Xte.values)[:, 1]


def late_fusion(feats, ytr, seed, inner_k=5):
    """Stacked generalisation: inner OOF predictions of RNA, MUT and CN models
    feed a logistic-regression meta-learner."""
    from sklearn.model_selection import StratifiedKFold
    blocks = ("RNA", "MUT", "CN")
    ntr = len(ytr)
    Z = np.zeros((ntr, 3))
    skf = StratifiedKFold(inner_k, shuffle=True, random_state=seed + 99)
    for itr, ite in skf.split(np.zeros(ntr), ytr):
        for j, b in enumerate(blocks):
            X = feats[b][0]
            Z[ite, j] = fit_predict(X.iloc[itr], ytr[itr], X.iloc[ite], seed)
    meta = LogisticRegression(max_iter=1000).fit(Z, ytr)
    Zte = np.column_stack([fit_predict(feats[b][0], ytr, feats[b][1], seed) for b in blocks])
    return meta.predict_proba(Zte)[:, 1]


# -------------------------------------------------------------- statistics
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong(y, *preds):
    """Fast DeLong (Sun & Xu 2014). Returns AUCs and covariance matrix."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    n = len(y) - m
    P = np.vstack([np.asarray(p)[order] for p in preds])
    k = P.shape[0]
    tx = np.array([_midrank(P[r, :m]) for r in range(k)])
    ty = np.array([_midrank(P[r, m:]) for r in range(k)])
    tz = np.array([_midrank(P[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, S


def delong_ci(y, p, alpha=0.05):
    a, S = delong(y, p)
    se = np.sqrt(S[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return a[0], a[0] - z * se, a[0] + z * se


def delong_paired(y, p1, p2):
    """Difference p1 - p2 with 95% CI and two-sided p."""
    a, S = delong(y, p1, p2)
    d = a[0] - a[1]
    se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    z = d / se if se > 0 else 0.0
    return d, d - 1.96 * se, d + 1.96 * se, 2 * stats.norm.sf(abs(z)), se


def corrected_t(diffs, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test for k-fold x r-repeat CV."""
    d = np.asarray(diffs)
    J = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1.0 / J + n_test / n_train) * var)
    t = d.mean() / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), J - 1)
    h = stats.t.ppf(0.975, J - 1) * se
    return d.mean(), d.mean() - h, d.mean() + h, p


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def boot_auc(y, p, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    out = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, [2.5, 97.5])
