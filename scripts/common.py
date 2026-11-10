import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
RNG = 20271022
CATS = {  # taxonomy category -> ApacheJIT columns
 "C1.1 Change size":            ["la","ld"],
 "C1.2 Dispersion":             ["nf","nd","ns","ent"],
 "C1.3 Change purpose":         ["fix"],
 "C2.1 Change frequency":       ["nuc"],
 "C2.2 Code age":               ["age"],
 "C3.1 Experience":             ["aexp","arexp","asexp"],
 "C3.3 Timing":                 ["hour","dow"],
 "C4.2 Ownership":              ["ndev"],
}
FEATS = [f for v in CATS.values() for f in v]
LOG = ["la","ld","nf","nd","ns","ndev","age","nuc","aexp","arexp","asexp"]
def load(drop_hadoop=False):
    d = pd.read_csv("data/apachejit_total.csv")
    if drop_hadoop: d = d[d.project!="apache/hadoop"]
    t = pd.to_datetime(d.author_date, unit="s", utc=True)
    d["hour"] = t.dt.hour; d["dow"] = t.dt.dayofweek
    d["fix"] = d["fix"].astype(int); d["y"] = d["buggy"].astype(int)
    for c in LOG: d[c] = np.log1p(d[c].clip(lower=0))
    return d.sort_values("author_date").reset_index(drop=True)
def group_perm_importance(model, X, y, groups, n_rep=5, seed=RNG):
    """AUC drop when all columns of a group are permuted jointly (same row shuffle)."""
    rs = np.random.RandomState(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:,1]); out = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(n_rep):
            Xp = X.copy(); idx = rs.permutation(len(Xp))
            Xp[cols] = Xp[cols].values[idx]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:,1]))
        out[g] = float(np.mean(drops))
    return base, out
