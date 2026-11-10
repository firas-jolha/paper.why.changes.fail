import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from common import *
def gbt(seed=RNG): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced",
                        categorical_features=[FEATS.index("dow")], random_state=seed)
def kendall_w(R):  # R: m raters x n items of ranks
    m, n = R.shape; S = ((R.sum(0) - m*(n+1)/2)**2).sum(); return 12*S/(m**2*(n**3-n))
d = load(drop_hadoop=True); tr, te = d[d.year<=2016], d[d.year>=2017].reset_index(drop=True)
out = {}
# (1) conditional importance: permute within strata of size decile x fix
m = gbt().fit(tr[FEATS], tr.y); X, y = te[FEATS], te.y
base = roc_auc_score(y, m.predict_proba(X)[:,1])
strata = pd.qcut(te.la + te.ld, 10, labels=False, duplicates="drop").astype(str) + "_" + te.fix.astype(str)
rs = np.random.RandomState(RNG); cond = {}
for g, cols in CATS.items():
    if g in ("C1.1 Change size", "C1.3 Change purpose"): continue
    drops = []
    for _ in range(5):
        Xp = X.copy()
        for s, idx in te.groupby(strata).groups.items():
            idx = np.asarray(idx); Xp.loc[idx, cols] = X.loc[rs.permutation(idx), cols].values
        drops.append(base - roc_auc_score(y, m.predict_proba(Xp)[:,1]))
    cond[g] = [float(np.mean(drops)), float(np.std(drops))]
_, uncond = group_perm_importance(m, X, y, CATS, n_rep=5)
out["conditional"] = {"base_auc": base, "conditional": cond, "unconditional": uncond}
print("conditional", {k: round(v[0],4) for k,v in cond.items()}, flush=True)
# (2) stability: bootstrap resamples of training data
metr = {f:[f] for f in FEATS}; cat_r, met_r, cat_v, met_v = [], [], [], []
for b in range(20):
    s = tr.sample(frac=1.0, replace=True, random_state=RNG+b)
    mb = gbt(RNG+b).fit(s[FEATS], s.y)
    _, ci = group_perm_importance(mb, X, y, CATS, n_rep=2, seed=b)
    _, mi = group_perm_importance(mb, X, y, metr, n_rep=2, seed=b)
    cat_v.append(ci); met_v.append(mi)
    cat_r.append(pd.Series(ci).rank(ascending=False).values); met_r.append(pd.Series(mi).rank(ascending=False).values)
    print("boot", b, flush=True)
out["boot"] = {"W_cat": kendall_w(np.array(cat_r)), "W_metric": kendall_w(np.array(met_r)),
               "cat_vals": cat_v, "metric_vals": met_v}
# (3) stability across years (train on preceding 3 years)
cy, my = [], []
for yr in range(2009, 2020):
    a, bte = d[(d.year>=yr-3)&(d.year<yr)], d[d.year==yr]
    mm = gbt().fit(a[FEATS], a.y)
    _, ci = group_perm_importance(mm, bte[FEATS], bte.y, CATS, n_rep=2)
    _, mi = group_perm_importance(mm, bte[FEATS], bte.y, metr, n_rep=2)
    cy.append(pd.Series(ci).rank(ascending=False).values); my.append(pd.Series(mi).rank(ascending=False).values)
    print("year", yr, flush=True)
out["years"] = {"W_cat": kendall_w(np.array(cy)), "W_metric": kendall_w(np.array(my))}
# (4) across projects
cp, mp = [], []
for p, g in d.groupby("project"):
    g = g.sort_values("author_date"); k = int(len(g)*0.7); a, bte = g.iloc[:k], g.iloc[k:]
    if bte.y.sum()<30 or a.y.sum()<30: continue
    mm = gbt().fit(a[FEATS], a.y)
    _, ci = group_perm_importance(mm, bte[FEATS], bte.y, CATS, n_rep=2)
    _, mi = group_perm_importance(mm, bte[FEATS], bte.y, metr, n_rep=2)
    cp.append(pd.Series(ci).rank(ascending=False).values); mp.append(pd.Series(mi).rank(ascending=False).values)
out["projects"] = {"W_cat": kendall_w(np.array(cp)), "W_metric": kendall_w(np.array(mp)), "n": len(cp)}
print({k: (round(out[k]["W_cat"],3), round(out[k]["W_metric"],3)) for k in ["boot","years","projects"]})
json.dump(out, open("results/cond_stability.json","w"), indent=1)
