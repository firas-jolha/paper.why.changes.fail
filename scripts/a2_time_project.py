import json, numpy as np, pandas as pd, time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from common import *
def gbt(): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced",
                        categorical_features=[FEATS.index("dow")], random_state=RNG)
d = load(drop_hadoop=True)
out = {"time": {}, "project": {}}
# (a) temporal: train on preceding 3 years, test on the year
for y in range(2009, 2020):
    tr, te = d[(d.year>=y-3)&(d.year<y)], d[d.year==y]
    if te.y.nunique()<2 or tr.y.nunique()<2: continue
    m = gbt().fit(tr[FEATS], tr.y)
    base, imp = group_perm_importance(m, te[FEATS], te.y, CATS, n_rep=3)
    out["time"][y] = {"auc": base, "n_train": len(tr), "n_test": len(te), "buggy_test": float(te.y.mean()), "imp": imp}
    print("year", y, round(base,3), flush=True)
# (b) per project: chronological 70/30 split
for p, g in d.groupby("project"):
    g = g.sort_values("author_date"); k = int(len(g)*0.7); tr, te = g.iloc[:k], g.iloc[k:]
    if te.y.sum()<30 or tr.y.sum()<30: print("skip", p); continue
    m = gbt().fit(tr[FEATS], tr.y)
    base, imp = group_perm_importance(m, te[FEATS], te.y, CATS, n_rep=3)
    out["project"][p] = {"auc": base, "n": len(g), "buggy": float(g.y.mean()), "imp": imp}
    print(p, round(base,3), flush=True)
json.dump(out, open("results/time_project.json","w"), indent=1)
