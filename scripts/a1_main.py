import json, numpy as np, pandas as pd, time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score
from scipy.stats import spearmanr
from common import *
res = {}
for tag, dh in [("all", False), ("no_hadoop", True)]:
    d = load(dh); tr, te = d[d.year<=2016], d[d.year>=2017]
    Xtr, ytr, Xte, yte = tr[FEATS], tr.y, te[FEATS], te.y
    models = {
      "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
      "GBT": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                            categorical_features=[FEATS.index("dow")], random_state=RNG)}
    r = {"n_train": len(tr), "n_test": len(te), "buggy_train": float(ytr.mean()), "buggy_test": float(yte.mean())}
    for name, m in models.items():
        t0=time.time(); m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:,1]; yh = (p>=0.5).astype(int)
        base, gimp = group_perm_importance(m, Xte, yte, CATS)
        _, mimp = group_perm_importance(m, Xte, yte, {f:[f] for f in FEATS})
        r[name] = {"auc": roc_auc_score(yte,p), "mcc": matthews_corrcoef(yte,yh), "f1": f1_score(yte,yh),
                   "cat_imp": gimp, "metric_imp": mimp, "secs": time.time()-t0}
        if name=="LR":
            co = m[-1].coef_[0]; r[name]["coef"] = dict(zip(FEATS, map(float, co)))
        print(tag, name, round(r[name]["auc"],3), round(r[name]["mcc"],3), round(time.time()-t0,1), flush=True)
    res[tag] = r
    if tag=="all":
        rho = spearmanr(tr[FEATS]).statistic
        res["spearman"] = pd.DataFrame(rho, index=FEATS, columns=FEATS).round(3).to_dict()
json.dump(res, open("results/main.json","w"), indent=1)
