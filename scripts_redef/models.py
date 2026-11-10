import json, numpy as np, pandas as pd, sys
sys.path.insert(0,"scripts")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score
from common import group_perm_importance, RNG
d=pd.read_csv("data/redef_features.csv")
for c in ["la","ld","hunks","fn_loc"]: d[c]=np.log1p(d[c])
CATS={"C1.1 Change size":["la","ld","hunks"],"C1.3 Change purpose":["fix"],
      "C2.3 Touched-code size":["fn_loc"],"C3.3 Timing":["dow"]}
F=[f for v in CATS.values() for f in v]
tr,te=d[d.split!="test"],d[d.split=="test"]   # train+valid -> earlier commits; test = latest 10% per project
out={"n_train":len(tr),"n_test":len(te),"def_test":float(te.y.mean())}
for name,m in {"LR":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")),
               "GBT":HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,class_weight="balanced",
                     categorical_features=[F.index("dow")],random_state=RNG)}.items():
    m.fit(tr[F],tr.y); p=m.predict_proba(te[F])[:,1]; yh=(p>=0.5).astype(int)
    base,imp=group_perm_importance(m,te[F],te.y,CATS,n_rep=10)
    out[name]=dict(auc=roc_auc_score(te.y,p),prauc=average_precision_score(te.y,p),
                   mcc=matthews_corrcoef(te.y,yh),f1=f1_score(te.y,yh),imp=imp)
    print(name,{k:round(v,3) for k,v in out[name].items() if k!="imp"},{k:round(v,4) for k,v in imp.items()})
# single-feature AUCs on test (raw direction)
single={c:roc_auc_score(te.y,te[c]) for c in ["la","ld","hunks","fn_loc","fix"]}
single["la+ld"]=roc_auc_score(te.y,te.la+te.ld)
out["single_auc_test"]=single; print({k:round(v,3) for k,v in single.items()})
# bootstrap CI for la-only AUC on test, and on full data
rs=np.random.RandomState(RNG); bs=[]
for _ in range(1000):
    i=rs.randint(0,len(te),len(te)); bs.append(roc_auc_score(te.y.values[i],te.la.values[i]))
out["la_auc_ci"]=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
out["la_auc_all"]=roc_auc_score(d.y,d.la)
print("la AUC test CI",np.round(out["la_auc_ci"],3),"all",round(out["la_auc_all"],3))
# per project la AUC
pp={p:roc_auc_score(g.y,g.la) for p,g in d.groupby("project") if g.y.nunique()==2 and g.y.sum()>=20}
out["la_auc_project"]=pp; print("projects",len(pp),"median",round(np.median(list(pp.values())),3),"range",round(min(pp.values()),3),round(max(pp.values()),3))
json.dump(out,open("results_redef/redef_results.json","w"),indent=1)
