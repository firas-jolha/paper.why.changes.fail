import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
r=pd.read_csv("data/redef_features.csv"); q95=float((r.la+r.ld).quantile(0.95)); q50=float((r.la+r.ld).median())
a=pd.read_csv("data/apachejit_total.csv"); a=a[a.project!="apache/hadoop"]
out={"redef_churn_p50":q50,"redef_churn_p95":q95}
def auc(g): return roc_auc_score(g.buggy,g.la) if g.buggy.nunique()==2 else float("nan")
subsets={"All commits":a,"Single-file":a[a.nf==1],f"Single-file, churn <= {q95:.0f}":a[(a.nf==1)&(a.la+a.ld<=q95)],
         "Single-file, churn <= 20, not fix":a[(a.nf==1)&(a.la+a.ld<=20)&(~a.fix)]}
for k,g in subsets.items():
    out[k]=dict(n=len(g),buggy=float(g.buggy.mean()),la_auc=auc(g),zero_la_buggy=int(((g.la==0)&g.buggy).sum()),zero_la=int((g.la==0).sum()))
    print(k,out[k])
json.dump(out,open("results_redef/apachejit_control.json","w"),indent=1)
