import sys, json, numpy as np, pandas as pd
sys.path.insert(0,"scripts")
from common import *
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def boot(y,p,B=1000,seed=RNG):
    rs=np.random.RandomState(seed); y=np.asarray(y); p=np.asarray(p); v=[]
    for _ in range(B):
        i=rs.randint(0,len(y),len(y))
        if y[i].min()!=y[i].max(): v.append(roc_auc_score(y[i],p[i]))
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
out={}
# ApacheJIT (no hadoop): GBT and LA-only on 2017-2019 test
d=load(True); tr,te=d[d.year<=2016],d[d.year>=2017]
m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,class_weight="balanced",categorical_features=[FEATS.index("dow")],random_state=RNG).fit(tr[FEATS],tr.y)
p=m.predict_proba(te[FEATS])[:,1]
out["apache_gbt"]=[roc_auc_score(te.y,p)]+boot(te.y,p)
out["apache_la"]=[roc_auc_score(te.y,te.la)]+boot(te.y,te.la)
# ApacheJIT single-file small control (all years)
a=pd.read_csv("scripts/data/apachejit_total.csv"); a=a[a.project!="apache/hadoop"]
s=a[(a.nf==1)&(a.la+a.ld<=29)]; out["apache_small_la"]=[roc_auc_score(s.buggy,s.la)]+boot(s.buggy,s.la)
s1=a[a.nf==1]; out["apache_single_la"]=[roc_auc_score(s1.buggy,s1.la)]+boot(s1.buggy,s1.la)
# ReDef GBT
r=pd.read_csv("data/redef_features.csv")
for c in ["la","ld","hunks","fn_loc"]: r[c+"_l"]=np.log1p(r[c])
F=["la_l","ld_l","hunks_l","fix","fn_loc_l","dow"]
rtr,rte=r[r.split!="test"],r[r.split=="test"]
mr=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,class_weight="balanced",categorical_features=[F.index("dow")],random_state=RNG).fit(rtr[F],rtr.y)
pr=mr.predict_proba(rte[F])[:,1]
out["redef_gbt"]=[roc_auc_score(rte.y,pr)]+boot(rte.y,pr)
out["redef_la"]=[roc_auc_score(rte.y,rte.la)]+boot(rte.y,rte.la)
out["redef_la_all"]=[roc_auc_score(r.y,r.la)]+boot(r.y,r.la)
for k,v in out.items(): print(k,[round(x,3) for x in v])
json.dump(out,open("results_redef/cis.json","w"),indent=1)
