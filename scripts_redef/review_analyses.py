import sys, json, numpy as np, pandas as pd
sys.path.insert(0,"scripts")
from common import *
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
out={}
def boot(y,p,B=1000,seed=RNG):
    rs=np.random.RandomState(seed); y=np.asarray(y); p=np.asarray(p); v=[]
    for _ in range(B):
        i=rs.randint(0,len(y),len(y))
        if y[i].min()!=y[i].max(): v.append(roc_auc_score(y[i],p[i]))
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
def gbt(cat_idx): return HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,class_weight="balanced",categorical_features=cat_idx,random_state=RNG)
# ---------- ApacheJIT
d=load(True); tr,te=d[d.year<=2016],d[d.year>=2017]
# (1) FIX circularity: model without FIX; and non-fix commits only
F_nofix=[f for f in FEATS if f!="fix"]; C_nofix={k:v for k,v in CATS.items() if k!="C1.3 Change purpose"}
m=gbt([F_nofix.index("dow")]).fit(tr[F_nofix],tr.y); base,imp=group_perm_importance(m,te[F_nofix],te.y,C_nofix,n_rep=5)
out["apache_nofix_feature"]={"auc":base,"size_imp":imp["C1.1 Change size"],"imp":imp}
trn,ten=tr[tr.fix==0],te[te.fix==0]
m=gbt([F_nofix.index("dow")]).fit(trn[F_nofix],trn.y); base,imp=group_perm_importance(m,ten[F_nofix],ten.y,C_nofix,n_rep=5)
out["apache_nonfix_commits"]={"auc":base,"size_imp":imp["C1.1 Change size"],"n_test":len(ten),"buggy":float(ten.y.mean()),"la_auc":roc_auc_score(ten.y,ten.la)}
print("FIX checks",{k:{kk:round(vv,3) for kk,vv in v.items() if not isinstance(vv,dict)} for k,v in out.items()})
# (2) Random forest, ApacheJIT
rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=-1,random_state=RNG).fit(tr[FEATS],tr.y)
base,imp=group_perm_importance(rf,te[FEATS],te.y,CATS,n_rep=5)
out["apache_rf"]={"auc":base,"prauc":average_precision_score(te.y,rf.predict_proba(te[FEATS])[:,1]),"imp":imp}
print("RF apache",round(base,3),{k:round(v,3) for k,v in imp.items()})
# (3) permutation SDs, GBT ApacheJIT (justify 5 reps)
mg=gbt([FEATS.index("dow")]).fit(tr[FEATS],tr.y); rs=np.random.RandomState(RNG); b0=roc_auc_score(te.y,mg.predict_proba(te[FEATS])[:,1]); sds={}
for g,cols in CATS.items():
    dr=[]
    for _ in range(5):
        Xp=te[FEATS].copy(); idx=rs.permutation(len(Xp)); Xp[cols]=Xp[cols].values[idx]; dr.append(b0-roc_auc_score(te.y,mg.predict_proba(Xp)[:,1]))
    sds[g]=float(np.std(dr))
out["apache_perm_sd_max"]=max(sds.values()); print("max perm SD apache",round(max(sds.values()),4))
# ---------- cutoff sensitivity (ReDef churn percentiles)
a=pd.read_csv("scripts/data/apachejit_total.csv"); a=a[a.project!="apache/hadoop"]
r=pd.read_csv("data/redef_features.csv"); churn=r.la+r.ld
sens={}
for q in [0.90,0.95,0.99]:
    c=float(churn.quantile(q)); s=a[(a.nf==1)&(a.la+a.ld<=c)]
    sens[str(q)]={"cutoff":c,"n":len(s),"buggy":float(s.buggy.mean()),"la_auc":[roc_auc_score(s.buggy,s.la)]+boot(s.buggy,s.la),
                  "zero_la":int((s.la==0).sum()),"zero_la_buggy":int(((s.la==0)&s.buggy).sum())}
    print("cutoff",q,c,len(s),[round(x,3) for x in sens[str(q)]["la_auc"]])
out["cutoff_sensitivity"]=sens
# ---------- ReDef: RF, noise-feature null, detectability
for c in ["la","ld","hunks","fn_loc"]: r[c+"_l"]=np.log1p(r[c])
RC={"C1.1 Change size":["la_l","ld_l","hunks_l"],"C1.3 Change purpose":["fix"],"C2.3 Touched-code size":["fn_loc_l"],"C3.3 Timing":["dow"]}
RF_=[f for v in RC.values() for f in v]
rtr,rte=r[r.split!="test"].copy(),r[r.split=="test"].copy()
rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=-1,random_state=RNG).fit(rtr[RF_],rtr.y)
base,imp=group_perm_importance(rf,rte[RF_],rte.y,RC,n_rep=10)
out["redef_rf"]={"auc":base,"prauc":average_precision_score(rte.y,rf.predict_proba(rte[RF_])[:,1]),"imp":imp}
print("RF redef",round(base,3),{k:round(v,3) for k,v in imp.items()})
# noise-feature null: add a pure-noise column, measure its grouped importance distribution
nulls=[]
for s in range(20):
    rs=np.random.RandomState(1000+s); rtr2,rte2=rtr.copy(),rte.copy(); rtr2["noise"]=rs.normal(size=len(rtr2)); rte2["noise"]=rs.normal(size=len(rte2))
    F2=RF_+["noise"]; mm=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,class_weight="balanced",categorical_features=[F2.index("dow")],random_state=RNG).fit(rtr2[F2],rtr2.y)
    _,im=group_perm_importance(mm,rte2[F2],rte2.y,{"noise":["noise"],"size":RC["C1.1 Change size"],"purpose":["fix"]},n_rep=5,seed=s)
    nulls.append(im)
nd=pd.DataFrame(nulls); out["redef_noise_null"]={"noise_mean":float(nd.noise.mean()),"noise_p95":float(nd.noise.quantile(0.95)),
   "size_mean":float(nd["size"].mean()),"size_min":float(nd["size"].min()),"purpose_mean":float(nd.purpose.mean())}
print("noise null",{k:round(v,4) for k,v in out["redef_noise_null"].items()})
# detectability: CI half-width for LA AUC on ReDef test and all; simulated power to detect AUC 0.64
out["redef_detect"]={"test_ci":boot(rte.y,rte.la),"all_ci":boot(r.y,r.la),"n_test_def":int(rte.y.sum()),"n_test_clean":int((1-rte.y).sum())}
# simulate: binormal scores with true AUC 0.64 at ReDef test sizes, how often CI lower bound > 0.55
from scipy.stats import norm
dmu=np.sqrt(2)*norm.ppf(0.64); hits=0; rs=np.random.RandomState(7)
for _ in range(500):
    y=np.r_[np.ones(317),np.zeros(1027)]; s_=np.r_[rs.normal(dmu,1,317),rs.normal(0,1,1027)]
    if roc_auc_score(y,s_)>0.56: hits+=1
out["redef_detect"]["power_auc064_gt056"]=hits/500
print("detect",out["redef_detect"])
# ---------- FIX coverage per project (Zeppelin, ZooKeeper)
fx=a.groupby("project").agg(fix_share=("fix","mean"),buggy=("buggy","mean"),p_bug_fix=("buggy",lambda s: s[a.loc[s.index,"fix"]].mean()))
out["fix_by_project"]=fx.round(3).to_dict(orient="index"); print(fx.round(3).sort_values("fix_share"))
# ReDef projects list
out["redef_projects"]=sorted(r.project.unique().tolist()); print(len(out["redef_projects"]),out["redef_projects"])
json.dump(out,open("results_redef/review_analyses.json","w"),indent=1)
