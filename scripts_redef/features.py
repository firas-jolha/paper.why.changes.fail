import json, re, difflib, pandas as pd, numpy as np
rows=json.load(open("data/redef_all.json"))
FIX=re.compile(r"\b(fix(es|ed|ing)?|bug|crash|leak|regression|overflow|null|segfault|incorrect|wrong|broken|error)\b", re.I)
recs=[]
for r in rows:
    b=r["function_before"].splitlines(); a=r["function_after"].splitlines()
    la=ld=hunks=0
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,b,a,autojunk=False).get_opcodes():
        if tag=="equal": continue
        hunks+=1; ld+=i2-i1; la+=j2-j1
    subj=r["commit_message"].strip().splitlines()[0] if r["commit_message"].strip() else ""
    recs.append(dict(project=r["project"],split=r["split"],date=r["date"],y=int(r["defective_modification"]),
        la=la,ld=ld,hunks=hunks,fn_loc=len(b),
        fix=int(bool(FIX.search(r["commit_message"]))),
        dow=pd.Timestamp(r["date"]).dayofweek, commit_type=r["commit_type"]))
d=pd.DataFrame(recs); d.to_csv("data/redef_features.csv",index=False)
print(d.groupby("y")[["la","ld","hunks","fn_loc"]].median())
print("zero-added share by label", d.groupby("y").apply(lambda g:(g.la==0).mean()).round(3).to_dict())
print("P(def|fix)", d.groupby("fix").y.mean().round(3).to_dict(), "fix share", d.fix.mean().round(3))
print("P(fix|commit_type)", d.groupby("commit_type").fix.mean().round(3).to_dict())
print(d.groupby("split").date.agg(["min","max"]))
