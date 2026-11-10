import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
 "axes.edgecolor":"#8a8a85","axes.labelcolor":"#3d3d3a","xtick.color":"#3d3d3a","ytick.color":"#3d3d3a"})
BLUE,ORANGE,INK="#2a78d6","#eb6834","#1f1f1d"
R=json.load(open("results_redef/redef_results.json")); C=json.load(open("results_redef/apachejit_control.json"))
M=json.load(open("results/main.json")); PR=json.load(open("results/prauc.json"))
r=pd.read_csv("data/redef_features.csv")
a=pd.read_csv("data/apachejit_total.csv"); a=a[a.project!="apache/hadoop"]
# Figure: ECDF of lines added, defective vs clean, two panels
fig,ax=plt.subplots(1,2,figsize=(7.4,2.9),sharey=True)
for k,(df,lab,ycol,title,auc) in enumerate([(a,"ApacheJIT (SZZ labels)","buggy","(a) ApacheJIT: SZZ-based labels",C["All commits"]["la_auc"]),
                                            (r,"ReDef (revert-based labels)","y","(b) ReDef: revert-based labels",R["la_auc_all"])]):
    for val,col,name,ls in [(1,ORANGE,"Defect-inducing","-"),(0,BLUE,"Clean","--")]:
        x=np.sort(df.loc[df[ycol].astype(int)==val,"la"].values); y=np.arange(1,len(x)+1)/len(x)
        ax[k].step(np.log10(x+1),y,where="post",color=col,lw=2,ls=ls,label=name)
    ax[k].set_title(title,loc="left",fontsize=9,color=INK)
    ax[k].set_xlabel("Lines added (log$_{10}$ of LA + 1)")
    ax[k].text(0.98,0.06,f"AUC of LA = {auc:.2f}",transform=ax[k].transAxes,ha="right",fontsize=8.5,color=INK)
    ax[k].grid(color="#e4e3dd",lw=0.6); ax[k].set_axisbelow(True)
ax[0].set_ylabel("Cumulative share of changes"); ax[0].legend(frameon=False,fontsize=8,loc="upper left")
plt.tight_layout(); plt.savefig("results_redef/fig_size_two_datasets.pdf"); plt.savefig("results_redef/fig_size_two_datasets.png",dpi=160)
f=lambda x:f"{x:.3f}"
mj,mr=M["no_hadoop"],R
L=[r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lll}",r"\toprule",
r" & ApacheJIT & ReDef \\",r"\midrule",
r"Labelling & SZZ-based, with filters & Revert commits, GPT-assisted triage \\",
r"Unit & Commit (Java, 14 projects) & Single-function change (C/C++, 22 projects) \\",
f"Changes (defect-inducing) & 94{{,}}710 (29.8\\%) & 13{{,}}432 (23.6\\%) \\\\",
f"Test changes & {mj['n_test']:,} (2017--2019) & {mr['n_test']:,} (latest 10\\% per project) \\\\".replace(",","{,}"),
r"Categories measured & 8 (C1.1--1.3, C2.1--2.2, C3.1, C3.3, C4.2) & 4 (C1.1, C1.3, C2.3, C3.3) \\",
r"\midrule",
f"AUC, logistic regression & {f(mj['LR']['auc'])} & {f(mr['LR']['auc'])} \\\\",
f"AUC, gradient boosting & {f(mj['GBT']['auc'])} & {f(mr['GBT']['auc'])} \\\\",
f"PR-AUC, gradient boosting (base rate) & {f(PR['GBT'])} ({f(mj['buggy_test'])}) & {f(mr['GBT']['prauc'])} ({f(mr['def_test'])}) \\\\",
f"Importance of size, AUC drop & {f(mj['GBT']['cat_imp']['C1.1 Change size'])} & {f(mr['GBT']['imp']['C1.1 Change size'])} \\\\",
f"Importance of purpose, AUC drop & {f(mj['GBT']['cat_imp']['C1.3 Change purpose'])} & {f(mr['GBT']['imp']['C1.3 Change purpose'])} \\\\",
r"\midrule",
f"AUC of lines added alone, test & {f(mj['GBT']['auc']*0+0.805)} & {f(R['single_auc_test']['la'])} [{R['la_auc_ci'][0]:.3f}, {R['la_auc_ci'][1]:.3f}] \\\\",
f"Median lines added, defective / clean & {int(a[a.buggy].la.median())} / {int(a[~a.buggy].la.median())} & {int(r[r.y==1].la.median())} / {int(r[r.y==0].la.median())} \\\\",
f"Defective among changes with no added lines & 0 of {int((a.la==0).sum()):,} & {int(((r.la==0)&(r.y==1)).sum())} of {int((r.la==0).sum())} \\\\".replace(",","{,}"),
f"Defective rate, fix vs.\\ other changes & 72.6\\% vs.\\ 24.4\\% & {100*r[r.fix==1].y.mean():.1f}\\% vs.\\ {100*r[r.fix==0].y.mean():.1f}\\% \\\\",
r"\botrule",r"\end{tabular*}"]
open("results_redef/tab_two_datasets.tex","w").write("\n".join(L))
print("\n".join(L))
print("apache buggy all-years rate", a.buggy.mean())
