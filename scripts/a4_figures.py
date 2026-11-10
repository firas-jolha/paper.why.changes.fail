import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":"#8a8a85","axes.labelcolor":"#3d3d3a","xtick.color":"#3d3d3a","ytick.color":"#3d3d3a"})
BLUE="#2a78d6"; INK="#1f1f1d"; MUTED="#6b6b66"
M=json.load(open("results/main.json")); C=json.load(open("results/cond_stability.json"))
TP=json.load(open("results/time_project.json"))
cats=list(M["no_hadoop"]["GBT"]["cat_imp"].keys())
boot=pd.DataFrame(C["boot"]["cat_vals"])
unc=pd.Series(M["no_hadoop"]["GBT"]["cat_imp"]).sort_values()
lo,hi=boot.quantile(0.025),boot.quantile(0.975)
cond=pd.Series({k:v[0] for k,v in C["conditional"]["conditional"].items()}).sort_values()
csd=pd.Series({k:v[1] for k,v in C["conditional"]["conditional"].items()})
fig,ax=plt.subplots(1,2,figsize=(7.4,3.0),gridspec_kw={"width_ratios":[1.15,1]})
y=np.arange(len(unc))
ax[0].barh(y,unc.values,color=BLUE,height=0.62)
ax[0].errorbar(unc.values,y,xerr=[unc.values-lo[unc.index].values,hi[unc.index].values-unc.values],fmt="none",ecolor=INK,elinewidth=0.8,capsize=2)
ax[0].set_yticks(y,unc.index); ax[0].set_xlabel("AUC drop when category is permuted")
ax[0].set_title("(a) All categories",loc="left",fontsize=9,color=INK)
for yi,v in zip(y,unc.values): ax[0].text(max(v,0)+0.006,yi,f"{v:.3f}",va="center",fontsize=7.5,color=MUTED)
ax[0].set_xlim(-0.005,0.25)
y2=np.arange(len(cond))
ax[1].barh(y2,cond.values,color=BLUE,height=0.62)
ax[1].errorbar(cond.values,y2,xerr=csd[cond.index].values,fmt="none",ecolor=INK,elinewidth=0.8,capsize=2)
ax[1].set_yticks(y2,cond.index); ax[1].set_xlabel("AUC drop, permuted within size × fix strata")
ax[1].set_title("(b) Beyond size and purpose",loc="left",fontsize=9,color=INK)
for yi,v in zip(y2,cond.values): ax[1].text(v+0.0006,yi,f"{v:.4f}",va="center",fontsize=7.5,color=MUTED)
ax[1].set_xlim(0,0.0165)
for a in ax: a.grid(axis="x",color="#e4e3dd",lw=0.6); a.set_axisbelow(True)
plt.tight_layout(); plt.savefig("results/fig_importance.pdf"); plt.savefig("results/fig_importance.png",dpi=160)
# heatmaps: share of positive importance
def share(dct):
    df=pd.DataFrame(dct).T.clip(lower=0); return (df.div(df.sum(axis=1),axis=0)*100)[cats]
T=share({y:v["imp"] for y,v in TP["time"].items()})
P=share({p.split("/")[1]:v["imp"] for p,v in TP["project"].items()})
cmap=LinearSegmentedColormap.from_list("b",["#f4f8fd","#cde2fb","#86b6ef","#3987e5","#1c5cab","#0d366b"])
fig,ax=plt.subplots(1,2,figsize=(8.2,4.4),gridspec_kw={"width_ratios":[1,1.15]})
fig.subplots_adjust(wspace=0.62)
short=[c.split(" ",1)[1] for c in cats]
for a,df,ttl in [(ax[0],T,"(a) Test year (train: preceding 3 years)"),(ax[1],P,"(b) Project (chronological 70/30 split)")]:
    im=a.imshow(df.values,cmap=cmap,vmin=0,vmax=100,aspect="auto")
    a.set_xticks(range(len(short)),short,rotation=40,ha="right"); a.set_yticks(range(len(df)),df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v=df.values[i,j]; a.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=6.5,color="white" if v>45 else INK)
    a.set_title(ttl,loc="left",fontsize=9,color=INK)
    for s in a.spines.values(): s.set_visible(False)
    a.tick_params(length=0)
cb=fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02); cb.set_label("Share of total importance (%)"); cb.outline.set_visible(False)
plt.savefig("results/fig_heatmaps.pdf",bbox_inches="tight"); plt.savefig("results/fig_heatmaps.png",dpi=160,bbox_inches="tight")
T.round(1).to_csv("results/share_years.csv"); P.round(1).to_csv("results/share_projects.csv")
print(T.round(1)); print(P.round(1))
