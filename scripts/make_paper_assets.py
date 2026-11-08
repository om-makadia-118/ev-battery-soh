"""Regenerate every table, figure and number used in paper/paper.tex.

Run from the repository root:  python scripts/make_paper_assets.py
Reads results/full_study/*/metrics.csv, adaptation_metrics.csv and
data/processed/{unified,data_quality}.csv. Writes paper/figs/*.pdf (also copied
to Figure_1..Figure_5.pdf), paper/tabs/*.tex and paper/numbers.json.
"""
import os, shutil, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

R="results/full_study"; U="data/processed/unified.csv"
OUT="paper"
plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral"],"mathtext.fontset":"stix",
 "font.size":7.5,"axes.labelsize":7.5,"axes.titlesize":8,"legend.fontsize":6.5,"xtick.labelsize":7,"ytick.labelsize":7,
 "axes.linewidth":0.6,"pdf.fonttype":42,"savefig.bbox":"tight","savefig.pad_inches":0.02})
os.makedirs(f"{OUT}/figs",exist_ok=True); os.makedirs(f"{OUT}/tabs",exist_ok=True)
C={"nasa":"#0072B2","calce":"#E69F00","xjtu":"#009E73","oxford":"#CC79A7","matr":"#8A8A8A"}
NM={"nasa":"NASA","calce":"CALCE","xjtu":"XJTU","oxford":"Oxford","matr":"MATR"}
ORD=["nasa","calce","xjtu","oxford","matr"]
MD=["mean","ridge","elasticnet","xgboost"]; ML={"mean":"Mean","ridge":"Ridge","elasticnet":"ElasticNet","xgboost":"XGBoost"}
u=pd.read_csv(U,low_memory=False); u["soh"]=100*u.capacity_ah/u.rated_capacity_ah
dq=pd.read_csv("data/processed/data_quality.csv").set_index("dataset")
mx=pd.read_csv(f"{R}/matrix/metrics.csv"); lo=pd.read_csv(f"{R}/lodo/metrics.csv"); ad=pd.read_csv(f"{R}/adaptation/adaptation_metrics.csv")
NUM={}
def pm(m,s,d=2): return f"{m:.{d}f}\\,$\\pm$\\,{s:.{d}f}"
def best(vals,i): return min(range(len(vals)),key=lambda j:vals[j])==i

# ---------- Table I
rows=[]
meta={"nasa":("LCO/graphite","18650","2.0","24"),"calce":("LCO/graphite","Prismatic","1.1","RT"),
 "xjtu":("NCM/graphite","18650","2.0","RT"),"oxford":("LCO/graphite","Pouch","0.74","40"),"matr":("LFP/graphite","18650","1.1","30")}
for d in ORD:
    x=u[u.dataset==d]; g=x.soh
    rows.append(f"{NM[d]} & {meta[d][0]} & {meta[d][1]} & {meta[d][2]} & {meta[d][3]} & {x.battery_id.nunique()} & {len(x):,} & {int(x.cycle.max()):,} & {int(dq.loc[d,'ever_eol_batteries'])} & {g.mean():.1f}\\,$\\pm$\\,{g.std():.1f} \\\\")
tot=f"Total & & & & & {u.battery_id.nunique()} & {len(u):,} & & {int(dq.ever_eol_batteries.sum())} & \\\\"
open(f"{OUT}/tabs/t1.tex","w").write("\n".join(rows)+"\n\\midrule\n"+tot+"\n")
NUM["cells"]=int(u.battery_id.nunique()); NUM["rows"]=int(len(u))

# ---------- Table II within
w=mx[mx.experiment=="within"]
L=[]
for d in ORD:
    s=w[w.target==d].groupby("model").MAE.agg(["mean","std"]); v=[s.loc[m,"mean"] for m in MD]
    cells=[(f"\\textbf{{{pm(s.loc[m,'mean'],s.loc[m,'std'])}}}" if best(v,i) else pm(s.loc[m,'mean'],s.loc[m,'std'])) for i,m in enumerate(MD)]
    L.append(f"{NM[d]} & "+" & ".join(cells)+" \\\\")
s=w.groupby("model").MAE.agg(["mean","std"]); v=[s.loc[m,"mean"] for m in MD]
cells=[(f"\\textbf{{{pm(s.loc[m,'mean'],s.loc[m,'std'])}}}" if best(v,i) else pm(s.loc[m,'mean'],s.loc[m,'std'])) for i,m in enumerate(MD)]
open(f"{OUT}/tabs/t2.tex","w").write("\n".join(L)+"\n\\midrule\nAll (25 runs) & "+" & ".join(cells)+" \\\\\n")
NUM["within"]={m:[float(s.loc[m,"mean"]),float(s.loc[m,"std"])] for m in MD}

# ---------- Table III cross
c=mx[mx.experiment=="cross"]; pairs=[(s_,t_) for s_ in ORD for t_ in ORD if s_!=t_]
blocks=[[],[]]
agg=c.groupby(["source","target","model"]).MAE.agg(["mean","std"])
wins={"ridge":0,"xgboost":0}
for k,(s_,t_) in enumerate(pairs):
    v=[agg.loc[(s_,t_,m),"mean"] for m in ["mean","ridge","xgboost"]]
    if v[1]<v[0]: wins["ridge"]+=1
    if v[2]<v[0]: wins["xgboost"]+=1
    cells=[]
    for i,m in enumerate(["mean","ridge","xgboost"]):
        mu,sd=agg.loc[(s_,t_,m),"mean"],agg.loc[(s_,t_,m),"std"]
        f=(f"{mu:.1f}\\,$\\pm$\\,{sd:.1f}" if mu<100 else f"{mu:.0f}\\,$\\pm$\\,{sd:.0f}")
        cells.append(f"\\textbf{{{f}}}" if best(v,i) else f)
    blocks[k//10].append(f"{NM[s_]}$\\to${NM[t_]} & "+" & ".join(cells))
T3=[f"{a} & {b} \\\\" for a,b in zip(blocks[0],blocks[1])]
open(f"{OUT}/tabs/t3.tex","w").write("\n".join(T3)+"\n")
NUM["wins"]=wins
NUM["cross"]={m:[float(c[c.model==m].MAE.mean()),float(c[c.model==m].MAE.std())] for m in MD}
NUM["ridge_gt100"]=int(sum(agg.loc[(s_,t_,"ridge"),"mean"]>100 for s_,t_ in pairs))

# ---------- Table IV LODO
L=[]
lg=lo.groupby(["target","model"]).MAE.agg(["mean","std"]); tr=lo.groupby("target").train_batteries.first()
for d in ORD:
    v=[lg.loc[(d,m),"mean"] for m in MD]
    cells=[(f"\\textbf{{{pm(*lg.loc[(d,m)].values)}}}" if best(v,i) else pm(*lg.loc[(d,m)].values)) for i,m in enumerate(MD)]
    L.append(f"{NM[d]} & {int(tr[d])} & "+" & ".join(cells)+" \\\\")
s=lo.groupby("model").MAE.agg(["mean","std"]); v=[s.loc[m,"mean"] for m in MD]
cells=[(f"\\textbf{{{pm(s.loc[m,'mean'],s.loc[m,'std'])}}}" if best(v,i) else pm(s.loc[m,'mean'],s.loc[m,'std'])) for i,m in enumerate(MD)]
open(f"{OUT}/tabs/t4.tex","w").write("\n".join(L)+"\n\\midrule\nAll (25 runs) & & "+" & ".join(cells)+" \\\\\n")
NUM["lodo"]={m:[float(s.loc[m,"mean"]),float(s.loc[m,"std"])] for m in MD}
NUM["lodo_train"]={d:int(tr[d]) for d in ORD}

# ---------- Table V few-shot (targets feasible for every k)
f=ad[(ad.method!="not_feasible")].copy(); TG=["calce","matr","xjtu"]; ks=[0,1,3,5,10]
def macro(model,method,k):
    d=f[(f.model==model)&(f.method==method)&(f.k==k)&(f.target.isin(TG))]
    return d.groupby("target").MAE.mean().mean()
rowsdef=[("Mean (target mean)","mean","scratch"),("Ridge, from scratch","ridge","scratch"),("Ridge, pooled","ridge","pooled"),
 ("XGBoost, from scratch","xgboost","scratch"),("XGBoost, pooled","xgboost","pooled"),("XGBoost, CORAL","xgboost","coral"),("XGBoost, mean--var. norm.","xgboost","normalization")]
L=[];FS={}
zs={m:macro(m,"none",0) for m in ["mean","ridge","xgboost"]}
L.append("Zero-shot (k=0): Mean / Ridge / XGBoost & \\multicolumn{5}{c}{"+f"{zs['mean']:.2f} / {zs['ridge']:.2f} / {zs['xgboost']:.2f}"+"} \\\\")
L=[]
for lab,m,me in rowsdef:
    vals=[]
    for k in ks:
        if k==0: vals.append(zs[m] if m in zs else np.nan)
        else: vals.append(macro(m,me,k))
    FS[lab]=vals
    L.append(f"{lab} & "+" & ".join("--" if np.isnan(x) else f"{x:.2f}" for x in vals)+" \\\\")
open(f"{OUT}/tabs/t5.tex","w").write("\n".join(L)+"\n")
NUM["fewshot"]={k:[None if np.isnan(x) else float(x) for x in v] for k,v in FS.items()}
inf=ad[ad.method=="not_feasible"]
NUM["infeasible"]=int(len(inf)); NUM["executed"]=int(len(ad)+len(mx)+len(lo)-len(inf))
NUM["total_logged"]=int(len(ad)+len(mx)+len(lo))
cn=f[f.method.isin(["coral","normalization"])].pivot_table(index=["source","target","model","seed","k"],columns="method",values="MAE")
NUM["coral_norm_meanabsdiff"]=float((cn.coral-cn.normalization).abs().mean()); NUM["coral_norm_max"]=float((cn.coral-cn.normalization).abs().max())

# sensitivity without CALCE and low-SOH stats
nc=c[(c.source!="calce")&(c.target!="calce")].groupby(["source","target","model"]).MAE.mean().unstack()
NUM["noCALCE_pairs"]=int(len(nc)); NUM["noCALCE_means"]={m:float(nc[m].mean()) for m in ["mean","ridge","xgboost"]}
NUM["noCALCE_wins"]={"xgboost":int((nc.xgboost<nc["mean"]).sum()),"ridge":int((nc.ridge<nc["mean"]).sum())}
cl=u[u.dataset=="calce"]; NUM["calce_lt50"]=[int((cl.soh<50).sum()),int(len(cl))]; NUM["calce_lt30"]=int((cl.soh<30).sum())
rows_by=u.groupby("dataset").size(); NUM["matr_row_share"]=float(rows_by["matr"]/len(u))
NUM["eol_cells"]={d:int(dq.loc[d,"ever_eol_batteries"]) for d in ORD}
# ---------- Fig 1 heterogeneity
fig,ax=plt.subplots(1,2,figsize=(7.16,2.5))
for d in ["matr","calce","xjtu","oxford","nasa"]:
    for j,(b,x) in enumerate(u[u.dataset==d].groupby("battery_id")):
        x=x.sort_values("cycle")
        if len(x)>400: x=x.iloc[::max(1,len(x)//250)]
        ax[0].plot(np.maximum(x.cycle,1),x.soh,color=C[d],lw=0.4,alpha=0.5,label=NM[d] if j==0 else None)
ax[0].set_xscale("log"); ax[0].set_xlabel("Cycle index (log scale)"); ax[0].set_ylabel("SOH (%)"); ax[0].set_ylim(0,115)
hh,ll=ax[0].get_legend_handles_labels(); od=[ll.index(NM[d]) for d in ORD]; ax[0].legend([hh[i] for i in od],[ll[i] for i in od],ncol=5,loc="lower left",frameon=False,handlelength=1.2,columnspacing=0.8); ax[0].set_title("(a) SOH trajectories of all 190 cells",loc="left")
for d in ORD:
    x=u[u.dataset==d]; ax[1].hist(x.soh.clip(0,115),bins=np.arange(0,116,2.5),weights=np.ones(len(x))/len(x)*100,histtype="step",color=C[d],lw=1,label=NM[d])
ax[1].set_xlabel("SOH (%)"); ax[1].set_ylabel("Share of observations (%)"); ax[1].set_title("(b) SOH distribution per dataset",loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/figs/fig1_data.pdf"); plt.close()

# ---------- Fig 2 protocol schematic
fig,ax=plt.subplots(figsize=(7.16,1.9)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,26)
def box(x,y,w_,h,t,fc="#F2F2F2",fs=6.8):
    ax.add_patch(FancyBboxPatch((x,y),w_,h,boxstyle="round,pad=0.25,rounding_size=0.8",fc=fc,ec="#333",lw=0.6)); ax.text(x+w_/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arr(x1,y1,x2,y2): ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=0.7,color="#333"))
box(1,8,15,10,"Five public\ndatasets\n(190 cells,\n132,389 obs.)","#DCE9F5"); arr(16.5,13,20.5,13)
box(21,8,17,10,"Harmonisation\nSOH = $Q_t/Q_{rated}$\u00d7100\ncycle index only","#FBEBC8"); arr(38.5,13,42.5,13)
box(43,8,16,10,"Battery-disjoint\nsplits (cells never\nshared across\ntrain / test)","#DFF0E3"); arr(59.5,13,63.5,13)
for i,(t) in enumerate(["P1 Within-dataset","P2 Pairwise 5\u00d75 transfer","P3 Leave-one-dataset-out","P4 Few-shot (k=0,1,3,5,10)"]):
    box(64,19.2-i*6.1,22,4.6,t,fs=6.4)
arr(59.5,13,63.4,21.5); arr(59.5,13,63.4,3.2)
box(88,8,11.5,10,"Models: Mean,\nRidge, ElasticNet,\nXGBoost\n5 seeds \u2192 MAE","#F6DCE6",fs=6.3)
for i in range(4): arr(86.4,21.5-i*6.1,87.9,13)
plt.savefig(f"{OUT}/figs/fig2_protocol.pdf"); plt.close()

# ---------- Fig 3 heatmaps
def mat(m):
    M=np.full((5,5),np.nan)
    for i,s_ in enumerate(ORD):
        for j,t_ in enumerate(ORD):
            if s_==t_: M[i,j]=w[(w.target==t_)&(w.model==m)].MAE.mean()
            else: M[i,j]=agg.loc[(s_,t_,m),"mean"]
    return M
Mx,Mm=mat("xgboost"),mat("mean"); G=Mm-Mx
fig,ax=plt.subplots(1,2,figsize=(7.16,2.9))
im=ax[0].imshow(np.minimum(Mx,40),cmap="viridis_r",vmin=0,vmax=40)
for i in range(5):
    for j in range(5): ax[0].text(j,i,f"{Mx[i,j]:.1f}",ha="center",va="center",fontsize=7,color="white" if Mx[i,j]<20 else "black",fontweight="bold" if i==j else None)
ax[0].set_title("(a) XGBoost test MAE (pp)",loc="left"); plt.colorbar(im,ax=ax[0],fraction=0.046,pad=0.03)
im=ax[1].imshow(np.clip(G,-12,12),cmap="RdBu",vmin=-12,vmax=12)
for i in range(5):
    for j in range(5):
        if i==j: ax[1].text(j,i,"--",ha="center",va="center",fontsize=7)
        else: ax[1].text(j,i,f"{G[i,j]:+.1f}",ha="center",va="center",fontsize=7)
ax[1].set_title("(b) Mean-baseline MAE minus XGBoost MAE (pp)",loc="left"); plt.colorbar(im,ax=ax[1],fraction=0.046,pad=0.03)
for a in ax:
    a.set_xticks(range(5)); a.set_yticks(range(5)); a.set_xticklabels([NM[d] for d in ORD],rotation=30); a.set_yticklabels([NM[d] for d in ORD])
    a.set_xlabel("Target"); a.set_ylabel("Source")
plt.tight_layout(); plt.savefig(f"{OUT}/figs/fig3_matrix.pdf"); plt.close()

# ---------- Fig 4 extrapolation diagnosis
n=u[u.dataset=="nasa"]; o=u[u.dataset=="oxford"]
rg=Ridge(alpha=1.0).fit(n[["cycle"]],n.soh); NUM["nasa_slope"]=float(rg.coef_[0]); NUM["nasa_intercept"]=float(rg.intercept_)
import xgboost as xgb
xg=xgb.XGBRegressor(n_estimators=100,max_depth=4,learning_rate=0.05,subsample=0.8,random_state=11).fit(n[["cycle"]],n.soh)
gx=np.arange(0,8201,50)[:,None]
NUM["ridge_pred_8200"]=float(rg.predict([[8200]])[0]); NUM["xgb_pred_8200"]=float(xg.predict(np.array([[8200.]]))[0])
cm=u.groupby("dataset").cycle.max(); NUM["cmax"]={d:int(cm[d]) for d in ORD}
fig,ax=plt.subplots(1,2,figsize=(7.16,2.7))
ax[0].scatter(o.cycle,o.soh,s=2.5,color="#444",alpha=.55,label="Oxford observations")
ax[0].plot(gx[:,0],rg.predict(gx),color="#D55E00",lw=1.3,label="Ridge fitted on NASA")
ax[0].plot(gx[:,0],xg.predict(gx),color="#0072B2",lw=1.3,label="XGBoost fitted on NASA")
ax[0].axvspan(0,168,color="#0072B2",alpha=.12,lw=0); ax[0].text(260,-1250,"NASA training\nrange (\u2264168)",fontsize=6.2,color="#0072B2")
ax[0].axhline(0,color="k",lw=.4,ls=":"); ax[0].set_xlabel("Cycle index"); ax[0].set_ylabel("Predicted / observed SOH (%)")
ax[0].legend(frameon=False,loc="center right",fontsize=6.2); ax[0].set_title("(a) Zero-shot NASA$\\to$Oxford extrapolation",loc="left")
cc=c.groupby(["source","target","model"]).MAE.mean().unstack().reset_index()
cc["ratio"]=[cm[t]/cm[s] for s,t in zip(cc.source,cc.target)]
for m,col,mk in [("mean","#777","s"),("ridge","#D55E00","o"),("xgboost","#0072B2","^")]:
    rho,p=spearmanr(cc.ratio,cc[m]); NUM[f"rho_{m}"]=[float(rho),float(p)]
    ax[1].scatter(cc.ratio,cc[m],s=14,color=col,marker=mk,alpha=.85,label=f"{ML[m]} ($\\rho$={rho:.2f})")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].axvline(1,color="k",lw=.4,ls=":")
ax[1].set_xlabel("Target / source maximum cycle index"); ax[1].set_ylabel("Zero-shot MAE (pp)"); ax[1].legend(frameon=False,fontsize=6.2,loc="upper left")
ax[1].set_title("(b) Error vs. cycle-range mismatch (20 pairs)",loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/figs/fig4_extrapolation.pdf"); plt.close()

# ---------- Fig 5 few-shot per target
fig,axs=plt.subplots(2,3,figsize=(7.16,4.0)); axs=axs.ravel()
wd=w[w.model=="xgboost"].groupby("target").MAE.mean()
for ax_,t_ in zip(axs,ORD):
    d=f[(f.target==t_)]
    def cur(model,method,kk): 
        return [d[(d.model==model)&(d.method==method)&(d.k==k)].MAE.mean() for k in kk]
    kk=[1,3,5,10]
    z=d[(d.model=="xgboost")&(d.method=="none")].MAE.mean()
    ax_.axhline(z,color="#777",ls="--",lw=.9,label="XGBoost zero-shot"); ax_.axhline(wd[t_],color="#009E73",ls=":",lw=1,label="XGBoost within-dataset")
    for model,method,col,mk,lab in [("mean","scratch","#777","s","Mean (target cells)"),("xgboost","scratch","#0072B2","o","XGBoost scratch"),("xgboost","pooled","#E69F00","^","XGBoost pooled"),("xgboost","coral","#CC79A7","v","XGBoost CORAL")]:
        y=cur(model,method,kk); ax_.plot(kk,y,color=col,marker=mk,ms=3,lw=1,label=lab)
    ax_.set_xscale("log"); ax_.set_xticks(kk); ax_.set_xticklabels(kk); ax_.set_title(NM[t_],loc="left"); ax_.set_xlabel("Support cells $k$"); ax_.set_ylabel("MAE (pp)")
    ax_.set_xlim(.8,12)
h,l=axs[0].get_legend_handles_labels(); axs[5].axis("off"); axs[5].legend(h,l,loc="center",frameon=False,fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/figs/fig5_fewshot.pdf"); plt.close()
for _a,_b in {"fig1_data":"Figure_1","fig2_protocol":"Figure_2","fig3_matrix":"Figure_3","fig4_extrapolation":"Figure_4","fig5_fewshot":"Figure_5"}.items(): shutil.copy(f"{OUT}/figs/{_a}.pdf",f"{OUT}/figs/{_b}.pdf")
json.dump(NUM,open(f"{OUT}/numbers.json","w"),indent=1,default=float)
print(json.dumps(NUM,indent=1,default=float))
