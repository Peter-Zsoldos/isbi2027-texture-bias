import pandas as pd, numpy as np
from scipy.stats import spearmanr
from analyze_pyrad import NONSMOOTH
df=pd.read_csv('tumour_full.csv'); df=df[~df.feature.str.startswith('diagnostics')]
def ftype(f):
    c,n=f.split('_',1)
    if n in ('Energy','TotalEnergy'): return 'extensive'
    if c=='firstorder' and n in NONSMOOTH: return 'B'
    if c in ('firstorder','glcm'): return 'A'
    return 'C'
W=df.pivot_table(index=['case','comp','vol','nvox'],columns=['method','feature'],values='value').reset_index()
print('tumours',len(W),'nvox range',W.nvox.min(),W.nvox.max(),'median',W.nvox.median())
rows=[]
for f in df.feature.unique():
    p=W[('plugin',f)]; j=W[('JK',f)]
    if p.std()==0: continue
    rows.append(dict(feature=f,type=ftype(f),rho_p=spearmanr(W.vol,p)[0],rho_j=spearmanr(W.vol,j)[0],
                     shift=((j-p)/p.std()).median()))
R=pd.DataFrame(rows); R.to_csv('tumour_summary.csv',index=False)
for t,g in R.groupby('type'):
    print(t,len(g),'median |rho| plug-in %.2f -> JK %.2f; features with |rho|>0.3: %d -> %d'%(g.rho_p.abs().median(),g.rho_j.abs().median(),(g.rho_p.abs()>0.3).sum(),(g.rho_j.abs()>0.3).sum()))
a=R[R.type=='A'].sort_values('rho_p')
print(a[['feature','rho_p','rho_j','shift']].round(2).to_string())
