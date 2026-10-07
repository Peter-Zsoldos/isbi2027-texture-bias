import pandas as pd, numpy as np, sys
df=pd.read_csv('pyrad_full.csv')
df=df[~df.feature.str.startswith('diagnostics')]
out=[]
for (f,r),d in df.groupby(['feature','r']):
    ref_sd=d.drop_duplicates(['case','region']).ref.std()
    p=d[d.method=='plugin'].err.values; j=d[d.method=='JK'].err.values
    if ref_sd==0 or np.isnan(ref_sd): continue
    out.append(dict(feature=f,r=r,cls=f.split('_')[0],bias_p=p.mean()/ref_sd,bias_j=j.mean()/ref_sd,
                    rmse_p=np.sqrt((p**2).mean())/ref_sd,rmse_j=np.sqrt((j**2).mean())/ref_sd))
T=pd.DataFrame(out); T.to_csv('pyrad_summary.csv',index=False)
for r in [4,6]:
    t=T[T.r==r]; big=t[t.bias_p.abs()>0.2]
    print(f'r={r}: features {len(t)}, with |plug-in bias|>0.2 SD: {len(big)}; of these JK reduces |bias| by >50%: {(big.bias_j.abs()<0.5*big.bias_p.abs()).sum()}, RMSE lower: {(big.rmse_j<big.rmse_p).sum()}')
    print('  median |bias| plug-in %.2f -> JK %.2f (all features); among biased: %.2f -> %.2f'%(t.bias_p.abs().median(),t.bias_j.abs().median(),big.bias_p.abs().median(),big.bias_j.abs().median()))
    print('  RMSE worse by >10%% for %d features: %s'%(((t.rmse_j>1.1*t.rmse_p)).sum(), ', '.join(t[t.rmse_j>1.1*t.rmse_p].feature.head(12))))
    print(t.groupby('cls').apply(lambda g: pd.Series({'n':len(g),'|b_p|':g.bias_p.abs().median(),'|b_j|':g.bias_j.abs().median(),'rmse_p':g.rmse_p.median(),'rmse_j':g.rmse_j.median()})).round(2).to_string())
NONSMOOTH=['10Percentile','90Percentile','Median','InterquartileRange','RobustMeanAbsoluteDeviation','Range','Minimum','Maximum']
def ftype(f):
    c,n=f.split('_',1)
    if c=='firstorder' and n in NONSMOOTH: return 'B non-smooth (quantile)'
    if c in ('firstorder','glcm'): return 'A smooth (hist./GLCM)'
    return 'C region-geometry'
T['type']=T.feature.map(ftype); T.to_csv('pyrad_summary.csv',index=False)
for r in [4,6]:
    t=T[T.r==r]
    g=t.groupby('type').apply(lambda g: pd.Series({'n':len(g),'biased(>0.2SD)':(g.bias_p.abs()>0.2).sum(),
        'med|b| plug':g.bias_p.abs().median(),'med|b| JK':g.bias_j.abs().median(),
        'RMSE lower':(g.rmse_j<g.rmse_p).sum(),'RMSE >10% worse':(g.rmse_j>1.1*g.rmse_p).sum()}),include_groups=False)
    print('r=',r); print(g.round(2).to_string())
    a=t[t.type.str.startswith('A')&(t.bias_p.abs()>0.2)]
    print('  type A biased features:', ', '.join(f'{x}:{p:+.2f}->{j:+.2f}' for x,p,j in zip(a.feature,a.bias_p,a.bias_j)))
