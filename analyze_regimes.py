"""Real CT soft tissue in four correlation regimes: plug-in vs block jackknife vs criterion-gated jackknife (Table 1, Fig. 1b)."""
import pandas as pd, numpy as np

NVOX = {3: 123, 4: 257, 5: 515, 6: 925, 8: 2109}
frames = []
for f, s in [('real_full.csv', 0.0), ('smooth_0.7.csv', 0.7), ('smooth_1.4.csv', 1.4), ('smooth_2.0.csv', 2.0)]:
    d = pd.read_csv(f)
    nf, nm = d.feature.nunique(), d.method.nunique()
    d['roi'] = np.arange(len(d)) // (nf * nm)          # rows are written ROI by ROI
    assert d.groupby('roi')[['r', 'region', 'case']].nunique().max().max() == 1
    d['smooth'] = s
    frames.append(d)
D = pd.concat(frames)
D['N'] = D.r.map(NVOX)
D['neff'] = D.N / D.xi ** 3
D.to_csv('real_all_smooth.csv', index=False)

W = D.pivot_table(index=['smooth', 'roi', 'r', 'feature', 'neff', 'ref'], columns='method', values='value').reset_index()
W['gated'] = np.where(W.neff >= 20, W.JK, W.plugin)
rows = []
for (feat, s), g in W.groupby(['feature', 'smooth']):
    e = {m: g[m] - g.ref for m in ['plugin', 'JK', 'gated']}
    rows.append(dict(feature=feat, smooth=s, applied=(g.neff >= 20).mean(),
                     **{f'bias_{m}': e[m].mean() for m in e}, **{f'rmse_{m}': np.sqrt((e[m] ** 2).mean()) for m in e}))
T = pd.DataFrame(rows)
T.to_csv('real_regimes_summary.csv', index=False)
pd.set_option('display.width', 200)
print('ROIs per regime', W.groupby('smooth').roi.nunique().to_dict())
print('median xi', D.drop_duplicates(['smooth', 'case', 'region']).groupby('smooth').xi.median().round(2).to_dict())
print(T.round(3).to_string())
