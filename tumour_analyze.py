"""TUM-1 analysis. python tumour_analyze.py"""
import json, numpy as np, pandas as pd
NV = {3: 123, 4: 257, 5: 515, 6: 925}
d = pd.read_csv('tumour_regions.csv'); d['tum'] = d.case + '_' + d.comp.astype(str) + '_' + d.region.astype(str); d['N'] = d.r.map(NV)
W = d[d.method.isin(['plugin', 'JK'])].pivot_table(index=['tum', 'r', 'feature', 'ref', 'xi', 'N', 'value'], columns='method', values='err', aggfunc='first') if False else None
P = d[d.method == 'plugin'].reset_index(drop=True); J = d[d.method == 'JK'].reset_index(drop=True); assert (P.feature.values == J.feature.values).all()
P['jk'] = J.err.values; P['gated'] = np.where(P.N / P.xi ** 3 >= 20, P.jk, P.err); res = {'n_regions': int(P.tum.nunique()), 'xi_median': float(P.drop_duplicates('tum').xi.median())}
def clus(g, col):
    m = g.groupby('tum')[col].mean(); return float(m.mean()), float(1.96 * m.std(ddof=1) / np.sqrt(len(m)))
for f in ['Entropy', 'Uniformity', 'JointEntropy', 'JointEnergy', 'Contrast', 'Idm', 'Correlation']:
    g = P[P.feature == f]; g3 = g[g.r == 3]
    res[f] = dict(bias3_plug=clus(g3, 'err'), bias3_gated=clus(g3, 'gated'), rmse_plug=float(np.sqrt((g.err ** 2).mean())), rmse_gated=float(np.sqrt((g.gated ** 2).mean())), gate_open=float((g.N / g.xi ** 3 >= 20).mean()))
    print(f, {k: (np.round(v, 4).tolist() if isinstance(v, tuple) else round(v, 4)) for k, v in res[f].items()})
t1 = all(res[f]['bias3_plug'][0] + res[f]['bias3_plug'][1] < 0 and abs(res[f]['bias3_gated'][0]) <= 0.5 * abs(res[f]['bias3_plug'][0]) for f in ['Entropy', 'JointEntropy'])
t2 = sum(res[f]['rmse_gated'] <= res[f]['rmse_plug'] for f in ['Entropy', 'Uniformity', 'JointEntropy', 'JointEnergy']) >= 3
res['T1'], res['T2'] = bool(t1), bool(t2); print('regions', res['n_regions'], 'xi median %.2f' % res['xi_median'], 'T1', t1, 'T2', t2); json.dump(res, open('tumour_regions.json', 'w'), indent=1)
