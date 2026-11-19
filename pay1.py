"""PAY-1 (PREREG_IPMI.md): slice-thickness dependence of RIDER tumour texture, plug-in vs gated jackknife. CPU. python pay1.py"""
import json, numpy as np, pandas as pd
d = pd.read_csv('../data/rider/rider_tumour_texture.csv'); F4 = ['Entropy', 'JointEntropy', 'Uniformity', 'JointEnergy']; res = {}
a = d[d.thickness == 1.25].set_index(['patient', 'visit', 'kernel', 'feature']); b = d[d.thickness == 5.0].set_index(['patient', 'visit', 'kernel', 'feature'])
J = a.join(b, lsuffix='_1', rsuffix='_5', how='inner').reset_index()
for (k, f), g in J.groupby(['kernel', 'feature']):
    dp = (g.plugin_5 - g.plugin_1).abs(); dg = (g.gated_5 - g.gated_1).abs(); med = g.N_5.median(); sm = g.N_5 <= med
    red = lambda m: float(1 - dg[m].mean() / dp[m].mean()) if dp[m].mean() > 0 else 0.0
    res[f'{k}_{f}'] = dict(n=int(len(g)), plug=float(dp.mean()), gated=float(dg.mean()), red=red(np.ones(len(g), bool)), red_small=red(sm.values), red_large=red(~sm.values),
                           gate_open_5mm=float((g.gated_5 == g.JK_5).mean()), N5_median=float(med))
    print(k, f, {x: round(v, 3) if isinstance(v, float) else v for x, v in res[f'{k}_{f}'].items()})
res['D1'] = all(sum(res[f'{k}_{f}']['red'] >= 0.2 for f in F4) >= 3 for k in ('LUNG', 'STANDARD'))
res['D2'] = all(sum(res[f'{k}_{f}']['red_small'] > res[f'{k}_{f}']['red_large'] for f in F4) >= 3 for k in ('LUNG', 'STANDARD'))
print('D1', res['D1'], 'D2', res['D2']); json.dump(res, open('pay1.json', 'w'), indent=1)
