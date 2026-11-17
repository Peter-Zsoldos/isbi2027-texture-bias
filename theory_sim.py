"""TH-A2 (PREREG_IPMI.md): powered simulation of plug-in first-order Entropy/Uniformity bias vs the analytic population value. CPU. python theory_sim.py"""
import numpy as np, pandas as pd, json
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from bench_synth import sphere, RADII
import theory_bias as T


def feats(v):
    b = np.floor(v / T.W).astype(int); _, c = np.unique(b, return_counts=True); p = c / c.sum()
    return -(p * np.log2(p)).sum(), (p ** 2).sum()


def job(a):
    sig, rep = a; rng = np.random.default_rng(77 + 1000 * rep + int(10 * sig)); n = 128
    x = gaussian_filter(rng.normal(size=(n, n, n)), sig, mode='wrap'); img = x / x.std() * T.SD + T.MU; out = []
    for r in RADII:
        m0 = sphere((2 * r + 1,) * 3, (r, r, r), r)
        for _ in range(20):
            c = rng.integers(0, n - 2 * r - 1, 3); v = img[c[0]:c[0] + 2 * r + 1, c[1]:c[1] + 2 * r + 1, c[2]:c[2] + 2 * r + 1][m0]; e, u = feats(v)
            out.append(dict(sigma=sig, r=r, Entropy=e, Uniformity=u))
    return out


if __name__ == '__main__':
    with Pool(10) as p: R = p.map(job, [(s, k) for s in (0.75, 1.5, 3.0) for k in range(20)])
    df = pd.DataFrame([x for rr in R for x in rr]); u, v = T.bins(); pi = norm.cdf(v) - norm.cdf(u); pi = pi[pi > 1e-15]
    truth = dict(Entropy=float(-(pi * np.log2(pi)).sum()), Uniformity=float((pi ** 2).sum())); rows = []
    for s in (0.75, 1.5, 3.0):
        for r in RADII:
            p = T.predict(s, r); d = df[(df.sigma == s) & (df.r == r)]
            for f in ('Entropy', 'Uniformity'):
                e = d[f] - truth[f]; rows.append(dict(sigma=s, r=r, N=p['N'], neff=p['N'] / (2 * s) ** 3, feature=f, pred=p[f], meas=e.mean(), se=e.std(ddof=1) / np.sqrt(len(e)), n=len(e)))
    Tb = pd.DataFrame(rows); out = {}
    for f in ('Entropy', 'Uniformity'):
        t = Tb[Tb.feature == f]; print(t[['sigma', 'r', 'N', 'neff', 'pred', 'meas', 'se']].round(4).to_string(index=False)); g = t[t.neff >= 20]
        if f == 'Uniformity': out[f] = dict(frac_ok=float(((g.pred - g.meas).abs() <= np.maximum(2 * g.se, 0.1 * g.meas.abs())).mean()), n_cells=int(len(g)))
        else: out[f] = dict(median_rel_err=float(((g.pred - g.meas).abs() / g.meas.abs()).median()), n_cells=int(len(g)))
        print(f, out[f])
    out['GO'] = bool(out['Uniformity']['frac_ok'] >= 0.8 and out['Entropy']['median_rel_err'] <= 0.2); print('GO', out['GO'])
    Tb.to_csv('theory_sim.csv', index=False); json.dump(out, open('theory_sim.json', 'w'), indent=1)
