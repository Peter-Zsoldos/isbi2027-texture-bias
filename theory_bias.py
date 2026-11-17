"""TH-A (PREREG_IPMI.md): closed-form plug-in bias of first-order Entropy and Uniformity for a quantised Gaussian random field in a spherical ROI,
compared with the simulation in bench_full.csv. No fitted parameter. CPU. python theory_bias.py"""
import numpy as np, pandas as pd, json
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, pearsonr
from bench_synth import sphere, RADII

W, MU, SD, NH = 25.0, 40.0, 60.0, 30   # bin width, intensity mean/s.d. of bench_synth, Hermite terms


def corr1d(sig, L=64):
    """Exact per-axis correlation of white noise filtered with scipy's Gaussian kernel (truncate 4)."""
    d = np.zeros(2 * L + 1); d[L] = 1; k = gaussian_filter1d(d, sig, mode='constant'); k = k[k > 0]
    ac = np.correlate(k, k, 'full'); ac = ac / ac.max(); c = len(ac) // 2
    out = np.zeros(2 * L + 1); m = min(c, L); out[L - m:L + m + 1] = ac[c - m:c + m + 1]; return out   # index L = lag 0


def bins():
    """Bin edges in z units for absolute bins floor(x / W), covering +-8 s.d."""
    lo, hi = np.floor((MU - 8 * SD) / W), np.ceil((MU + 8 * SD) / W)
    e = (np.arange(lo, hi + 1) * W - MU) / SD; return e[:-1], e[1:]


def cov_table(u, v):
    """C_k(t) = P(u<X<v, u<Y<v) - pi_k^2 for corr(X,Y)=t, by Price's theorem: dC/dt = phi2(v,v) - 2 phi2(u,v) + phi2(u,u). Exact up to quadrature."""
    t = np.concatenate([np.linspace(0, 0.99, 4000, endpoint=False), 1 - np.logspace(-2, -9, 4000)])
    def phi2(a, b):
        q = 1 - t[None] ** 2; return np.exp(-(a[:, None] ** 2 - 2 * t[None] * a[:, None] * b[:, None] + b[:, None] ** 2) / (2 * q)) / (2 * np.pi * np.sqrt(q))
    D = phi2(v, v) - 2 * phi2(u, v) + phi2(u, u)
    C = np.concatenate([np.zeros((len(u), 1)), np.cumsum((D[:, 1:] + D[:, :-1]) / 2 * np.diff(t)[None], 1)], 1)
    return t, C


def pair_counts(r):
    """Number of ordered voxel pairs in a radius-r sphere at each 3D offset (FFT autocorrelation of the mask)."""
    n = 2 * r + 3; m = sphere((n, n, n), (n // 2,) * 3, r).astype(float); M = 2 * n
    F = np.fft.rfftn(m, (M, M, M)); ac = np.fft.irfftn(F * np.conj(F), (M, M, M)); ac = np.fft.fftshift(np.round(ac))
    return ac, M // 2, int(m.sum())


def var_pi(sig, r, tab, pi):
    ac, c, N = pair_counts(r); L = 64; c1 = corr1d(sig, L); M = ac.shape[0]
    idx = np.arange(M) - c; ok = np.abs(idx) <= L; rr = np.zeros(M); rr[ok] = c1[idx[ok] + L]
    R = rr[:, None, None] * rr[None, :, None] * rr[None, None, :]                 # separable 3D correlation at each offset
    nz = ac > 0; Ru, inv = np.unique(np.round(R[nz], 12), return_inverse=True); Wt = np.bincount(inv, ac[nz])
    t, C = tab; Ck = np.stack([np.interp(Ru, t, C[k]) for k in range(len(pi))])     # C_k at each distinct correlation
    Ck[:, Ru >= 1 - 1e-12] = (pi * (1 - pi))[:, None]                               # lag 0 exactly
    return (Ck @ Wt) / N ** 2, N


def predict(sig, r):
    u, v = bins(); pi = norm.cdf(v) - norm.cdf(u); keep = pi > 1e-12; tab = cov_table(u[keep], v[keep]); pi = pi[keep]
    V, N = var_pi(sig, r, tab, pi)
    return dict(N=N, Uniformity=float(V.sum()), Entropy=float(-(V / pi).sum() / (2 * np.log(2))))


def main():
    df = pd.read_csv('bench_full.csv'); df = df[(df.method == 'plugin') & df.feature.isin(['Entropy', 'Uniformity'])]
    ref = {s: predict(s, 60) for s in (0.75, 1.5, 3.0)}; rows = []
    for s in (0.75, 1.5, 3.0):
        xi = 2 * s
        for r in RADII:
            p = predict(s, r)
            for f in ('Entropy', 'Uniformity'):
                e = df[(df.sigma == s) & (df.r == r) & (df.feature == f)].err
                rows.append(dict(sigma=s, r=r, N=p['N'], neff=p['N'] / xi ** 3, feature=f, pred=p[f] - ref[s][f], meas=e.mean(), se=e.std(ddof=1) / np.sqrt(len(e)), n=len(e)))
    T = pd.DataFrame(rows); out = {}
    for f in ('Entropy', 'Uniformity'):
        t = T[T.feature == f]; sel = t[(t.neff >= 20) & (t.meas.abs() > 2 * t.se)]; rel = ((sel.pred - sel.meas).abs() / sel.meas.abs())
        out[f] = dict(median_rel_err=float(rel.median()) if len(sel) else None, n_cells=int(len(sel)), pearson=float(pearsonr(t.pred, t.meas)[0]))
        print(t[['sigma', 'r', 'N', 'neff', 'pred', 'meas', 'se']].round(4).to_string(index=False)); print(f, out[f])
    out['GO'] = bool(all(out[f]['median_rel_err'] is not None and out[f]['median_rel_err'] <= 0.2 and out[f]['pearson'] >= 0.95 for f in ('Entropy', 'Uniformity')))
    print('GO', out['GO']); T.to_csv('theory_bias.csv', index=False); json.dump(out, open('theory_bias.json', 'w'), indent=1)


if __name__ == '__main__': main()
