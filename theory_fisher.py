"""TH-B (PREREG_IPMI.md): efficient Fisher information about log sigma in a spherical ROI with the correlation length unknown (Matern 1/2, 3/2). CPU."""
import numpy as np, json
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress


def matern(d, xi, nu):
    a = d / xi
    if nu == 0.5: return np.exp(-a), a * np.exp(-a)                       # R, dR/dlog xi
    s = np.sqrt(3) * a; return (1 + s) * np.exp(-s), s ** 2 * np.exp(-s)  # nu = 3/2


def info(r, xi, nu):
    g = np.arange(-r, r + 1); P = np.array([(x, y, z) for x in g for y in g for z in g if x * x + y * y + z * z <= r * r], float)
    D = squareform(pdist(P)); R, dR = matern(D, xi, nu); R += 1e-9 * np.eye(len(P)); Ri = np.linalg.inv(R)
    A = Ri @ dR; Iss = 2 * len(P); Isx = np.trace(A); Ixx = 0.5 * np.sum(A * A.T)    # dS/dlog s = 2S -> I_ss = 2n, I_sx = tr(R^-1 dR), I_xx = 1/2 tr((R^-1 dR)^2)
    return len(P), (Iss - Isx ** 2 / Ixx) / 2


out = []
for nu in (0.5, 1.5):
    for xi in (0.5, 1, 2, 4, 8):
        for r in (2, 3, 4, 5, 6, 8):
            N, I = info(r, xi, nu); out.append(dict(nu=nu, xi=xi, r=r, N=N, neff=N / xi ** 3, I_indep=I)); print(nu, xi, r, N, round(N / xi ** 3, 2), round(I, 2), flush=True)
res = {}
for nu in (0.5, 1.5):
    big = [o for o in out if o['nu'] == nu and o['neff'] >= 20]; small = [o for o in out if o['nu'] == nu and o['neff'] <= 1]
    sl = linregress(np.log([o['neff'] for o in big]), np.log([o['I_indep'] for o in big])).slope
    res[str(nu)] = dict(slope_large=float(sl), max_small=float(max(o['I_indep'] for o in small)) if small else None)
res['GO'] = bool(all(0.85 <= res[k]['slope_large'] <= 1.15 and res[k]['max_small'] is not None and res[k]['max_small'] <= 5 for k in ('0.5', '1.5')))
print(res); json.dump(dict(rows=out, summary=res), open('theory_fisher.json', 'w'), indent=1)
