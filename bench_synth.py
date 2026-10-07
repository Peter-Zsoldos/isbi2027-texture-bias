"""Synthetic benchmark: Gaussian random field textures, spherical ROIs, bias vs population value."""
import numpy as np, pandas as pd, sys, time
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
from texfeat import features
from corrections import evaluate

RADII = [3, 4, 5, 6, 8, 10, 14]


def sphere(shape, c, r):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r


def job(args):
    sig, rep, k = args
    rng = np.random.default_rng(1000 * rep + int(10 * sig))
    n = 128
    x = gaussian_filter(rng.normal(size=(n, n, n)), sig, mode='wrap')
    img = (x / x.std() * 60 + 40).astype(np.float32)
    ref = features(img, sphere(img.shape, (64, 64, 64), 60))
    rows = []
    for r in RADII:
        for t in range(2):
            c = tuple(rng.integers(r + 2, n - r - 2, 3))
            keys, est = evaluate(features, img, sphere(img.shape, c, r), k)
            for name, v in est.items():
                for kk, vv in zip(keys, v):
                    rows.append(dict(sigma=sig, rep=rep, r=r, method=name, feature=kk, value=vv, ref=ref[kk]))
    return rows


if __name__ == '__main__':
    reps = int(sys.argv[1]); k = int(sys.argv[2]); out = sys.argv[3]
    t0 = time.time()
    with Pool(11) as p:
        R = p.map(job, [(s, r, k) for s in [0.75, 1.5, 3.0] for r in range(reps)])
    df = pd.DataFrame([x for rr in R for x in rr]); df['err'] = df.value - df.ref
    df.to_csv(out, index=False)
    print('time', round(time.time() - t0))
    tab = df.groupby(['feature', 'sigma', 'method', 'r']).err.mean().unstack('r')
    pd.set_option('display.width', 200)
    print(tab.round(3).to_string())
