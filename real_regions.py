"""Real CT texture with an internal reference: large homogeneous soft-tissue spheres in MSD lung CTs,
small sub-spheres inside them; plug-in vs corrected features (fast implementation, PyRadiomics-equivalent)."""
import numpy as np, pandas as pd, SimpleITK as sitk, glob, sys, time
from multiprocessing import Pool
from scipy.ndimage import binary_erosion, median_filter, distance_transform_edt, label, gaussian_filter
SMOOTH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
from texfeat import features
from corrections import evaluate

RBIG, RADII = 14, [3, 4, 5, 6, 8]


def ball(r):
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return z * z + y * y + x * x <= r * r


def corr_length(v):
    """1/e correlation length (voxels) of a 3D block along the three axes, averaged."""
    v = v - v.mean(); L = []
    for ax in range(3):
        a = np.moveaxis(v, ax, 0); c0 = (a * a).mean(); lag = 1
        while lag < a.shape[0] // 2 and (a[lag:] * a[:-lag]).mean() / c0 > np.exp(-1):
            lag += 1
        # linear interpolation of the crossing
        r1 = (a[lag - 1:] * a[:a.shape[0] - lag + 1]).mean() / c0 if lag > 1 else 1.0
        r2 = (a[lag:] * a[:-lag]).mean() / c0
        L.append(lag - 1 + (r1 - np.exp(-1)) / max(r1 - r2, 1e-9))
    return float(np.mean(L))


def job(f):
    rng = np.random.default_rng(abs(hash(f)) % 2**31)
    img = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float32)
    soft = median_filter(((img > -50) & (img < 150)).astype(np.uint8), 3).astype(bool)
    if SMOOTH > 0:
        img = gaussian_filter(img, SMOOTH)
    dist = distance_transform_edt(soft)
    cand = np.argwhere(dist > RBIG + 1)
    rows = []
    if len(cand) == 0:
        return rows
    for t in range(3):                      # up to 3 big regions per scan
        c = cand[rng.integers(len(cand))]
        sl = tuple(slice(ci - RBIG - 2, ci + RBIG + 3) for ci in c)
        sub = img[sl]
        if sub.shape != (2 * RBIG + 5,) * 3:
            continue
        B = np.zeros(sub.shape, bool); B[2:-2, 2:-2, 2:-2] = ball(RBIG)
        ref = features(sub, B)
        xi = corr_length(sub[2:-2, 2:-2, 2:-2][4:-4, 4:-4, 4:-4])
        sd = float(sub[B].std())
        for r in RADII:
            for s in range(4):
                off = rng.integers(-(RBIG - r), RBIG - r + 1, 3)
                if np.sum(off ** 2) > (RBIG - r) ** 2:
                    continue
                m = np.zeros(sub.shape, bool)
                cc = np.array(sub.shape) // 2 + off
                m[tuple(slice(ci - r, ci + r + 1) for ci in cc)] = ball(r)
                keys, est = evaluate(features, sub, m, 2)
                for name, v in est.items():
                    for kk, vv in zip(keys, v):
                        rows.append(dict(case=f[-15:], smooth=SMOOTH, region=t, xi=xi, sd=sd, r=r, method=name, feature=kk,
                                         value=vv, ref=ref[kk]))
    return rows


if __name__ == '__main__':
    files = sorted(glob.glob(sys.argv[1] + '/imagesTr/lung_*.nii.gz'))[:int(sys.argv[2])]
    t0 = time.time()
    with Pool(6) as p:
        R = p.map(job, files)
    df = pd.DataFrame([x for rr in R for x in rr]); df['err'] = df.value - df.ref
    df.to_csv(sys.argv[3], index=False)
    print('time', round(time.time() - t0), 'regions', df.groupby(['case', 'region']).ngroups,
          'xi median %.2f' % df.drop_duplicates(['case', 'region']).xi.median())
    pd.set_option('display.width', 200)
    for f in ['Entropy', 'JointEntropy', 'Correlation', 'Contrast']:
        d = df[df.feature == f]
        print(f, '| mean err:'); print(d.groupby(['method', 'r']).err.mean().unstack('r').round(3).to_string())
