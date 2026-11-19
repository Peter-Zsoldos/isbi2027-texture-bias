"""TUM-1 (PREREG_IPMI.md): internal-reference bias test inside MSD lung tumours. python tumour_regions.py <Task06 dir> tumour_regions.csv"""
import numpy as np, pandas as pd, SimpleITK as sitk, glob, sys, time
from multiprocessing import Pool
from scipy.ndimage import distance_transform_edt, label
from texfeat import features
from corrections import evaluate
from real_regions import ball, corr_length
RBIG, RADII = 10, [3, 4, 5, 6]

def job(f):
    rng = np.random.default_rng(abs(hash(f[-15:])) % 2**31); rows = []
    img = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float32)
    lab = sitk.GetArrayFromImage(sitk.ReadImage(f.replace('imagesTr', 'labelsTr'))) > 0
    comps, n = label(lab)
    for ci in range(1, n + 1):
        d = distance_transform_edt(comps == ci); used = np.zeros_like(lab)
        for t in range(2):
            dd = np.where(used, 0, d)
            if dd.max() < RBIG + 1: break
            c = np.array(np.unravel_index(np.argmax(dd), dd.shape)); sl = tuple(slice(x - RBIG - 2, x + RBIG + 3) for x in c)
            if any(x - RBIG - 2 < 0 or x + RBIG + 3 > s for x, s in zip(c, img.shape)): break
            sub = img[sl]; B = np.zeros(sub.shape, bool); B[2:-2, 2:-2, 2:-2] = ball(RBIG); ref = features(sub, B)
            xi = corr_length(sub[2:-2, 2:-2, 2:-2][3:-3, 3:-3, 3:-3])
            z, y, x = np.ogrid[:img.shape[0], :img.shape[1], :img.shape[2]]; used |= ((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) <= (2 * RBIG) ** 2
            for r in RADII:
                for s in range(4):
                    off = rng.integers(-(RBIG - r), RBIG - r + 1, 3)
                    if np.sum(off ** 2) > (RBIG - r) ** 2: continue
                    m = np.zeros(sub.shape, bool); cc = np.array(sub.shape) // 2 + off; m[tuple(slice(q - r, q + r + 1) for q in cc)] = ball(r)
                    keys, est = evaluate(features, sub, m, 2)
                    for name, v in est.items():
                        for kk, vv in zip(keys, v): rows.append(dict(case=f[-15:], comp=ci, region=t, xi=xi, r=r, method=name, feature=kk, value=vv, ref=ref[kk]))
    return rows

if __name__ == '__main__':
    import os
    files = sorted(glob.glob(sys.argv[1] + '/imagesTr/lung_*.nii.gz')); t0 = time.time(); PART = 'tumour_parts'; os.makedirs(PART, exist_ok=True)
    todo = [f for f in files if not os.path.exists(os.path.join(PART, os.path.basename(f) + '.csv'))]
    with Pool(2) as p:                                               # 2 workers: each holds one full CT in RAM; per-case files make the run resumable
        for f, rows in zip(todo, p.imap(job, todo)):
            pd.DataFrame(rows).to_csv(os.path.join(PART, os.path.basename(f) + '.csv'), index=False); print('done', os.path.basename(f), len(rows), flush=True)
    parts = [pd.read_csv(os.path.join(PART, os.path.basename(f) + '.csv')) for f in files if os.path.getsize(os.path.join(PART, os.path.basename(f) + '.csv')) > 2]
    df = pd.concat(parts); df['err'] = df.value - df.ref; df.to_csv(sys.argv[2], index=False)
    print('time', round(time.time() - t0), 'tumour regions', df.groupby(['case', 'comp', 'region']).ngroups, 'xi median %.2f' % df.drop_duplicates(['case', 'comp', 'region']).xi.median())
