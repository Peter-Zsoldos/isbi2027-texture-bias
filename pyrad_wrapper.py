"""Model-free use: the block jackknife wrapped around the full PyRadiomics extractor (all non-shape features),
on homogeneous soft-tissue regions of MSD lung CTs with an internal large-sphere reference."""
import numpy as np, pandas as pd, SimpleITK as sitk, glob, sys, time, logging
from multiprocessing import Pool
from radiomics import featureextractor
from corrections import evaluate
from real_regions import ball, RBIG
from scipy.ndimage import median_filter, distance_transform_edt

logging.getLogger('radiomics').setLevel(logging.ERROR)
CLASSES = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
RADII = [4, 6]
_ext = None


def pyrad(img, mask):
    global _ext
    if _ext is None:
        _ext = featureextractor.RadiomicsFeatureExtractor(binWidth=25)
        _ext.disableAllFeatures()
        for c in CLASSES:
            _ext.enableFeatureClassByName(c)
    r = _ext.execute(sitk.GetImageFromArray(img.astype(np.float32)), sitk.GetImageFromArray(mask.astype(np.uint8)))
    return {k[len('original_'):]: float(v) for k, v in r.items() if k.startswith('original_')}


def job(f):
    rng = np.random.default_rng(abs(hash(f)) % 2**31)
    img = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float32)
    soft = median_filter(((img > -50) & (img < 150)).astype(np.uint8), 3).astype(bool)
    cand = np.argwhere(distance_transform_edt(soft) > RBIG + 1)
    rows = []
    if len(cand) == 0:
        return rows
    for t in range(2):
        c = cand[rng.integers(len(cand))]
        sub = img[tuple(slice(ci - RBIG - 2, ci + RBIG + 3) for ci in c)]
        if sub.shape != (2 * RBIG + 5,) * 3:
            continue
        B = np.zeros(sub.shape, bool); B[2:-2, 2:-2, 2:-2] = ball(RBIG)
        ref = pyrad(sub, B)
        for r in RADII:
            for s in range(2):
                off = rng.integers(-(RBIG - r), RBIG - r + 1, 3)
                if np.sum(off ** 2) > (RBIG - r) ** 2:
                    continue
                m = np.zeros(sub.shape, bool)
                cc = np.array(sub.shape) // 2 + off
                m[tuple(slice(ci - r, ci + r + 1) for ci in cc)] = ball(r)
                keys, est = evaluate(pyrad, sub, m, 2, halves=False)
                for name, v in est.items():
                    if name == 'SH':
                        continue
                    for kk, vv in zip(keys, v):
                        rows.append(dict(case=f[-15:], region=t, r=r, method=name, feature=kk, value=vv, ref=ref[kk]))
    return rows


if __name__ == '__main__':
    files = sorted(glob.glob(sys.argv[1] + '/imagesTr/lung_*.nii.gz'))[:int(sys.argv[2])]
    t0 = time.time()
    with Pool(int(sys.argv[4])) as p:
        R = p.map(job, files)
    df = pd.DataFrame([x for rr in R for x in rr]); df['err'] = df.value - df.ref
    df.to_csv(sys.argv[3], index=False)
    print('time', round(time.time() - t0), 'rois', len(df) // max(df.feature.nunique(), 1) // 2)
