"""Cross-tumour analysis (MSD lung): PyRadiomics features, plug-in vs block-jackknife, and their Spearman
correlation with tumour volume."""
import numpy as np, pandas as pd, SimpleITK as sitk, glob, sys, time, os
from multiprocessing import Pool
from scipy.ndimage import label as cc
from corrections import evaluate
from pyrad_wrapper import pyrad


def job(f):
    img = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float32)
    L = sitk.GetArrayFromImage(sitk.ReadImage(f.replace('imagesTr', 'labelsTr'))) > 0
    sp = sitk.ReadImage(f).GetSpacing()[::-1]
    lab, n = cc(L, structure=np.ones((3, 3, 3)))
    rows = []
    for j in range(1, n + 1):
        m = lab == j
        if m.sum() < 64:
            continue
        keys, est = evaluate(pyrad, img, m, 2, halves=False)
        for name, v in est.items():
            for kk, vv in zip(keys, v):
                rows.append(dict(case=os.path.basename(f), comp=j, nvox=int(m.sum()),
                                 vol=float(m.sum() * np.prod(sp)), method=name, feature=kk, value=vv))
    return rows


if __name__ == '__main__':
    files = sorted(glob.glob(sys.argv[1] + '/imagesTr/lung_*.nii.gz'))
    t0 = time.time()
    with Pool(int(sys.argv[3])) as p:
        R = p.map(job, files)
    df = pd.DataFrame([x for rr in R for x in rr])
    df.to_csv(sys.argv[2], index=False)
    print('time', round(time.time() - t0), 'tumours', df.groupby(['case', 'comp']).ngroups)
