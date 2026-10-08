"""Practicality of the texture correction on real RIDER tumours: runtime of PyRadiomics with and without radjk,
and the share of tumours flagged unreliable (N/xi^3 < 20) per reconstruction."""
import sys, time, logging, numpy as np, pandas as pd, SimpleITK as sitk
sys.path.insert(0, '../../texture')
from radiomics import featureextractor
from radjk import extract_corrected
logging.getLogger('radiomics').setLevel(logging.ERROR)

idx = pd.read_csv('index.csv'); idx = idx[idx.uniform]
rng = np.random.default_rng(0)
sel = idx.groupby(['kernel', 'thickness']).sample(4, random_state=0)
for classes, label in [(['firstorder', 'glcm'], 'first-order + GLCM'), (None, 'all classes')]:
    ext = featureextractor.RadiomicsFeatureExtractor(binWidth=25)
    if classes:
        ext.disableAllFeatures()
        for c in classes:
            ext.enableFeatureClassByName(c)
    tp, tj = [], []
    for _, r in sel.iterrows():
        I, M = sitk.ReadImage(r.ct), sitk.ReadImage(r.seg)
        M = sitk.Cast(M > 0, sitk.sitkUInt8)
        a = time.perf_counter(); ext.execute(I, M); tp.append(time.perf_counter() - a)
        a = time.perf_counter(); extract_corrected(ext, I, M); tj.append(time.perf_counter() - a)
    print(f'{label}: PyRadiomics {np.median(tp):.2f} s/tumour, with radjk {np.median(tj):.2f} s/tumour (median of {len(tp)})')
T = pd.read_csv('rider_tumour_texture.csv')
T = T[T.feature == 'JointEntropy']
print('share of tumours with N/xi^3 >= 20 per reconstruction:')
print(T.assign(ok=T.neff >= 20).groupby(['kernel', 'thickness']).ok.mean().round(2).to_string())
print('tumour N (voxels) median by thickness:', T.groupby('thickness').N.median().to_dict())
