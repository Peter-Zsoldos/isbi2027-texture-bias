import numpy as np, SimpleITK as sitk, logging
from scipy.ndimage import gaussian_filter
from radiomics import featureextractor
from texfeat import features
logging.getLogger('radiomics').setLevel(logging.ERROR)
ext = featureextractor.RadiomicsFeatureExtractor(binWidth=25)
ext.disableAllFeatures()
ext.enableFeaturesByName(firstorder=['Entropy', 'Uniformity'],
                         glcm=['JointEntropy', 'JointEnergy', 'Contrast', 'Idm', 'Correlation'])
rng = np.random.default_rng(0)
x = gaussian_filter(rng.normal(size=(64, 64, 64)), 1.5)
img = (x / x.std() * 60 + 40).astype(np.float32)
z, y, xx = np.ogrid[:64, :64, :64]
for r in [3, 6, 12]:
    m = (z - 32) ** 2 + (y - 30) ** 2 + (xx - 33) ** 2 <= r * r
    res = ext.execute(sitk.GetImageFromArray(img), sitk.GetImageFromArray(m.astype(np.uint8)))
    pr = {k.split('_')[-1]: float(v) for k, v in res.items() if k.startswith('original_')}
    me = features(img, m)
    print(r, {k: (round(pr[k], 5), round(me[k], 5)) for k in me})
