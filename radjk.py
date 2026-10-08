"""radjk: gated block-jackknife correction for any radiomics extractor (drop-in for PyRadiomics).

    from radiomics import featureextractor
    from radjk import extract_corrected
    ext = featureextractor.RadiomicsFeatureExtractor(params)        # your usual configuration
    feats = extract_corrected(ext, image, mask)                     # SimpleITK image + mask, as for ext.execute

Returns the extractor's usual 'original_*' features, plus for each the gated corrected value
'jk_*' and diagnostics 'jk_xi' (correlation length, voxels), 'jk_neff' (N/xi^3) and 'jk_reliable' (neff >= 20).
Only features listed in CORRECT (smooth histogram/GLCM functionals) are corrected; others are copied unchanged.
"""
import numpy as np, SimpleITK as sitk
from corrections import grid_blocks, crop
from real_regions import corr_length

CORRECT = ('firstorder_Entropy', 'firstorder_Uniformity', 'firstorder_Energy_normalized',
           'glcm_JointEntropy', 'glcm_JointEnergy', 'glcm_SumEntropy', 'glcm_DifferenceEntropy',
           'glcm_MaximumProbability', 'glcm_MCC', 'glcm_Imc1', 'glcm_Imc2')
THRESH = 20.0


def _run(ext, img_arr, msk_arr, ref_img):
    I = sitk.GetImageFromArray(img_arr); I.SetSpacing(ref_img.GetSpacing())
    M = sitk.GetImageFromArray(msk_arr.astype(np.uint8)); M.SetSpacing(ref_img.GetSpacing())
    r = ext.execute(I, M)
    return {k: float(v) for k, v in r.items() if k.startswith('original_')}


def extract_corrected(ext, image, mask, k=2, thresh=THRESH):
    img = sitk.GetArrayFromImage(image).astype(np.float32)
    m = sitk.GetArrayFromImage(mask) > 0
    img, m = crop(img, m, pad=2)
    full = _run(ext, img, m, image)
    N = int(m.sum())
    xi = corr_length(np.where(m, img, img[m].mean())) if min(img.shape) > 8 else np.nan
    neff = N / xi ** 3 if xi and np.isfinite(xi) else np.nan
    out = dict(full, jk_xi=xi, jk_neff=neff, jk_reliable=bool(neff >= thresh))
    keys = [kk for kk in full if kk[len('original_'):] in CORRECT]
    if not out['jk_reliable'] or not keys:
        out.update({'jk_' + kk[len('original_'):]: full[kk] for kk in keys})
        return out
    lab = grid_blocks(m, k)
    B = lab.max()
    nb = np.array([(lab == b).sum() for b in range(1, B + 1)])
    loo = [_run(ext, img, m & (lab != b), image) for b in range(1, B + 1)]
    for kk in keys:
        out['jk_' + kk[len('original_'):]] = B * full[kk] - sum((1 - nb[b] / N) * loo[b][kk] for b in range(B))
    return out
