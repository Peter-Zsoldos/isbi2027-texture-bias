"""Fast re-implementation of PyRadiomics (IBSI) histogram and GLCM features for binary ROIs (3D or 2D)."""
import numpy as np, itertools


def discretize(img, mask, bw=25.0):
    v = img[mask]
    lo = np.floor(v.min() / bw)
    q = np.zeros(img.shape, np.int32)
    q[mask] = (np.floor(img[mask] / bw) - lo).astype(np.int32) + 1
    return q


def firstorder(q, mask):
    v = q[mask]
    c = np.bincount(v)[1:]
    p = c[c > 0] / v.size
    return {'Entropy': float(-(p * np.log2(p + 2.2e-16)).sum()), 'Uniformity': float((p ** 2).sum())}


def _dirs(nd):
    return [d for d in itertools.product([-1, 0, 1], repeat=nd) if d > (0,) * nd]


def glcm(q, mask):
    Ng = int(q.max())
    present = np.unique(q[mask])
    feats = []
    for d in _dirs(q.ndim):
        sa = tuple(slice(max(-k, 0), q.shape[i] - max(k, 0)) for i, k in enumerate(d))
        sb = tuple(slice(max(k, 0), q.shape[i] - max(-k, 0)) for i, k in enumerate(d))
        a, b = q[sa], q[sb]
        ok = mask[sa] & mask[sb]
        if not ok.any():
            continue
        P = np.zeros((Ng + 1, Ng + 1))
        np.add.at(P, (a[ok], b[ok]), 1)
        P = P + P.T
        P = P[np.ix_(present, present)]
        P /= P.sum()
        i = present[:, None].astype(float)
        j = present[None, :].astype(float)
        px = P.sum(1)
        mu = (present * px).sum()
        sig = np.sqrt(((present - mu) ** 2 * px).sum())
        nz = P > 0
        feats.append({'JointEntropy': -(P[nz] * np.log2(P[nz] + 2.2e-16)).sum(),
                      'JointEnergy': (P ** 2).sum(),
                      'Contrast': ((i - j) ** 2 * P).sum(),
                      'Idm': (P / (1 + (i - j) ** 2)).sum(),
                      'Correlation': (((i - mu) * (j - mu) * P).sum() / sig ** 2) if sig > 0 else 1.0})
    return {k: float(np.mean([f[k] for f in feats])) for k in feats[0]}


def features(img, mask, bw=25.0):
    q = discretize(img, mask, bw)
    out = firstorder(q, mask)
    out.update(glcm(q, mask))
    return out
