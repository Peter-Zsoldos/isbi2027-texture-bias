"""Finite-ROI bias corrections for any radiomic feature function f(img, mask) -> dict."""
import numpy as np


def crop(img, mask, pad=1):
    idx = np.nonzero(mask)
    sl = tuple(slice(max(i.min() - pad, 0), i.max() + pad + 1) for i in idx)
    return img[sl], mask[sl]


def grid_blocks(mask, k):
    """Partition ROI voxels into (up to) k^d contiguous blocks by splitting the bounding box into a k-grid
    with equal voxel quantiles along each axis. Returns an int label array (0 = background)."""
    idx = np.nonzero(mask)
    lab = np.zeros(mask.shape, np.int32)
    cell = np.zeros(len(idx[0]), np.int64)
    for a, coord in enumerate(idx):
        edges = np.quantile(coord, np.linspace(0, 1, k + 1)[1:-1])
        cell = cell * k + np.searchsorted(edges, coord, side='right')
    u, inv = np.unique(cell, return_inverse=True)
    lab[idx] = inv + 1
    return lab


def _as_arr(d, keys):
    return np.array([d[k] for k in keys])


def evaluate(f, img, mask, k=3, halves=True):
    """Returns dict of estimates: plug-in, delete-a-block jackknife (JK), second-order generalized jackknife (JK2)."""
    img, mask = crop(img, mask)
    full = f(img, mask)
    keys = list(full.keys())
    th = _as_arr(full, keys)
    lab = grid_blocks(mask, k)
    B = lab.max()
    N = mask.sum()
    nb = np.array([(lab == b).sum() for b in range(1, B + 1)])
    loo = np.array([_as_arr(f(img, mask & (lab != b)), keys) for b in range(1, B + 1)])
    # delete-a-group jackknife with unequal groups (Busing et al. 1999)
    jk = B * th - ((1 - nb / N)[:, None] * loo).sum(0)
    if not halves:
        return keys, {'plugin': th, 'JK': jk}
    # split-half (Richardson with halves along each axis): theta_inf ~ 2 theta - mean(theta_half)
    idx = np.nonzero(mask)
    halves = []
    for coord in idx:
        med = np.median(coord)
        for side in (coord <= med, coord > med):
            m = np.zeros_like(mask)
            m[tuple(c[side] for c in idx)] = True
            if m.sum() > 2:
                halves.append(_as_arr(f(img, m), keys))
    sh = 2 * th - np.mean(halves, 0)
    return keys, {'plugin': th, 'JK': jk, 'SH': sh}
