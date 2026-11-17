"""IPMI overview figure (LNCS text width, 4.8 in): (a) the problem: small ROIs look homogeneous; (b) why: ROI = window on a correlated
field, bias = c/N, predicted with no fitted parameter (uniformity, TH-A2); (c) when it cannot be fixed: information about the texture
scale saturates when the ROI spans few correlation lengths (TH-B), hence the gate N/xi^3 >= 20; (d) the fix: block jackknife.
Real-CT numbers in (a) are the paper's 123-voxel result. Run in Research/ISBI/texture: python fig_overview_ipmi.py"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from scipy.ndimage import gaussian_filter
plt.rcParams.update({'font.size': 6.5, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5})
BLUE, RED, INK, BLOCK = '#2a78d6', '#e34948', '#1f1e1c', '#d9d7cf'
T = pd.read_csv('theory_sim.csv'); F = json.load(open('theory_fisher.json'))['rows']
fig = plt.figure(figsize=(4.8, 2.75))
def title(x, y, s): fig.text(x, y, s, fontsize=7, fontweight='bold', va='top')
# (a)
title(0.0, 1.0, '(a) Small ROIs look homogeneous')
a = fig.add_axes([0.0, 0.53, 0.47, 0.37]); a.set_xlim(0, 10); a.set_ylim(0, 6); a.set_aspect('equal'); a.axis('off')
img = gaussian_filter(np.random.default_rng(3).normal(size=(40, 40)), 0.9)
a.imshow(img, cmap='gray', extent=(0, 5.6, 0.2, 5.8), interpolation='nearest')
a.add_patch(Circle((2.8, 3.0), 2.4, fill=False, ec='white', lw=0.9, ls='--')); a.add_patch(Circle((3.6, 3.9), 0.6, fill=False, ec='#ffd166', lw=1.4))
a.text(6.0, 5.7, 'real CT, 123-voxel\nROIs, joint entropy:', fontsize=6, va='top', linespacing=1.1)
a.text(6.0, 3.0, 'plug-in', fontsize=6, va='center'); a.text(6.0, 2.3, '$-$0.63 bits', fontsize=6.6, va='center', color=BLUE, fontweight='bold')
a.text(6.0, 1.4, 'block jackknife', fontsize=6, va='center'); a.text(6.0, 0.7, '$-$0.01 bits', fontsize=6.6, va='center', color=RED, fontweight='bold')
# (b)
title(0.53, 1.0, '(b) Why: bias $=c/N$, $c$ set by correlation')
b = fig.add_axes([0.63, 0.58, 0.33, 0.30]); u = T[T.feature == 'Uniformity']
for s, c in zip(sorted(u.sigma.unique()), ('#9ecae1', '#4292c6', '#08306b')):
    g = u[u.sigma == s]; b.errorbar(g.pred.abs(), g.meas.abs(), yerr=2 * g.se, fmt='o', ms=2.5, lw=0.6, color=c, label=r'$\xi$=%g' % (2 * s))
m = [u.pred.abs().min() * .7, u.pred.abs().max() * 1.3]; b.plot(m, m, 'k--', lw=0.6); b.set_xscale('log'); b.set_yscale('log')
b.set_xlabel('predicted bias (closed form)', fontsize=6, labelpad=1); b.set_ylabel('simulated', fontsize=6, labelpad=1); b.tick_params(labelsize=5.5, length=2, pad=1)
b.legend(fontsize=5.3, frameon=False, handletextpad=0.2, borderaxespad=0.1, loc='upper left'); b.text(0.98, 0.04, 'no fitted\nparameter', transform=b.transAxes, fontsize=5.5, ha='right', va='bottom')
# (c)
title(0.0, 0.43, '(c) When no ROI method can work')
c = fig.add_axes([0.09, 0.13, 0.30, 0.24])
for xi, col in zip((0.5, 1, 2, 4, 8), plt.cm.viridis(np.linspace(0, .9, 5))):
    r = [o for o in F if o['nu'] == 0.5 and o['xi'] == xi]; c.plot([o['N'] for o in r], [o['I_indep'] for o in r], 'o-', ms=2, lw=0.8, color=col, label=r'$\xi$=%g' % xi)
c.plot([30, 2200], [30, 2200], 'k--', lw=0.6); c.set_xscale('log'); c.set_yscale('log')
c.set_xlabel('ROI voxels $N$', fontsize=6, labelpad=1); c.set_ylabel('worth (indep. voxels)', fontsize=6, labelpad=1); c.tick_params(labelsize=5.5, length=2, pad=1)
c.legend(fontsize=5, frameon=False, handlelength=1, handletextpad=0.2, borderaxespad=0.1, loc='center left', bbox_to_anchor=(1.0, 0.5))
c.annotate('coarse texture: 2,109\nvoxels are worth 11', (2109, 11), (36, 300), fontsize=5.6, arrowprops=dict(arrowstyle='->', lw=0.5, color='#555'))
# (d)
title(0.53, 0.43, '(d) Fix: delete-a-block jackknife')
d = fig.add_axes([0.53, 0.05, 0.47, 0.33]); d.set_xlim(0, 10); d.set_ylim(0, 6); d.set_aspect('equal'); d.axis('off')
d.add_patch(Circle((1.8, 3.6), 1.6, fc=BLOCK, ec=INK, lw=0.6)); d.plot([0.2, 3.4], [3.6, 3.6], color=INK, lw=0.4); d.plot([1.8, 1.8], [2.0, 5.2], color=INK, lw=0.4)
d.text(1.8, 1.6, 'ROI in $B$ blocks', ha='center', va='top', fontsize=6)
for j, (x0, y0) in enumerate([(5.6, 4.6), (7.9, 4.6), (5.6, 2.5), (7.9, 2.5)]):
    for k, (t1, t2) in enumerate([(90, 180), (0, 90), (180, 270), (270, 360)]):
        d.add_patch(Wedge((x0, y0), 0.9, t1, t2, fc='white' if k == j else BLOCK, ec=INK, lw=0.35))
d.annotate('', (4.4, 3.6), (3.7, 3.6), arrowprops=dict(arrowstyle='->', lw=0.6, color=INK))
d.text(6.75, 1.25, 'drop one block, recompute,\ncombine: cancels $c/N$ without\nknowing $c$; wraps any extractor', ha='center', va='top', fontsize=5.8)
fig.savefig('ipmi/figures/fig_overview_ipmi.pdf', pad_inches=0.02); fig.savefig('fig_overview_ipmi.png', dpi=220, pad_inches=0.02)
