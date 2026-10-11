"""Paper 2, overview figure (column width): (a) small ROIs look homogeneous, (b) block jackknife: recompute without
each block, (c) the reliability gate N/xi^3 >= 20. The texture in (a) is an illustrative Gaussian random field;
the numbers quoted are the paper's real-CT result for 123-voxel ROIs (Sec. 5)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Wedge
from scipy.ndimage import gaussian_filter

plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'pdf.fonttype': 42})
BLUE, RED, INK, GREYF, BLOCK = '#2a78d6', '#e34948', '#1f1e1c', '#efeee9', '#d9d7cf'

rng = np.random.default_rng(3)
img = gaussian_filter(rng.normal(size=(40, 40)), 0.9)

fig = plt.figure(figsize=(3.4, 1.6))
W = [0.0, 0.355, 0.69]                      # left edges of the three panels (figure fraction)
axs = [fig.add_axes([x, 0.0, 0.31, 0.8]) for x in W]
for ax, x, t in zip(axs, W, ['(a) small ROIs look\n      homogeneous', '(b) block jackknife', '(c) reliability gate']):
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.9); ax.set_aspect('equal'); ax.axis('off')
    fig.text(x, 0.93, t, fontsize=7, fontweight='bold', va='top')

# (a) texture with a large and a small ROI
a = axs[0]
a.imshow(img, cmap='gray', extent=(0, 10, 3.4, 13.4), interpolation='nearest')
a.add_patch(Circle((5, 8.4), 4.2, fill=False, ec='white', lw=1.0, ls='--'))
a.add_patch(Circle((6.4, 10.0), 1.0, fill=False, ec='#ffd166', lw=1.5))
a.text(0.0, 2.9, 'small ROI (123 voxels),\nreal CT entropy bias:', fontsize=5.8, va='top', color=INK)
a.text(0.0, 0.75, 'plug-in', fontsize=6.2, va='center', color=INK)
a.text(5.0, 0.75, '−0.63 bits', fontsize=6.2, va='center', color=BLUE, fontweight='bold')
a.text(0.0, -0.45, 'jackknife', fontsize=6.2, va='center', color=INK)
a.text(5.0, -0.45, '−0.01 bits', fontsize=6.2, va='center', color=RED, fontweight='bold')

# (b) leave-one-block-out on a 2D sketch of the octants
b = axs[1]
b.add_patch(Circle((2.3, 8.4), 2.1, fc=BLOCK, ec=INK, lw=0.7))
b.plot([0.2, 4.4], [8.4, 8.4], color=INK, lw=0.5); b.plot([2.3, 2.3], [6.3, 10.5], color=INK, lw=0.5)
b.text(2.3, 5.8, 'ROI split\ninto $B{=}8$\nblocks', ha='center', va='top', fontsize=5.8)
for j, (x0, y0) in enumerate([(6.4, 10.4), (8.9, 10.4), (6.4, 7.6), (8.9, 7.6)]):
    for k, (t1, t2) in enumerate([(90, 180), (0, 90), (180, 270), (270, 360)]):
        b.add_patch(Wedge((x0, y0), 1.05, t1, t2, fc='white' if k == j else BLOCK, ec=INK, lw=0.4))
b.annotate('', (5.0, 8.9), (4.5, 8.9), arrowprops=dict(arrowstyle='->', lw=0.6, color=INK))
b.text(7.65, 5.8, 'drop one,\nrecompute,\ncombine', ha='center', va='top', fontsize=5.8)
b.text(5.0, 0.6, 'cancels the $c/N$ bias\nfor any $c$', ha='center', va='top', fontsize=5.8, color='#444')

# (c) reliability gate
g = axs[2]
g.add_patch(FancyBboxPatch((0.6, 9.2), 8.8, 2.2, boxstyle='round,pad=0.2', fc=GREYF, ec=INK, lw=0.6))
g.text(5.0, 10.3, r'$N/\xi^3\geq20$ ?', ha='center', va='center', fontsize=7)
g.text(5.0, 8.2, r'$\xi$: correlation length', ha='center', va='center', fontsize=5.6, color='#555')
for x, lab, txt, col in [(2.4, 'yes', 'jackknife', RED), (7.6, 'no', 'plug-in +\nflag', BLUE)]:
    g.annotate('', (x, 4.9), (x, 7.4), arrowprops=dict(arrowstyle='->', lw=0.6, color=INK))
    g.text(x + (0.4 if x < 5 else -0.4), 6.2, lab, fontsize=5.8, ha='left' if x < 5 else 'right', va='center')
    g.text(x, 4.5, txt, ha='center', va='top', fontsize=6.4, color=col, fontweight='bold')
g.text(5.0, 0.6, 'outside the $1/N$ regime\nno estimator can recover', ha='center', va='top', fontsize=5.8, color='#444')
fig.savefig('paper/figures/fig_overview.pdf', bbox_inches='tight'); fig.savefig('fig_overview.png', dpi=250, bbox_inches='tight')
