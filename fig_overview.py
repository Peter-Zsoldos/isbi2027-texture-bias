"""Paper 2, overview figure (column width, two rows): (a) small ROIs look homogeneous, (b) block jackknife: recompute
without each block, (c) the reliability gate N/xi^3 >= 20. The texture in (a) is an illustrative Gaussian random field;
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

fig = plt.figure(figsize=(3.4, 1.85))
a = fig.add_axes([0.0, 0.29, 0.48, 0.64]); b = fig.add_axes([0.52, 0.29, 0.48, 0.64])
g = fig.add_axes([0.0, 0.0, 1.0, 0.2])
for ax in (a, b):
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.set_aspect('equal'); ax.axis('off')
g.set_xlim(0, 20); g.set_ylim(0, 2.2); g.axis('off')
fig.text(0.0, 0.97, '(a) small-ROI bias', fontsize=7.2, fontweight='bold', va='top')
fig.text(0.52, 0.97, '(b) block jackknife', fontsize=7.2, fontweight='bold', va='top')
fig.text(0.0, 0.245, '(c) reliability gate', fontsize=7.2, fontweight='bold', va='center')

# (a) texture with a large and a small ROI, next to the real-CT numbers
a.imshow(img, cmap='gray', extent=(0, 5.6, 0.7, 6.3), interpolation='nearest')
a.add_patch(Circle((2.8, 3.5), 2.4, fill=False, ec='white', lw=1.0, ls='--'))
a.add_patch(Circle((3.6, 4.4), 0.6, fill=False, ec='#ffd166', lw=1.5))
a.text(6.0, 6.2, 'small ROI\n(123 voxels),\nreal-CT\nentropy bias:', fontsize=6.0, va='top', color=INK, linespacing=1.15)
a.text(6.0, 2.55, 'plug-in', fontsize=6.0, va='center', color=INK)
a.text(6.0, 1.95, '−0.63 bits', fontsize=6.4, va='center', color=BLUE, fontweight='bold')
a.text(6.0, 1.25, 'jackknife', fontsize=6.0, va='center', color=INK)
a.text(6.0, 0.65, '−0.01 bits', fontsize=6.4, va='center', color=RED, fontweight='bold')

# (b) leave-one-block-out on a 2D sketch of the octants
b.add_patch(Circle((2.0, 4.2), 1.8, fc=BLOCK, ec=INK, lw=0.7))
b.plot([0.2, 3.8], [4.2, 4.2], color=INK, lw=0.5); b.plot([2.0, 2.0], [2.4, 6.0], color=INK, lw=0.5)
b.text(2.0, 1.9, 'ROI in $B{=}8$\nblocks', ha='center', va='top', fontsize=6.0)
for j, (x0, y0) in enumerate([(6.0, 5.4), (8.6, 5.4), (6.0, 3.0), (8.6, 3.0)]):
    for k, (t1, t2) in enumerate([(90, 180), (0, 90), (180, 270), (270, 360)]):
        b.add_patch(Wedge((x0, y0), 1.0, t1, t2, fc='white' if k == j else BLOCK, ec=INK, lw=0.4))
b.annotate('', (4.6, 4.2), (4.0, 4.2), arrowprops=dict(arrowstyle='->', lw=0.6, color=INK))
b.text(7.3, 1.75, 'drop one, recompute,\ncombine: cancels $c/N$', ha='center', va='top', fontsize=6.0)

# (c) gate, drawn as a horizontal flow
g.add_patch(FancyBboxPatch((6.2, 0.55), 4.6, 1.1, boxstyle='round,pad=0.12', fc=GREYF, ec=INK, lw=0.6))
g.text(8.5, 1.1, r'$N/\xi^3\geq20$ ?', ha='center', va='center', fontsize=7)
g.annotate('', (13.0, 1.55), (11.0, 1.35), arrowprops=dict(arrowstyle='->', lw=0.6, color=INK))
g.annotate('', (13.0, 0.65), (11.0, 0.85), arrowprops=dict(arrowstyle='->', lw=0.6, color=INK))
g.text(12.0, 1.85, 'yes', fontsize=5.8, ha='center', va='bottom')
g.text(12.0, 0.35, 'no', fontsize=5.8, ha='center', va='top')
g.text(13.2, 1.6, 'jackknife value', fontsize=6.4, va='center', color=RED, fontweight='bold')
g.text(13.2, 0.6, 'plug-in + "unreliable" flag', fontsize=6.4, va='center', color=BLUE, fontweight='bold')
g.text(0.0, 0.55, r'$\xi$: correlation length' + '\nmeasured in the ROI', fontsize=5.8, va='center', color='#555')
fig.savefig('paper/figures/fig_overview.pdf', bbox_inches='tight'); fig.savefig('fig_overview.png', dpi=250, bbox_inches='tight')
