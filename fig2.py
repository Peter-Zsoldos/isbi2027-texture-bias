"""Paper 2, Figure 1 (2x2): (a) synthetic and (b) real-CT JointEntropy bias vs effective sample size,
(c) all PyRadiomics features plug-in vs jackknife, (d) bin-anchoring artefact."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'legend.fontsize': 6.6,
                     'axes.titlesize': 7.8, 'axes.titleweight': 'bold'})
BLUE, RED, GREY, VIOLET, SHADE = '#2a78d6', '#e34948', '#7a7974', '#6250d6', '#efeee9'
AQUA, DYEL = '#199e70', '#c98500'   # feature types in (c): hues not used for methods
fig, ax = plt.subplots(2, 2, figsize=(7.1, 2.7), gridspec_kw=dict(wspace=0.24, hspace=0.75))
METH = [('plugin', BLUE, 'plug-in (PyRadiomics)'), ('JK', RED, 'block jackknife (ours)')]


def regime_axes(a, title):
    a.axvspan(1e-1, 20, color=SHADE, zorder=0, lw=0)
    a.axvline(20, color=GREY, lw=0.7, ls=':')
    a.axhline(0, color='#aaa', lw=0.6, zorder=1)
    a.set_xscale('log'); a.set_xlim(0.3, 8e3); a.set_xlabel(r'effective sample size $N/\xi^3$')
    a.set_ylabel('JointEntropy bias (bits)'); a.set_title(title, loc='left')


def regime_legend(a, xis, mks):
    h = [Line2D([], [], color='#444', marker=m, ls='', ms=3.5, label=x) for x, m in zip(xis, mks)]
    a.legend(handles=h, loc='lower right', frameon=False, ncol=len(xis), handletextpad=0.1, columnspacing=0.9,
             borderaxespad=0.2, title=r'correlation length $\xi$ (voxels)', title_fontsize=6.6)


# (a) synthetic textures
df = pd.read_csv('bench_full.csv'); df['xi'] = 2 * df.sigma
df['neff'] = (4 / 3 * np.pi * df.r ** 3) / df.xi ** 3
a = ax[0, 0]; MK = ['o', 's', '^']
for sig, mk in zip([0.75, 1.5, 3.0], MK):
    for meth, c, _ in METH:
        d = df[(df.feature == 'JointEntropy') & (df.sigma == sig) & (df.method == meth)]
        g = d.groupby('r').agg(e=('err', 'mean'), n=('neff', 'first')).sort_values('n')
        a.plot(g.n, g.e, marker=mk, ms=3, lw=1.0, color=c, mfc='white' if meth == 'plugin' else c, mew=0.9)
regime_axes(a, '(a) synthetic texture'); a.set_ylim(-2.3, 0.3)
regime_legend(a, ['1.5', '3', '6'], MK)

# (b) real CT soft tissue, internal reference, four correlation regimes
R = pd.read_csv('real_all_smooth.csv'); b = ax[0, 1]; MK = ['o', 's', '^', 'D']
for s, mk in zip([0.0, 0.7, 1.4, 2.0], MK):
    for meth, c, _ in METH:
        d = R[(R.feature == 'JointEntropy') & (R.method == meth) & (R.smooth == s)]
        g = d.groupby('r').agg(e=('err', 'mean'), n=('neff', 'median'), se=('err', 'sem')).sort_values('n')
        b.errorbar(g.n, g.e, yerr=1.96 * g.se, marker=mk, ms=3, lw=1.0, color=c, mfc='white' if meth == 'plugin' else c,
                   mew=0.9, elinewidth=0.5, capsize=0)
regime_axes(b, '(b) real CT soft tissue (mean, 95% CI)'); b.set_ylim(-1.15, 0.12)
regime_legend(b, ['0.9', '1.6', '3.6', '6.4'], MK)

# (c) all PyRadiomics features: |bias| plug-in vs JK (reference-SD units), r=4
T = pd.read_csv('pyrad_summary.csv'); t = T[T.r == 4]; c = ax[1, 0]
for ty, col, mk, lab in [('A', AQUA, 'o', 'histogram / GLCM (smooth)'), ('B', DYEL, 's', 'quantile-based'),
                         ('C', VIOLET, '^', 'region geometry')]:
    s = t[t.type.str.startswith(ty)]
    c.scatter(s.bias_p.abs(), s.bias_j.abs(), s=10, color=col, marker=mk, label=lab, lw=0, alpha=0.85)
lim = [1e-3, 10]; c.plot(lim, lim, color='#999', lw=0.7, ls='--'); c.set_xscale('log'); c.set_yscale('log')
c.fill_between(lim, [1e-3, 1e-3], lim, color='#f1f0ec', zorder=0, lw=0)
c.text(0.97, 0.06, 'below diagonal:\njackknife reduces bias', transform=c.transAxes, ha='right', fontsize=6.4, color='#555')
c.set_xlim(lim); c.set_ylim(lim); c.set_xlabel('|bias|, plug-in (reference SD)'); c.set_ylabel('|bias|, jackknife')
c.set_title('(c) all 93 PyRadiomics features, radius 4', loc='left')
c.legend(loc='upper left', frameon=False, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.2, markerscale=1.2)

# (d) bin anchoring
A = pd.read_csv('anchoring.csv'); d = ax[1, 1]
g = A.groupby('r')[['roi_min', 'fixed']].mean()
d.plot(g.index, g.roi_min, marker='o', ms=3, lw=1.1, color=BLUE, label='ROI-minimum (PyRadiomics)')
d.plot(g.index, g.fixed, marker='P', ms=3.5, lw=1.1, color=RED, label='fixed anchor (ours)')
d.axhline(0, color='#aaa', lw=0.6, zorder=0); d.set_xlabel('ROI radius (voxels)'); d.set_ylabel('grey-level index bias')
d.set_title('(d) bin anchoring shifts small ROIs', loc='left')
d.legend(loc='lower right', frameon=False, handlelength=1.6, borderaxespad=0.2, labelspacing=0.2)

H = [Line2D([], [], color=c, lw=1.2, marker='o', ms=3.2, mfc='white' if m == 'plugin' else c, mew=0.9, label=l)
     for m, c, l in METH]
H.append(Patch(color=SHADE, label=r'$N/\xi^3<20$: ROI flagged unreliable'))
fig.legend(handles=H, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False, handlelength=1.8,
           columnspacing=1.8)
fig.savefig('paper/figures/fig_main.pdf', bbox_inches='tight'); fig.savefig('fig2.png', dpi=220, bbox_inches='tight')
