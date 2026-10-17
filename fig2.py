"""Paper 2 result figures (at most two panels per row):
fig_bias.pdf  (full width, 2x2) rows: synthetic / real CT; columns: plug-in / block jackknife;
              JointEntropy bias vs effective sample size N/xi^3, one curve per correlation regime
              (lightness and marker = correlation length; blue = plug-in, red = jackknife);
fig_pyrad.pdf (column)     all 93 PyRadiomics features, |bias| plug-in vs jackknife."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'legend.fontsize': 6.6,
                     'axes.titlesize': 7.8, 'axes.titleweight': 'bold'})
GREY, VIOLET, SHADE = '#7a7974', '#6250d6', '#efeee9'
AQUA, DYEL = '#199e70', '#c98500'   # feature types: hues not used for methods
# light -> dark = short -> long correlation length; blue ramp for plug-in, red ramp for the jackknife
RAMP = {'plugin': ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b'], 'JK': ['#f0a09f', '#e34948', '#a8282a', '#6b1617']}
MK = ['o', 's', '^', 'D']


def panel(ax, curves, meth, xis, title, ylim, ci=False):
    ax.axvspan(1e-1, 20, color=SHADE, zorder=0, lw=0)
    ax.axvline(20, color=GREY, lw=0.7, ls=':')
    ax.axhline(0, color='#999', lw=0.7, zorder=1)
    hs = []
    for (n, e, se), xi, col, mk in zip(curves, xis, RAMP[meth][-len(xis):] if len(xis) < 4 else RAMP[meth], MK):
        if ci:
            ax.errorbar(n, e, yerr=1.96 * se, color=col, lw=1.2, marker=mk, ms=3.6, elinewidth=0.6, capsize=0)
        else:
            ax.plot(n, e, color=col, lw=1.2, marker=mk, ms=3.6)
        hs.append(Line2D([], [], color=col, lw=1.2, marker=mk, ms=3.6, label=xi))
    ax.set_xscale('log'); ax.set_xlim(0.3, 8e3); ax.set_ylim(*ylim)
    ax.text(0.35, ylim[1] - 0.04 * (ylim[1] - ylim[0]), r'flagged: $N/\xi^3<20$', fontsize=6.3, color='#666', va='top')
    ax.legend(handles=hs, loc='lower right', ncol=len(xis), frameon=False, handlelength=1.5, handletextpad=0.3,
              columnspacing=0.9, borderaxespad=0.2, title=r'correlation length $\xi$ (voxels)', title_fontsize=6.6)
    ax.set_title(title, loc='left')


# data: synthetic (3 regimes) and real CT (4 regimes), one curve per regime and method
df = pd.read_csv('bench_full.csv'); df['xi'] = 2 * df.sigma
df['neff'] = (4 / 3 * np.pi * df.r ** 3) / df.xi ** 3
syn = {m: [] for m in RAMP}
for sig in [0.75, 1.5, 3.0]:
    for m in RAMP:
        d = df[(df.feature == 'JointEntropy') & (df.sigma == sig) & (df.method == m)]
        g = d.groupby('r').agg(e=('err', 'mean'), n=('neff', 'first')).sort_values('n')
        syn[m].append((g.n, g.e, None))
R = pd.read_csv('real_all_smooth.csv')
ct = {m: [] for m in RAMP}
for s in [0.0, 0.7, 1.4, 2.0]:
    for m in RAMP:
        d = R[(R.feature == 'JointEntropy') & (R.method == m) & (R.smooth == s)]
        g = d.groupby('r').agg(e=('err', 'mean'), n=('neff', 'median'), se=('err', 'sem')).sort_values('n')
        ct[m].append((g.n, g.e, g.se))

fig, ax = plt.subplots(2, 2, figsize=(7.1, 2.8), sharey='row', gridspec_kw=dict(wspace=0.08, hspace=0.62))
panel(ax[0, 0], syn['plugin'], 'plugin', ['1.5', '3', '6'], '(a) synthetic texture: plug-in (PyRadiomics)', (-2.3, 0.35))
panel(ax[0, 1], syn['JK'], 'JK', ['1.5', '3', '6'], '(b) synthetic texture: block jackknife (ours)', (-2.3, 0.35))
panel(ax[1, 0], ct['plugin'], 'plugin', ['0.9', '1.6', '3.6', '6.4'],
      '(c) real CT soft tissue: plug-in (PyRadiomics)', (-1.15, 0.2), ci=True)
panel(ax[1, 1], ct['JK'], 'JK', ['0.9', '1.6', '3.6', '6.4'],
      '(d) real CT soft tissue: block jackknife (ours)', (-1.15, 0.2), ci=True)
for a in ax[:, 0]:
    a.set_ylabel('JointEntropy bias (bits)')
for a in ax[1]:
    a.set_xlabel(r'effective sample size $N/\xi^3$')
fig.savefig('paper/figures/fig_bias.pdf', bbox_inches='tight'); fig.savefig('fig_bias.png', dpi=220, bbox_inches='tight')

# all PyRadiomics features: |bias| plug-in vs JK (reference-SD units), r=4
T = pd.read_csv('pyrad_summary.csv'); t = T[T.r == 4]
fig, c = plt.subplots(figsize=(3.3, 2.1))
for ty, col, mk, lab in [('A', AQUA, 'o', 'histogram / GLCM (smooth)'), ('B', DYEL, 's', 'quantile-based'),
                         ('C', VIOLET, '^', 'run / zone (region geometry)')]:
    s = t[t.type.str.startswith(ty)]
    c.scatter(s.bias_p.abs(), s.bias_j.abs(), s=14, color=col, marker=mk, label=lab, lw=0, alpha=0.85)
lim = [1e-3, 10]; c.plot(lim, lim, color='#999', lw=0.7, ls='--'); c.set_xscale('log'); c.set_yscale('log')
c.fill_between(lim, [1e-3, 1e-3], lim, color='#f1f0ec', zorder=0, lw=0)
c.text(0.97, 0.05, 'below the diagonal:\njackknife reduces bias', transform=c.transAxes, ha='right', fontsize=6.6,
       color='#555')
c.set_xlim(lim); c.set_ylim(lim); c.set_xlabel('|bias| of plug-in (reference s.d.)'); c.set_ylabel('|bias| of jackknife')
c.legend(loc='upper left', frameon=False, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25, markerscale=1.1)
fig.savefig('paper/figures/fig_pyrad.pdf', bbox_inches='tight'); fig.savefig('fig_pyrad.png', dpi=250, bbox_inches='tight')
