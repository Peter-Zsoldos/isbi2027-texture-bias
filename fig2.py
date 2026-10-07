import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 6.5, 'font.family': 'serif', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'legend.fontsize': 5.4})
BLUE, RED, GREY, AQUA, VIOLET = '#2a78d6', '#e34948', '#7a7974', '#1baf7a', '#4a3aa7'
fig, ax = plt.subplots(1, 4, figsize=(7.1, 1.6), gridspec_kw=dict(wspace=0.45, width_ratios=[1.15, 1.0, 1.0, 0.9]))

# (a) synthetic: JointEntropy bias vs effective sample size N/xi^3
df = pd.read_csv('bench_full.csv'); df['xi'] = 2 * df.sigma
df['neff'] = (4 / 3 * np.pi * df.r ** 3) / df.xi ** 3
a = ax[0]
for sig, mk in [(0.75, 'o'), (1.5, 's'), (3.0, '^')]:
    for meth, c in [('plugin', BLUE), ('JK', RED)]:
        d = df[(df.feature == 'JointEntropy') & (df.sigma == sig) & (df.method == meth)]
        g = d.groupby('r').agg(e=('err', 'mean'), n=('neff', 'first')).sort_values('n')
        a.plot(g.n, g.e, marker=mk, ms=2.4, lw=0.8, color=c,
               label=(('plug-in' if meth == 'plugin' else 'block JK (ours)') if sig == 0.75 else None))
a.axhline(0, color='#bbb', lw=0.5, zorder=0); a.axvline(20, color=GREY, lw=0.6, ls=':')
a.set_xscale('log'); a.set_xlabel(r'effective sample size $N/\xi^3$'); a.set_ylabel('JointEntropy bias (bits)')
a.set_title('(a) synthetic texture', fontsize=6.5); a.set_ylim(-2.3, 0.3)
a.legend(loc='lower right', frameon=False, handlelength=1.3, borderaxespad=0, labelspacing=0.12, fontsize=4.8, ncol=1)

# (b) real CT soft tissue, four correlation regimes: JointEntropy bias vs effective sample size
R = pd.read_csv('real_all_smooth.csv'); b = ax[1]
for s, mk in [(0.0, 'o'), (0.7, 's'), (1.4, '^'), (2.0, 'D')]:
    for meth, c in [('plugin', BLUE), ('JK', RED)]:
        d = R[(R.feature == 'JointEntropy') & (R.method == meth) & (R.smooth == s)]
        g = d.groupby('r').agg(e=('err', 'mean'), n=('neff', 'median'), se=('err', 'sem')).sort_values('n')
        b.errorbar(g.n, g.e, yerr=1.96 * g.se, marker=mk, ms=2.4, lw=0.8, color=c, capsize=1.0, elinewidth=0.5,
                   label=(('plug-in' if meth == 'plugin' else 'block JK (ours)') if s == 0.0 else None))
b.axhline(0, color='#bbb', lw=0.5, zorder=0); b.axvline(20, color=GREY, lw=0.6, ls=':')
b.set_xscale('log'); b.set_xlabel(r'effective sample size $N/\xi^3$'); b.set_ylabel('JointEntropy bias (bits)')
b.set_title('(b) real CT, 4 correlation regimes', fontsize=6.5)
b.legend(loc='lower right', bbox_to_anchor=(1.0, 0.18), frameon=False, handlelength=1.3, borderaxespad=0, labelspacing=0.12, fontsize=4.8)

# (c) all PyRadiomics features: |bias| plug-in vs JK (SD units), r=4
T = pd.read_csv('pyrad_summary.csv'); t = T[T.r == 4]; c = ax[2]
for ty, col, lab in [('A', RED, 'smooth (hist./GLCM)'), ('B', GREY, 'quantile'), ('C', VIOLET, 'region-geometry')]:
    s = t[t.type.str.startswith(ty)]
    c.scatter(s.bias_p.abs(), s.bias_j.abs(), s=5, color=col, label=lab, lw=0, alpha=0.85)
lim = [1e-3, 10]; c.plot(lim, lim, color='#999', lw=0.6, ls='--'); c.set_xscale('log'); c.set_yscale('log')
c.set_xlim(lim); c.set_ylim(lim); c.set_xlabel('|bias| plug-in (SD)'); c.set_ylabel('|bias| block JK (SD)')
c.set_title('(c) PyRadiomics, r=4', fontsize=6.5)
c.legend(loc='upper left', frameon=False, handlelength=1, borderaxespad=0, labelspacing=0.1, fontsize=4.8, markerscale=1.5)

# (d) bin anchoring
A = pd.read_csv('anchoring.csv'); d = ax[3]
g = A.groupby('r')[['roi_min', 'fixed']].mean()
d.plot(g.index, g.roi_min, marker='o', ms=2.4, lw=0.8, color=BLUE, label='ROI-min anchor\n(PyRadiomics)')
d.plot(g.index, g.fixed, marker='P', ms=2.4, lw=0.8, color=RED, label='fixed anchor (ours)')
d.axhline(0, color='#bbb', lw=0.5, zorder=0); d.set_xlabel('ROI radius (voxels)'); d.set_ylabel('mean grey-level index bias')
d.set_title('(d) bin anchoring', fontsize=6.5)
d.legend(loc='center right', bbox_to_anchor=(1.0, 0.7), frameon=False, handlelength=1.3, borderaxespad=0, labelspacing=0.2, fontsize=4.8)
fig.savefig('paper/figures/fig_main.pdf', bbox_inches='tight'); fig.savefig('fig2.png', dpi=250, bbox_inches='tight')
