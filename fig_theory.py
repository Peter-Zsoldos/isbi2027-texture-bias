"""Figure for the IPMI version: (a,b) closed-form vs simulated plug-in bias (TH-A2), (c) ROI information about the texture scale (TH-B)."""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.minor.size': 1, 'ytick.minor.size': 1})
T = pd.read_csv('theory_sim.csv'); F = json.load(open('theory_fisher.json'))['rows']
fig, ax = plt.subplots(1, 3, figsize=(4.8, 1.75)); col = {0.75: '#9ecae1', 1.5: '#4292c6', 3.0: '#08306b'}
for i, f in enumerate(('Uniformity', 'Entropy')):
    a = ax[i]; t = T[T.feature == f]
    for s, g in t.groupby('sigma'):
        hi, lo = g[g.neff >= 20], g[g.neff < 20]
        a.errorbar(hi.pred.abs(), hi.meas.abs(), yerr=2 * hi.se, fmt='o', ms=2.2, lw=0.6, color=col[s], label=r'$\xi$=%g' % (2 * s))
        a.errorbar(lo.pred.abs(), lo.meas.abs(), yerr=2 * lo.se, fmt='o', ms=2.2, lw=0.6, mfc='white', mec=col[s], ecolor=col[s])
    m = [t.pred.abs().min() * .7, t.pred.abs().max() * 1.3]; a.plot(m, m, 'k--', lw=.8); a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel('predicted |bias| (closed form)'); a.set_ylabel('simulated |bias|')
    a.set_title(('(a) ' if i == 0 else '(b) ') + f + (' (bits)' if f == 'Entropy' else ''), fontsize=6.5)
ax[0].legend(fontsize=5.2, frameon=False, title=r'open: $N/\xi^3<20$', title_fontsize=5.2)
a = ax[2]
for xi, c in zip((0.5, 1, 2, 4, 8), plt.cm.viridis(np.linspace(0, .9, 5))):
    r = [o for o in F if o['nu'] == 0.5 and o['xi'] == xi]; a.plot([o['N'] for o in r], [o['I_indep'] for o in r], 'o-', ms=2, lw=0.8, color=c, label=r'$\xi$=%g' % xi)
a.plot([30, 2200], [30, 2200], 'k--', lw=.8); a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(r'ROI voxels $N$'); a.set_ylabel('information (independent voxels)')
a.set_title(r'(c) information about scale', fontsize=6.5); a.set_ylabel('worth (indep. voxels)'); a.set_xlim(right=9000)
for xi, c in zip((0.5, 1, 2, 4, 8), plt.cm.viridis(np.linspace(0, .9, 5))):
    o = [o for o in F if o['nu'] == 0.5 and o['xi'] == xi][-1]; a.text(o['N'] * 1.25, o['I_indep'], r'$\xi$=%g' % xi, fontsize=5.2, va='center', color=c)
plt.tight_layout(pad=0.2, w_pad=0.6); plt.savefig('ipmi/figures/fig_theory.pdf'); plt.savefig('fig_theory.png', dpi=110)
