"""Figure for the IPMI version: (a,b) closed-form vs simulated plug-in bias (TH-A2), (c) ROI information about the texture scale (TH-B)."""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T = pd.read_csv('theory_sim.csv'); F = json.load(open('theory_fisher.json'))['rows']
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1)); col = {0.75: '#9ecae1', 1.5: '#4292c6', 3.0: '#08306b'}
for i, f in enumerate(('Uniformity', 'Entropy')):
    a = ax[i]; t = T[T.feature == f]
    for s, g in t.groupby('sigma'):
        hi, lo = g[g.neff >= 20], g[g.neff < 20]
        a.errorbar(hi.pred.abs(), hi.meas.abs(), yerr=2 * hi.se, fmt='o', ms=4, color=col[s], label=r'$\xi$=%g' % (2 * s))
        a.errorbar(lo.pred.abs(), lo.meas.abs(), yerr=2 * lo.se, fmt='o', ms=4, mfc='white', mec=col[s], ecolor=col[s])
    m = [t.pred.abs().min() * .7, t.pred.abs().max() * 1.3]; a.plot(m, m, 'k--', lw=.8); a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel('predicted |bias| (closed form)'); a.set_ylabel('simulated |bias|')
    a.set_title(('(a) ' if i == 0 else '(b) ') + f + (' (bits)' if f == 'Entropy' else ''), fontsize=9)
ax[0].legend(fontsize=7, frameon=False, title=r'open: $N/\xi^3<20$', title_fontsize=7)
a = ax[2]
for xi, c in zip((0.5, 1, 2, 4, 8), plt.cm.viridis(np.linspace(0, .9, 5))):
    r = [o for o in F if o['nu'] == 0.5 and o['xi'] == xi]; a.plot([o['N'] for o in r], [o['I_indep'] for o in r], 'o-', ms=3, color=c, label=r'$\xi$=%g' % xi)
a.plot([30, 2200], [30, 2200], 'k--', lw=.8); a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(r'ROI voxels $N$'); a.set_ylabel('information (independent voxels)')
a.set_title(r'(c) information about scale, $\xi$ unknown', fontsize=9); a.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig('ipmi/figures/fig_theory.pdf'); plt.savefig('fig_theory.png', dpi=110)
