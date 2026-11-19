"""GATE-2 (PREREG_IPMI.md): model-derived information gate (I_eff from Prop. 3) vs N/xi^3. CPU. python gate2.py"""
import json, numpy as np, pandas as pd
F = json.load(open('theory_fisher.json'))['rows']; R = [r for r in F if r['nu'] == 0.5]
XI = sorted({r['xi'] for r in R}); FIT = {}
for x in XI:
    rr = sorted([r for r in R if r['xi'] == x], key=lambda r: r['N']); FIT[x] = np.polyfit(np.log([r['N'] for r in rr]), np.log([r['I_indep'] for r in rr]), 1)
def ieff(N, xi):
    lx = np.log(np.clip(xi, XI[0], XI[-1])); lX = np.log(XI); v = np.array([np.polyval(FIT[x], np.log(N)) for x in XI])   # per-xi log-log fit, then interpolate in log xi
    return float(np.exp(np.interp(lx, lX, v)))
NV = {3: 123, 4: 257, 5: 515, 6: 925, 8: 2109, 10: 4169, 14: 11513}
def wide(D, keys):
    return D[D.method.isin(['plugin', 'JK'])].pivot_table(index=keys + ['feature', 'ref'], columns='method', values='value').reset_index()
def objective(W, idx, T):
    out = []
    for f, g in W.groupby('feature'):
        gated = np.where(g[idx] >= T, g.JK, g.plugin); out.append(np.sqrt(((gated - g.ref) ** 2).mean()) / np.sqrt(((g.plugin - g.ref) ** 2).mean()))
    return float(np.mean(out))
S = pd.read_csv('bench_full.csv'); S['xi'] = 2 * S.sigma; S['N'] = S.r.map(NV)
Ws = wide(S, ['sigma', 'rep', 'r', 'xi', 'N']); Ws['old'] = Ws.N / Ws.xi ** 3; Ws['inf'] = [ieff(n, x) for n, x in zip(Ws.N, Ws.xi)]
res = {}
for idx in ('old', 'inf'):
    grid = np.quantile(Ws[idx], np.linspace(.05, .95, 19)); objs = [objective(Ws, idx, t) for t in grid]; res[idx] = dict(T=float(grid[int(np.argmin(objs))]), synth_obj=float(min(objs)))
D = pd.read_csv('real_all_smooth.csv'); Wr = wide(D, ['smooth', 'roi', 'r', 'xi', 'N']); Wr['old'] = Wr.N / Wr.xi ** 3; Wr['inf'] = [ieff(n, x) for n, x in zip(Wr.N, Wr.xi)]
for idx in ('old', 'inf'):
    per = {str(s): objective(g, idx, res[idx]['T']) for s, g in Wr.groupby('smooth')}; res[idx]['real_per_smooth'] = per; res[idx]['real_obj'] = float(np.mean(list(per.values())))
    res[idx]['open_frac'] = {str(s): float((g[idx] >= res[idx]['T']).mean()) for s, g in Wr.groupby('smooth')}
res['paper_gate_old_T20'] = dict(real_obj=float(np.mean([objective(g, 'old', 20) for s, g in Wr.groupby('smooth')])))
res['G1'] = res['inf']['real_obj'] <= res['old']['real_obj'] + 0.01; res['G2'] = res['inf']['real_obj'] <= res['old']['real_obj'] - 0.02
json.dump(res, open('gate2.json', 'w'), indent=1); print(json.dumps(res, indent=1))
