"""Experimental ridge-regularized rally-level logistic adjusted plus-minus."""
from collections import Counter
import numpy as np


def fit_apm(records, penalty=25.0, pair_penalty=80.0):
    """One row per rally, with four distinct players and a team-A result.

    Reference coding: +1 for team A, -1 for team B; team intercept
    (serve_team_a) and pair interactions. Coefficients are log-odds effects.
    """
    clean, seen = [], set()
    for i, r in enumerate(records):
        players = [str(r.get(k, '')).strip() for k in ('player', 'partner', 'opponent', 'opponent2')]
        if not all(players) or len(set(players)) != 4:
            raise ValueError(f'Row {i+1}: four distinct named players are required')
        mid, rid = str(r.get('match_id', '')).strip(), str(r.get('rally_id', '')).strip()
        if not mid or not rid:
            raise ValueError(f'Row {i+1}: match_id and rally_id are required')
        key = (mid, rid)
        if key in seen:
            raise ValueError(f'Row {i+1}: duplicate match_id/rally_id; use one record per rally')
        seen.add(key)
        if not isinstance(r.get('won'), bool):
            raise ValueError(f'Row {i+1}: won must be boolean')
        serve = str(r.get('serve_team', '')).strip().upper()
        if serve not in ('A', 'B'):
            raise ValueError(f'Row {i+1}: serve_team must be A or B')
        clean.append((players, float(r['won']), 1.0 if serve == 'A' else -1.0))
    if len(clean) < 20:
        raise ValueError('At least 20 distinct rallies are required')
    names = sorted({p for ps, _, _ in clean for p in ps})
    if len(names) < 5:
        raise ValueError('At least five distinct players are required for partner variation')
    counts = Counter(p for ps, _, _ in clean for p in ps)
    partners = {p:set() for p in names}
    for ps, _, _ in clean:
        partners[ps[0]].add(ps[1]); partners[ps[1]].add(ps[0]); partners[ps[2]].add(ps[3]); partners[ps[3]].add(ps[2])
    if not any(len(v)>1 for v in partners.values()):
        raise ValueError('Players must rotate partners to identify individual effects')
    pairs = sorted({tuple(sorted((ps[0],ps[1]))) for ps,_,_ in clean} | {tuple(sorted((ps[2],ps[3]))) for ps,_,_ in clean})
    pi = {p:i for i,p in enumerate(names)}; qi={p:i for i,p in enumerate(pairs)}
    n, k, m = len(clean), len(names), len(pairs)
    X = np.zeros((n, 2+k+m), dtype=float); y=np.zeros(n)
    X[:,0]=1.0
    for i,(ps,outcome,serve) in enumerate(clean):
        X[i,1]=serve; y[i]=outcome
        for p,s in zip(ps,(1,1,-1,-1)): X[i,2+pi[p]]=s
        X[i,2+k+qi[tuple(sorted(ps[:2]))]]=1
        X[i,2+k+qi[tuple(sorted(ps[2:]))]]=-1
    reg=np.array([0.001,penalty]+[penalty]*k+[pair_penalty]*m,dtype=float)
    beta=np.zeros(X.shape[1]); converged=False
    for _ in range(75):
        z=np.clip(X@beta,-30,30); pred=1/(1+np.exp(-z))
        w=np.maximum(pred*(1-pred),1e-8)
        grad=X.T@(y-pred)-reg*beta
        H=(X.T*w)@X+np.diag(reg)
        step=np.linalg.solve(H,grad)
        beta+=step
        if np.max(np.abs(step))<1e-7: converged=True; break
    z=np.clip(X@beta,-30,30); pred=1/(1+np.exp(-z))
    w=np.maximum(pred*(1-pred),1e-8)
    cov=np.linalg.inv((X.T*w)@X+np.diag(reg))
    # Express a player's effect on a 50/50 rally as change in win probability
    # when replacing a zero-effect player, keeping all else equal.
    def effect(v): return (1/(1+np.exp(-np.clip(v,-30,30)))-0.5)*100
    players_out=[]
    for p in names:
        j=2+pi[p]; b=float(beta[j]); se=float(np.sqrt(max(cov[j,j],0)))
        players_out.append({'player':p,'rallies':counts[p], 'partners':len(partners[p]),
            'adjusted_impact_pp':round(effect(b),2),
            'regularized_interval_pp':[round(effect(b-1.96*se),2),round(effect(b+1.96*se),2)]})
    pairs_out=[]
    for pair in pairs:
        j=2+k+qi[pair]; b=float(beta[j]); se=float(np.sqrt(max(cov[j,j],0)))
        pair_count=sum(tuple(sorted(ps[:2]))==pair or tuple(sorted(ps[2:]))==pair for ps,_,_ in clean)
        pairs_out.append({'players':list(pair),'rallies':pair_count,'adjusted_synergy_pp':round(effect(b),2),
            'regularized_interval_pp':[round(effect(b-1.96*se),2),round(effect(b+1.96*se),2)]})
    return {'method':'Ridge-regularized logistic adjusted plus-minus (experimental)',
        'rallies':n,'players':sorted(players_out,key=lambda v:-v['adjusted_impact_pp']),
        'partnerships':sorted(pairs_out,key=lambda v:-v['adjusted_synergy_pp']),
        'serve_advantage_pp':round(effect(float(beta[1])),2), 'converged':converged,
        'caveats':['Coefficients estimate context-adjusted rally impact, not WAR or causal effects.',
          'Intervals are approximate penalized-Hessian intervals, NOT calibrated confidence intervals.',
          'Sparse or disconnected player networks, selection effects, court side and match context can bias estimates.',
          'Players and partnerships are heavily shrunk toward zero; the baseline is a hypothetical 50% rally.',
          'Requires one verified rally per row, all four players, and serving team. Not derived from video analysis.']}
