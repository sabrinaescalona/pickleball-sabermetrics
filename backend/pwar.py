"""Experimental rally-scoring doubles PWAR using regularized APM and explicit baselines."""
from collections import defaultdict
from functools import lru_cache
import math
import numpy as np
from .adjusted_plus_minus import fit_apm


def match_probability(p, target=11):
    """Independent identically distributed rally outcomes, first to target win by two."""
    p = min(max(float(p), 1e-8), 1-1e-8)
    q = 1-p
    deuce = p*p/(p*p+q*q)
    @lru_cache(None)
    def f(a,b):
        if a>=target and a-b>=2: return 1.
        if b>=target and b-a>=2: return 0.
        if a==target-1 and b==target-1: return deuce
        return p*f(a+1,b)+q*f(a,b+1)
    return f(0,0)


def pwar(records, ratings, target=11, baseline_quantile=.20):
    if target not in (11,15,21): raise ValueError('Supported rally scoring targets: 11, 15, 21')
    fit=fit_apm(records)
    names=[x['player'] for x in fit['players']]
    missing=[n for n in names if n not in ratings]
    if missing: raise ValueError('DUPR/skill ratings required for every player: '+', '.join(missing))
    if any(not 2<=float(ratings[n])<=8 for n in names): raise ValueError('Ratings must be between 2.0 and 8.0')
    # APM effects are exposed as probabilities relative to zero; invert exactly to log-odds.
    effects={x['player']:math.log((.5+x['adjusted_impact_pp']/100)/(.5-x['adjusted_impact_pp']/100)) for x in fit['players']}
    # Important: coefficients are reconstructed from rounded effects, so precision is limited.
    groups=defaultdict(list)
    for n in names: groups[math.floor(float(ratings[n])*2)/2].append(n)
    universal=float(np.quantile(list(effects.values()),baseline_quantile))
    skill_baselines={g:float(np.quantile([effects[n] for n in ns],baseline_quantile)) for g,ns in groups.items() if len(ns)>=3}
    # Use only one match per player for each match_id; expected wins are summed over distinct matches.
    contexts=defaultdict(dict)
    for r in records:
        mid=str(r['match_id']); a,b,c,d=[r[k] for k in ('player','partner','opponent','opponent2')]
        for who,partner,opp1,opp2 in ((a,b,c,d),(b,a,c,d),(c,d,a,b),(d,c,a,b)):
            contexts[who].setdefault(mid,(partner,opp1,opp2))
    sig=lambda z:1/(1+math.exp(-max(-30,min(30,z))))
    result=[]
    for name in names:
        g=math.floor(float(ratings[name])*2)/2
        base_skill=skill_baselines.get(g)
        values={}
        for mode,baseline in (('skill_adjusted',base_skill),('universal',universal)):
            if baseline is None:
                values[mode]={'available':False,'reason':'At least 3 distinct players are needed in this 0.5 rating band'}
                continue
            total=0.; examples=[]
            for mid,(partner,o1,o2) in contexts[name].items():
                offset=effects[partner]-effects[o1]-effects[o2]
                actual=match_probability(sig(effects[name]+offset),target)
                repl=match_probability(sig(baseline+offset),target)
                total+=actual-repl
                examples.append({'match_id':mid,'actual_win_probability':round(actual,4),'replacement_win_probability':round(repl,4)})
            m=len(examples)
            values[mode]={'available':True,'pwar':round(total,3),'pwar_per_100':round(100*total/m,3) if m else None,
                          'matches':m,'replacement_effect':round(baseline,4),'match_examples':examples[:5]}
        result.append({'player':name,'rating':float(ratings[name]),'skill_band':f'{g:.1f}–{g+.49:.2f}',**values})
    return {'method':'Experimental APM-based rally-scoring match win equivalents', 'scoring':'rally', 'target':target,
            'players':result,'universal_replacement_effect':round(universal,4),
            'skill_baseline_bands':{str(k):round(v,4) for k,v in skill_baselines.items()},
            'caveats':['Replacement levels are provisional within-sample 20th-percentile APM effects, NOT validated replacement players.',
                       'Universal comparisons require a connected cross-skill matchup network; this prototype does not verify connectivity.',
                       'Match probabilities assume independent identical rally chances, ignore service/score effects, and are not calibrated.',
                       'Individual PWAR is a counterfactual model estimate, not causal attribution; no validated uncertainty interval.',
                       'Each unique match_id is counted once per player; results depend on uploaded match coverage.',
                       'Only rally scoring is supported. Side-out scoring and partnership-specific WAR are not yet implemented.']}
