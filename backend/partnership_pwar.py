"""Experimental partnership counterfactual PWAR; rally-scoring only."""
from collections import defaultdict
import math
import numpy as np
from .adjusted_plus_minus import fit_apm
from .pwar import match_probability


def partnership_pwar(records, ratings, target=11, baseline_quantile=.20):
    if target not in (11,15,21):
        raise ValueError('Supported rally-scoring targets: 11, 15, 21')
    fit = fit_apm(records)
    names = [x['player'] for x in fit['players']]
    missing = [n for n in names if n not in ratings]
    if missing: raise ValueError('Ratings required for: '+', '.join(missing))
    if any(not 2 <= float(ratings[n]) <= 8 for n in names):
        raise ValueError('Ratings must be between 2.0 and 8.0')
    def logit_from_pp(pp):
        p = min(.999999,max(.000001,.5+float(pp)/100))
        return math.log(p/(1-p))
    effects = {x['player']:logit_from_pp(x['adjusted_impact_pp']) for x in fit['players']}
    pair_effects = {tuple(x['players']):logit_from_pp(x['adjusted_synergy_pp']) for x in fit['partnerships']}
    groups = defaultdict(list)
    for n in names: groups[math.floor(float(ratings[n])*2)/2].append(n)
    universal = float(np.quantile(list(effects.values()),baseline_quantile))
    band_base = {g:float(np.quantile([effects[n] for n in ns],baseline_quantile)) for g,ns in groups.items() if len(ns)>=3}
    # A partnership receives one opportunity for each distinct match in which it appears.
    # Orient each observation so the focal partnership is Team A; deduplicate per match.
    contexts = defaultdict(dict)
    for r in records:
        a,b,c,d = (r[k] for k in ('player','partner','opponent','opponent2'))
        mid = str(r['match_id'])
        contexts[tuple(sorted((a,b)))].setdefault(mid,(c,d))
        contexts[tuple(sorted((c,d)))].setdefault(mid,(a,b))
    sigmoid=lambda z: 1/(1+math.exp(-max(-30,min(30,z))))
    output=[]
    for pair,matchups in sorted(contexts.items()):
        a,b=pair
        bands=[math.floor(float(ratings[n])*2)/2 for n in pair]
        # Only compare within a division if both players belong to the same band.
        skill_base = band_base.get(bands[0]) if bands[0]==bands[1] else None
        vals={}
        for mode,baseline in [('skill_adjusted',skill_base),('universal',universal)]:
            if baseline is None:
                vals[mode]={'available':False,'reason':('Mixed skill bands' if bands[0]!=bands[1] and mode=='skill_adjusted' else 'At least 3 rated players required in the skill band')}
                continue
            total=0.; chemistry=0.; examples=[]
            for mid,(o1,o2) in matchups.items():
                opponents=tuple(sorted((o1,o2)))
                offset=-effects[o1]-effects[o2]-pair_effects.get(opponents,0.)
                synergy=pair_effects.get(pair,0.)
                # Actual partnership, two hypothetical replacement players (no established pair synergy),
                # and the same actual individuals with their synergy removed.
                actual=match_probability(sigmoid(effects[a]+effects[b]+synergy+offset),target)
                replacement=match_probability(sigmoid(2*baseline+offset),target)
                no_synergy=match_probability(sigmoid(effects[a]+effects[b]+offset),target)
                total+=actual-replacement
                chemistry+=actual-no_synergy
                examples.append({'match_id':mid,'actual_win_probability':round(actual,4),'replacement_pair_win_probability':round(replacement,4)})
            count=len(matchups)
            vals[mode]={'available':True,'partnership_pwar':round(total,3),
                'partnership_pwar_per_100':round(100*total/count,3) if count else None,
                'chemistry_wins_added':round(chemistry,3),
                'chemistry_per_100':round(100*chemistry/count,3) if count else None,
                'matches':count,'replacement_effect_per_player':round(baseline,4),
                'match_examples':examples[:5]}
        output.append({'players':list(pair),'ratings':[float(ratings[a]),float(ratings[b])],
                       'rallies':next((x['rallies'] for x in fit['partnerships'] if tuple(x['players'])==pair),0),
                       'synergy_effect_pp':next((x['adjusted_synergy_pp'] for x in fit['partnerships'] if tuple(x['players'])==pair),0),
                       **vals})
    return {'method':'Experimental partnership counterfactual PWAR from regularized APM',
            'target':target,'scoring':'rally','partnerships':output,
            'caveats':['Partnership PWAR compares the observed pair with TWO replacement-level players; it is not the sum of individual PWAR.',
                       'Chemistry Wins Added isolates the fitted pair interaction holding individual effects fixed; it is not causal.',
                       'Replacement pairs are assumed to have zero synergy, an unvalidated modeling choice.',
                       'Skill-adjusted pair PWAR requires both players in the same 0.5 rating band with at least 3 rated players in that band.',
                       'Replacement levels are provisional 20th-percentile estimates from this dataset, not externally validated.',
                       'APM effects are reconstructed from rounded reported coefficients; estimates have rounding error.',
                       'Match probabilities assume independent identical rallies, ignore service and score effects, and have not been calibrated.',
                       'Cross-division comparisons require a connected matchup network; uncertainty intervals are not yet available.',
                       'Match opportunities are counted once per unique match ID for each pair; no side-out scoring support.']}
