"""Descriptive doubles partnership summaries; no causal player attribution."""
from collections import defaultdict


def summarize_partnerships(records, replacement_rate=0.40, points_per_win=11):
    groups = defaultdict(list)
    player_rows = defaultdict(list)
    for r in records:
        a = str(r.get('player', '')).strip()
        b = str(r.get('partner', '')).strip()
        if not a or not b or a == b or not isinstance(r.get('won'), bool):
            continue
        pair = tuple(sorted((a, b)))
        groups[pair].append(r)
        player_rows[a].append((b, r['won']))
    output = []
    for (a, b), rows in sorted(groups.items()):
        # One record per team per rally, when match_id exists; do not count both teammates twice.
        seen = set()
        unique = []
        for r in rows:
            mid = str(r.get('match_id') or '').strip()
            rally_id = str(r.get('rally_id') or '').strip()
            key = (mid, rally_id) if mid and rally_id else None
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            unique.append(r)
        n = len(unique)
        won = sum(r['won'] for r in unique)
        rate = won / n
        others = {}
        for p, teammate in ((a,b),(b,a)):
            sample = [w for partner,w in player_rows[p] if partner != teammate]
            if sample:
                others[p] = {'rallies':len(sample),'win_rate':round(sum(sample)/len(sample),4)}
        # Only descriptive; not an adjusted estimate of chemistry.
        comparison = None
        if len(others) == 2:
            baseline = sum(x['win_rate'] for x in others.values())/2
            comparison = round(rate-baseline,4)
        output.append({'players':[a,b], 'rallies':n, 'won':won,
                       'rally_win_rate':round(rate,4),
                       'rallies_above_replacement':round(won-replacement_rate*n,2),
                       'win_equivalents_above_replacement':round((won-replacement_rate*n)/points_per_win,3),
                       'other_partner_baselines':others,
                       'unadjusted_chemistry_gap':comparison})
    return {'partnerships':output, 'method':'Unadjusted partnership rally outcomes',
            'caveats':['Team results are not individual contributions.',
                       'Chemistry gap is a raw comparison to each player’s other-partner results; it does not control for opponent strength, selection, or game context.',
                       'Without match_id and rally_id, duplicated teammate rows cannot be automatically identified.',
                       'Win equivalents use a user-selected conversion assumption, not validated match wins.']}
