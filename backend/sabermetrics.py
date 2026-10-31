"""Transparent descriptive pickleball metrics. Not a trained expectancy model."""
from collections import defaultdict
from math import sqrt


def summarize(matches, replacement_rate=0.40, points_per_win=11):
    """Compute *unadjusted* wins above replacement from player-attributed rallies.

    Each record: player, partner, opponent, won (bool), optional match_id.
    WAR is an *equivalent*, not causal player value, and must not be treated as
    opponent- or partner-adjusted. One rally is attributed to both teammates.
    """
    if not 0 <= replacement_rate <= 1:
        raise ValueError('replacement_rate must be between 0 and 1')
    if points_per_win <= 0:
        raise ValueError('points_per_win must be positive')
    by_player = defaultdict(list)
    for r in matches:
        if not isinstance(r.get('won'), bool) or not r.get('player'):
            raise ValueError('Each rally requires player and boolean won')
        by_player[r['player']].append(r)
    result = []
    for player, rows in sorted(by_player.items()):
        n = len(rows)
        won = sum(r['won'] for r in rows)
        rate = won / n
        # Descriptive win equivalents: baseline difference in rally points / assumed points per win.
        excess = won - replacement_rate * n
        equivalent = excess / points_per_win
        # Wilson 95% CI for raw rally win rate, not for WAR.
        z = 1.96
        denom = 1 + z*z/n
        center = (rate + z*z/(2*n))/denom
        half = z*sqrt(rate*(1-rate)/n + z*z/(4*n*n))/denom
        result.append({'player':player, 'rallies':n, 'won':won,
                       'rally_win_rate':round(rate, 4),
                       'rate_ci95':[round(max(0,center-half),4),round(min(1,center+half),4)],
                       'rallies_above_replacement':round(excess,2),
                       'war_equivalent':round(equivalent,3),
                       'partners':sorted(set(str(r.get('partner') or 'Unknown') for r in rows))})
    return {'method':'Unadjusted Rally Wins Above Replacement Equivalent (rWAR-eq)',
            'replacement_rally_win_rate':replacement_rate,
            'rally_points_per_win_equivalent':points_per_win,
            'players':result,
            'caveats':['Not a validated WAR model or an estimate of individual causal contribution.',
                       'Does not adjust for opponents, partner skill, serve advantage, scoring rules or game state.',
                       'Rallies are attributed to each participating player; do not sum teammate values.',
                       'Points-per-win conversion is a user assumption, not a measured conversion.',
                       'Confidence interval assumes independent rallies, often unrealistic.']}
