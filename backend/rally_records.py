"""Convert detected candidate intervals + human verified labels to model-ready records.
No claim of automatically detecting winners, serving side or player identity.
"""
def build_records(analysis, match_id, players, labels):
    names=[str(players.get(k,'')).strip() for k in ('player','partner','opponent','opponent2')]
    if len(set(names)) != 4 or not all(names):
        raise ValueError('Provide four distinct player names')
    candidates={str(r['id']):r for r in analysis.get('rallies',[])}
    records=[]; rejected=0; incomplete=0
    seen=set()
    for label in labels:
        rid=str(label.get('rally_id',''))
        if rid in seen: raise ValueError('Duplicate rally label '+rid)
        seen.add(rid)
        if rid not in candidates: raise ValueError('Unknown candidate rally '+rid)
        if label.get('status')=='rejected': rejected+=1; continue
        if label.get('status')!='confirmed' or label.get('winner') not in ('A','B') or label.get('serve_team') not in ('A','B'):
            incomplete+=1;continue
        r=candidates[rid]
        records.append({'match_id':match_id,'rally_id':rid,'player':names[0],'partner':names[1],
                        'opponent':names[2],'opponent2':names[3],
                        'won':label['winner']=='A','serve_team':label['serve_team'],
                        'start':r['start'],'end':r['end'], 'source':'video_interval_human_verified'})
    return {'records':records,'included':len(records),'rejected':rejected,'incomplete':incomplete,
            'caveat':'Rally times are motion-detected candidates. Winners, service and player identities require human verification. Video does not automatically determine outcomes.'}
