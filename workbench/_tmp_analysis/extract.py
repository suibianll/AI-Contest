import json,glob,os
rows=[]
for p in sorted(glob.glob('artifacts/official_eval/**/*.json',recursive=True)):
    try: d=json.load(open(p,encoding='utf-8'))
    except Exception: continue
    if not isinstance(d,dict): continue
    res=d.get('results')
    if not isinstance(res,list) or not res: continue
    for r in res:
        if not isinstance(r,dict): continue
        sc=r.get('score') or {}; tm=r.get('timing') or {}; es=r.get('evaluation_scope') or {}
        rows.append(dict(
            file=os.path.basename(p), cand=r.get('candidate'), kind=es.get('kind'),
            comparable=es.get('comparable_for_proxy_ranking'),
            L=sc.get('linear_mean'), A=sc.get('attention_mean'), overall=sc.get('overall_mean'),
            lcases=sc.get('linear_cases'), acases=sc.get('attention_cases'),
            api=tm.get('api_seconds'), wall=tm.get('wall_seconds'),
            off_s=(r.get('official') or {}).get('score'), off_t=(r.get('official') or {}).get('time_seconds'),
            created=d.get('created_at')))
json.dump(rows,open('workbench/_tmp_analysis/rows.json','w'),indent=1)
print('total',len(rows))
d=[r for r in rows if r['comparable'] and r['L'] is not None and r['A'] is not None]
print('default both-panel:',len(d))
for r in d:
    f=lambda x: ('%.6f'%x) if isinstance(x,float) else str(x)
    print(f"{r['file'][:44]:44s} L={f(r['L'])} A={f(r['A'])} ov={f(r['overall'])} api={f(r['api'])} off={r['off_s']}/{r['off_t']}")
