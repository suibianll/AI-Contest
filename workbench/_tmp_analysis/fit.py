# -*- coding: utf-8 -*-
import json, math, statistics as st

def load(fn, side):
    d=json.load(open('artifacts/official_eval/'+fn,encoding='utf-8'))
    r=d['results'][0]
    return r['case_scores'][side]

# ---- 本地逐 case 向量：version -> (file, side) ----
LIN = {
 'std'      : ('sidecal-v164-linear-default.json','linear'),
 'v86L'     : ('reeval5-v086-default.json','linear'),
 'v138L'    : ('reeval5-v138-default.json','linear'),
 'v139L'    : ('reeval5-v139-default.json','linear'),
 'v140L'    : ('reeval5-v147-default.json','linear'),   # v147 = v140L + v86A
 'v155L'    : ('reeval5-v155-default.json','linear'),
 'v156L'    : ('reeval5-v156-default.json','linear'),
 'v157L'    : ('reeval5-v157-default.json','linear'),
 'v160L'    : ('sidecal-v163-linear-default.json','linear'),
 'v166L'    : ('sidecal-v166-linear-default.json','linear'),
 'v174L'    : ('v174-compact-linear.json','linear'),    # 仅 compact，协议不同，标记
 'v182L'    : ('v182-linear-default.json','linear'),
}
ATT = {
 'std'      : ('sidecal-v163-attn-default.json','attention'),
 'v84A'     : ('reeval5-v084-default.json','attention'),
 'v86A'     : ('reeval5-v086-default.json','attention'),
 'v138A'    : ('reeval5-v138-default.json','attention'),
 'v158A'    : ('reeval5-v158-default.json','attention'),
 'v160A'    : ('sidecal-v164-attn-default.json','attention'),
 'v168A'    : ('v168-attn-default.json','attention'),
 'v171A'    : ('v171-attn-default.json','attention'),
 'v176A'    : ('v176-compact-attn.json','attention'),   # 仅 compact
 'v180A'    : ('v180-attn-default.json','attention'),
 'v183A'    : ('v183-attn-default.json','attention'),
}

def vecs(tbl):
    out={}
    for k,(fn,side) in tbl.items():
        try: out[k]=load(fn,side)
        except Exception as e: out[k]=None
    return out

LV=vecs(LIN); AV=vecs(ATT)

# ---- 官方侧向差分（相对 v162=1001 零点的侧贡献 Δ） ----
# 推导见 AGENTS.md / docs/current-solution-status.md
LIN_PAIRS = [  # (parent, child, Δofficial 侧贡献)
 ('v86L','v157L', -15),
 ('v86L','v140L', -165),
 ('v86L','v155L', -1163),
 ('v86L','v156L', -1164),
 ('v138L','v139L', +1),
 ('v138L','v140L', +123),
 ('v86L','v138L', -288),
 ('std','v160L', +3586),
 ('v160L','v166L', +3),
 ('v166L','v182L', +1),
 ('v160L','v174L', -79),
]
ATT_PAIRS = [
 ('v86A','v158A', +117),
 ('v86A','v84A', -1227),
 ('v86A','v138A', -1741),
 ('std','v160A', +12944),
 ('v160A','v168A', +60),
 ('v160A','v171A', -288),
 ('v160A','v176A', +19),
 ('v168A','v180A', +3),
 ('v180A','v183A', 0),
]

def g(c): return c['gain']
def logit(x):
    x=min(max(x,1e-9),1-1e-9); return math.log(x/(1-x))

def stats(v):
    gs=[g(c) for c in v]
    n=len(gs); s=sorted(gs)
    mid=s[n//2] if n%2 else 0.5*(s[n//2-1]+s[n//2])
    return dict(
        mean=sum(gs)/n, median=mid, p10=s[int(0.10*n)], worst=s[0],
        logit=logit(sum(gs)/n),
        tail=sum(s[:max(1,n//5)])/max(1,n//5),
        n=n)

def delta_stats(pa,ch,tbl,V):
    a,b=V.get(pa),V.get(ch)
    if a is None or b is None: return None
    if len(a)!=len(b): return None
    sa,sb=stats(a),stats(b)
    d={k:sb[k]-sa[k] for k in sa if k!='n'}
    # per-case 配对差分
    dg=[g(y)-g(x) for x,y in zip(a,b)]
    d['pos']=sum(1 for x in dg if x>1e-12); d['neg']=sum(1 for x in dg if x<-1e-12)
    d['paired_mean']=sum(dg)/len(dg)
    d['paired_median']=st.median(dg)
    d['paired_min']=min(dg); d['paired_max']=max(dg)
    d['L1']=sum(abs(x) for x in dg)/len(dg)
    d['linf']=max(abs(x) for x in dg)
    return d

def report(name,pairs,tbl,V):
    print('='*100); print('###',name)
    hdr=['pair','Δofficial','Δmean','Δmedian','Δp10','Δworst','Δlogit','Δtail','+/−','Δpaired_med','L1','linf']
    print(f"{hdr[0]:18s}{hdr[1]:>10s}{hdr[2]:>12s}{hdr[3]:>12s}{hdr[4]:>12s}{hdr[5]:>12s}{hdr[6]:>11s}{hdr[7]:>11s}{hdr[8]:>10s}{hdr[9]:>13s}{hdr[10]:>10s}{hdr[11]:>10s}")
    rec=[]
    for pa,ch,off in pairs:
        d=delta_stats(pa,ch,tbl,V)
        if d is None:
            print(f"{pa+'->'+ch:18s}{off:>10d}  (缺少同协议逐 case 数据)"); continue
        print(f"{pa+'->'+ch:18s}{off:>10d}{d['mean']:>12.6f}{d['median']:>12.6f}{d['p10']:>12.6f}{d['worst']:>12.6f}{d['logit']:>11.5f}{d['tail']:>11.6f}{str(d['pos'])+'/'+str(d['neg']):>10s}{d['paired_median']:>13.6f}{d['L1']:>10.5f}{d['linf']:>10.5f}")
        rec.append((pa,ch,off,d))
    return rec

rl=report('LINEAR 侧',LIN_PAIRS,LIN,LV)
ra=report('ATTENTION 侧',ATT_PAIRS,ATT,AV)
json.dump({'lin':[(p,c,o,d) for p,c,o,d in rl],'att':[(p,c,o,d) for p,c,o,d in ra]},open('workbench/_tmp_analysis/pairs.json','w'),indent=1)
