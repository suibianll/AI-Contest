# -*- coding: utf-8 -*-
import json, math, statistics as st, itertools, os

BASE='artifacts/official_eval/'
def qload(fn):
    d=json.load(open(BASE+fn,encoding='utf-8'))
    if d.get('results'):
        r=d['results'][0]; return r['case_scores'], d.get('data_metadata',{})
    return d.get('case_scores'), d.get('data_metadata',{})

def gains(fn, side):
    cs,_=qload(fn)
    if not cs: return None
    return [c['gain'] for c in cs[side]]

def agg(fn, side):
    d=json.load(open(BASE+fn,encoding='utf-8'))
    sc=(d.get('results') and d['results'][0]['score']) or d.get('score')
    return sc[side+'_mean']

# ---------- 面板取值表 ----------
# (qwen_default, qwen_compact, gpt2, opt)
ATT={
 'v84A'  : ('reeval5-v084-default.json', None, None, None),
 'v86A'  : ('reeval5-v086-default.json', None, None, None),
 'v138A' : ('reeval5-v138-default.json','s0-v138-attn-compact.json', None, None),
 'v158A' : ('reeval5-v158-default.json','v159-attn-compact-parent.json','v159-attn-gpt2-parent.json',None),
 'std'   : ('sidecal-v163-attn-default.json', None, None, None),
 'v160A' : ('sidecal-v164-attn-default.json','v164-compact-attn-parent-ref.json','v160-final-gpt2-integration.json','v160-opt-parent.json'),
 'v168A' : ('v168-attn-default.json','v168-compact-attn.json','v168-gpt2-attn-compact.json',None),
 'v171A' : ('v171-attn-default.json','v171-compact-attn.json','v171-gpt2-attn-compact.json','v171-opt-integration.json'),
 'v176A' : (None,'v176-compact-attn.json','v176-gpt2-attn-compact.json','v176-opt-integration.json'),
 'v180A' : ('v180-attn-default.json','v180-compact-attn.json','v180-gpt2-attn-default.json','v180-opt-attn-default.json'),
 'v183A' : ('v183-attn-default.json','v183-compact-attn.json','v183-gpt2-attn-default.json','v183-opt-attn-default.json'),
}
LIN={
 'std'   : ('sidecal-v164-linear-default.json', None, None, None),
 'v86L'  : ('reeval5-v086-default.json', None, None, None),
 'v138L' : ('reeval5-v138-default.json', None, None, None),
 'v139L' : ('reeval5-v139-default.json', None, None, None),
 'v140L' : ('reeval5-v147-default.json', None, None, None),
 'v155L' : ('reeval5-v155-default.json', None, None, None),
 'v156L' : ('reeval5-v156-default.json', None, None, None),
 'v157L' : ('reeval5-v157-default.json', None, None, None),
 'v160L' : ('sidecal-v163-linear-default.json','v159-l1-batch-compact-candidate.json','v159-lin-gpt2-parent.json','v160-opt-parent.json'),
 'v166L' : ('sidecal-v166-linear-default.json','v166-compact-linear-smoke.json','v175-gpt2-linear.json','v175-opt-integration.json'),
 'v174L' : (None,'v174-compact-linear.json','v174-gpt2-linear.json','v174-opt-integration.json'),
 'v182L' : ('v182-linear-default.json','v182-compact-linear.json','v182-gpt2-linear.json','v182-opt-linear.json'),
}

def get(tbl,k,idx,side):
    e=tbl.get(k)
    if not e or idx>=len(e) or not e[idx]: return None
    try:
        if idx in (0,1): return gains(e[idx],side)
        return agg(e[idx],side)
    except Exception: return None

def mean(x): return sum(x)/len(x)
def logit(x):
    x=min(max(x,1e-9),1-1e-9); return math.log(x/(1-x))
def pct(x,p):
    s=sorted(x); i=min(len(s)-1,max(0,int(p*len(s)))); return s[i]

def dstats(a,b):
    n=len(a); dg=[y-x for x,y in zip(a,b)]
    return dict(
        d_mean=mean(b)-mean(a),
        d_median=st.median(b)-st.median(a),
        d_paired_med=st.median(dg),
        d_paired_mean=mean(dg),
        d_p10=pct(b,0.10)-pct(a,0.10),
        d_worst=min(b)-min(a),
        d_tail=(sum(sorted(b)[:max(1,n//5)])/max(1,n//5))-(sum(sorted(a)[:max(1,n//5)])/max(1,n//5)),
        d_logit=logit(mean(b))-logit(mean(a)),
        d_p90=pct(b,0.90)-pct(a,0.90),
        L1=mean([abs(x) for x in dg]),
        linf=max(abs(x) for x in dg),
        pos=sum(1 for x in dg if x>1e-12), neg=sum(1 for x in dg if x<-1e-12),
    )

LIN_PAIRS=[('v86L','v157L',-15),('v86L','v140L',-165),('v86L','v155L',-1163),('v86L','v156L',-1164),
           ('v138L','v139L',1),('v138L','v140L',123),('v86L','v138L',-288),
           ('std','v160L',3586),('v160L','v166L',3),('v166L','v182L',1),('v160L','v174L',-79)]
ATT_PAIRS=[('v86A','v158A',117),('v86A','v84A',-1227),('v86A','v138A',-1741),
           ('std','v160A',12944),('v160A','v168A',60),('v160A','v171A',-288),('v160A','v176A',19),
           ('v168A','v180A',3),('v180A','v183A',0)]

def build(side,tbl,pairs):
    rows=[]
    for pa,ch,off in pairs:
        rec={'pair':pa+'→'+ch,'off':off,'side':side}
        ok=False
        for idx,name in ((0,'q_def'),(1,'q_cmp')):
            a=get(tbl,pa,idx,side); b=get(tbl,ch,idx,side)
            if a and b and len(a)==len(b):
                d=dstats(a,b); rec.update({name+'_'+k:v for k,v in d.items()}); ok=True
        for idx,name in ((2,'gpt2'),(3,'opt')):
            a=get(tbl,pa,idx,side); b=get(tbl,ch,idx,side)
            if a is not None and b is not None:
                rec[name]=b-a
        if ok or 'gpt2' in rec or 'opt' in rec: rows.append(rec)
    return rows

RL=build('linear',LIN,LIN_PAIRS); RA=build('attention',ATT,ATT_PAIRS)
json.dump(RL+RA,open('workbench/_tmp_analysis/bake.json','w'),indent=1)

def sign(x,eps=1e-9): return 0 if abs(x)<eps else (1 if x>0 else -1)
def conc(rows,key,pert_only=False,thr=0.02):
    n=c=0
    for r in rows:
        if key not in r: continue
        if pert_only and abs(r.get('q_def_d_mean',0))>thr: continue
        so=sign(r['off']); sp=sign(r[key])
        if so==0 or sp==0: continue
        n+=1; c+= (so==sp)
    return c,n

def spearman(xs,ys):
    def rank(v):
        s=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        for k,i in enumerate(s): r[i]=k
        return r
    rx,ry=rank(xs),rank(ys); n=len(xs)
    mx,my=sum(rx)/n,sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den=math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else float('nan')

print('#'*110)
print('== 候选本地估计量 vs 官方侧向差分：符号一致率 / Spearman ==')
keys=[k for k in RL[0] if k not in ('pair','off','side')] if RL else []
allk=sorted(set().union(*[set(r) for r in RL+RA])-{'pair','off','side'})
print(f"{'统计量':22s}{'全样本 一致':>14s}{'微扰样本 一致':>16s}{'Spearman(全体)':>18s}{'Spearman(微扰)':>18s}")
for k in allk:
    rows=[r for r in RL+RA if k in r]
    if len(rows)<4: continue
    c1,n1=conc(rows,k); c2,n2=conc(rows,k,pert_only=True)
    xs=[r[k] for r in rows]; ys=[r['off'] for r in rows]
    sp1=spearman(xs,ys)
    pr=[r for r in rows if abs(r.get('q_def_d_mean',0))<=0.02]
    sp2=spearman([r[k] for r in pr],[r['off'] for r in pr]) if len(pr)>=4 else float('nan')
    print(f"{k:22s}{str(c1)+'/'+str(n1):>14s}{str(c2)+'/'+str(n2):>16s}{sp1:>18.3f}{sp2:>18.3f}")
