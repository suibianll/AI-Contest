import json,math,itertools,random
rows=json.load(open('workbench/_tmp_analysis/bake.json',encoding='utf-8'))
def sign(x,e=1e-9): return 0 if abs(x)<e else (1 if x>0 else -1)

# ---------- 1. 复合决策规则 bake-off ----------
def rule_eval(f,rows,label):
    tp=fp=tn=fn=skip=0
    for r in rows:
        try: d=f(r)
        except KeyError: skip+=1; continue
        if d is None: skip+=1; continue
        o=sign(r['off'])
        if d and o>0: tp+=1
        elif d and o<0: fp+=1
        elif (not d) and o<0: tn+=1
        elif (not d) and o>0: fn+=1
        else: skip+=1
    n=tp+fp+tn+fn
    prec=tp/(tp+fp) if tp+fp else float('nan')
    rec=tp/(tp+fn) if tp+fn else float('nan')
    print(f"{label:52s} n={n:2d}  提交且官方正={tp}  提交但官方负={fp}  拒了官方负={tn}  漏掉官方正={fn}  精确率={prec:.2f}")

R=[
 ("R1  Δmean>0",                       lambda r: r['q_def_d_mean']>0),
 ("R2  Δmean>0 且 L1<0.02",             lambda r: r['q_def_d_mean']>0 and r['q_def_L1']<0.02),
 ("R3  Δmean>0 且 Δtail>=0",            lambda r: r['q_def_d_mean']>0 and r['q_def_d_tail']>=0),
 ("R4  Δmean>0 且 L1<0.02 且 Δtail>=0", lambda r: r['q_def_d_mean']>0 and r['q_def_L1']<0.02 and r['q_def_d_tail']>=0),
 ("R5  Δtail>0",                       lambda r: r['q_def_d_tail']>0),
 ("R6  Δmean>0 且 Δworst>-0.01",        lambda r: r['q_def_d_mean']>0 and r['q_def_d_worst']>-0.01),
 ("R7  compact Δmean>0",               lambda r: r.get('q_cmp_d_mean',0)>0 if 'q_cmp_d_mean' in r else None),
 ("R8  Δmean>0 且 (无compact 或 compact>=0)", lambda r: r['q_def_d_mean']>0 and (r.get('q_cmp_d_mean',0)>=0 if 'q_cmp_d_mean' in r else True)),
 ("R9  Δmean>0 且 paired_med>=0",       lambda r: r['q_def_d_mean']>0 and r['q_def_d_paired_med']>=0),
 ("R10 Δmean>0 且 L1<0.02 且 linf<0.1", lambda r: r['q_def_d_mean']>0 and r['q_def_L1']<0.02 and r['q_def_linf']<0.1),
]
print('== 提交决策规则 bake-off（17 个官方已回传的侧向配对）==')
for lab,f in R: rule_eval(f,rows,lab)

# ---------- 2. 定量模型 + LOO ----------
def ols(X,y,lam=0.0):
    n=len(y); p=len(X[0])
    A=[[sum(X[i][a]*X[i][b] for i in range(n))+(lam if a==b and a>0 else 0) for b in range(p)] for a in range(p)]
    b=[sum(X[i][a]*y[i] for i in range(n)) for a in range(p)]
    M=[A[i][:]+[b[i]] for i in range(p)]
    for c in range(p):
        piv=max(range(c,p),key=lambda r:abs(M[r][c])); M[c],M[piv]=M[piv],M[c]
        if abs(M[c][c])<1e-12: return None
        for r2 in range(p):
            if r2!=c:
                fq=M[r2][c]/M[c][c]
                for k in range(c,p+1): M[r2][k]-=fq*M[c][k]
    return [M[i][p]/M[i][i] for i in range(p)]

def loo(rows,feats,lam=0.0):
    errs=[];sgn=0;tot=0
    for i in range(len(rows)):
        tr=[r for j,r in enumerate(rows) if j!=i]
        X=[[1.0]+[r[f] for f in feats] for r in tr]; y=[r['off'] for r in tr]
        b=ols(X,y,lam)
        if b is None: continue
        xi=[1.0]+[rows[i][f] for f in feats]
        p=sum(bi*xx for bi,xx in zip(b,xi))
        errs.append(abs(p-rows[i]['off']))
        if sign(rows[i]['off'])!=0 and sign(p)==sign(rows[i]['off']): sgn+=1
        tot+=1
    return sum(errs)/len(errs) if errs else float('nan'), (sgn,tot)

F=[r for r in rows if all(k in r for k in ('q_def_d_mean','q_def_L1','q_def_d_tail'))]
print()
print('== 定量模型（LOO 交叉验证，官方侧向差分 = 因变量）==')
cands=[
 ['q_def_d_mean'],
 ['q_def_d_mean','q_def_L1'],
 ['q_def_d_mean','q_def_d_tail'],
 ['q_def_d_mean','q_def_L1','q_def_d_tail'],
 ['q_def_d_mean','q_def_L1','q_def_d_tail','q_def_d_worst'],
 ['q_def_d_tail'],
]
for feats in cands:
    m,s=loo(F,feats,lam=1e-6)
    print(f"  特征={str([f.replace('q_def_','') for f in feats]):58s} LOO MAE={m:8.1f} 分   符号准确率 {s[0]}/{s[1]}")

# ---------- 3. 换挡：分段（微扰 vs 大改）----------
print()
print('== 分段：本地改动幅度决定可信度 ==')
pert=[r for r in F if r['q_def_L1']<0.02]
big =[r for r in F if r['q_def_L1']>=0.02]
for nm,S in (('微扰区间 L1<0.02',pert),('大改区间 L1>=0.02',big)):
    if len(S)<2: continue
    ok=sum(1 for r in S if sign(r['q_def_d_mean'])==sign(r['off']) and sign(r['off'])!=0)
    tot=sum(1 for r in S if sign(r['off'])!=0 and sign(r['q_def_d_mean'])!=0)
    print(f"  {nm}: n={len(S)}  Δmean 符号一致 {ok}/{tot}")
    for r in S:
        print(f"      {r['side'][:4]:4s} {r['pair']:22s} Δmean={r['q_def_d_mean']:+.6f} L1={r['q_def_L1']:.5f}  官方={r['off']:+6d}")
