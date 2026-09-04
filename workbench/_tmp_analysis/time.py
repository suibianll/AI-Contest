import json,os,math
BASE='artifacts/official_eval/'
def api(fn):
    d=json.load(open(BASE+fn,encoding='utf-8'))
    r=d['results'][0]; t=r['timing']['api_seconds']
    return t
def tot(fn):
    return sum(api(fn).values())

# (版本, [(文件...)] , 官方时间)
CASES=[
 ('v162',['sidecal-v162-both-default.json'],146.0),
 ('v84',['reeval5-v084-default.json'],252.563),
 ('v86',['reeval5-v086-default.json'],222.7),
 ('v138',['reeval5-v138-default.json'],208.0),
 ('v139',['reeval5-v139-default.json'],202.0),
 ('v140',['reeval5-v140-default.json'],207.0),
 ('v147',['reeval5-v147-default.json'],211.0),
 ('v155',['reeval5-v155-default.json'],208.5),
 ('v156',['reeval5-v156-default.json'],204.3),
 ('v157',['reeval5-v157-default.json'],218.96),
 ('v158',['reeval5-v158-default.json'],223.0),
 ('v160',['v160-integration-default.json'],232.0),
 ('v163',['sidecal-v163-linear-default.json','sidecal-v163-attn-default.json'],202.0),
 ('v164',['sidecal-v164-linear-default.json','sidecal-v164-attn-default.json'],204.0),
 ('v166',['sidecal-v166-linear-default.json','sidecal-v166-attn-default.json'],226.0),
 ('v168',['sidecal-v164-linear-default.json','v168-attn-default.json'],210.0),
 ('v171',['sidecal-v164-linear-default.json','v171-attn-default.json'],214.0),
 ('v175',['sidecal-v166-linear-default.json','v168-attn-default.json'],245.0),
 ('v180',['v180-linear-default.json','v180-attn-default.json'],242.0),
 ('v182',['v182-linear-default.json','diag-v182-attn-default.json'],273.0),
 ('v183',['v182-linear-default.json','v183-attn-default.json'],279.7),
]
rows=[]
for name,fs,ot in CASES:
    t={}
    for f in fs:
        for k,v in api(f).items(): t[k]=t.get(k,0.0)+v
    wq=t.get('hif4_calibration_and_quantize_weight',0.0); act=t.get('hif4_dynamic_quantize_activation',0.0)
    ac=t.get('hif4_calibration_attention',0.0)
    dyn=t.get('hif4_dynamic_quantize_q',0.0)+t.get('hif4_dynamic_quantize_k',0.0)+t.get('hif4_dynamic_quantize_v',0.0)
    rows.append(dict(v=name,wq=wq,act=act,ac=ac,dyn=dyn,api=wq+act+ac+dyn,off=ot))
rows.sort(key=lambda r:r['api'])
print(f"{'ver':7s}{'Wcalib':>10s}{'Acalib':>10s}{'dynAct':>9s}{'dynQKV':>9s}{'localAPI':>11s}{'官方':>9s}{'残差(拟合1)':>13s}")
def ols(X,y):
    n=len(y); p=len(X[0])
    A=[[sum(X[i][a]*X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    bv=[sum(X[i][a]*y[i] for i in range(n)) for a in range(p)]
    # gauss
    M=[A[i][:]+[bv[i]] for i in range(p)]
    for c in range(p):
        piv=max(range(c,p),key=lambda r:abs(M[r][c])); M[c],M[piv]=M[piv],M[c]
        for r in range(p):
            if r!=c:
                f=M[r][c]/M[c][c]
                for k in range(c,p+1): M[r][k]-=f*M[c][k]
    return [M[i][p]/M[i][i] for i in range(p)]
def pred(b,X): return [sum(bi*xi for bi,xi in zip(b,x)) for x in X]

X1=[[1.0,r['api']] for r in rows]; y=[r['off'] for r in rows]
b1=ols(X1,y); p1=pred(b1,X1)
X2=[[1.0,r['wq'],r['ac'],r['dyn']] for r in rows]
b2=ols(X2,y); p2=pred(b2,X2)
X3=[[1.0,r['wq'],r['ac'],r['dyn'],r['act']] for r in rows]
b3=ols(X3,y); p3=pred(b3,X3)
def r2(p,y):
    my=sum(y)/len(y); return 1-sum((a-b)**2 for a,b in zip(p,y))/sum((v-my)**2 for v in y)
def mae(p,y): return sum(abs(a-b) for a,b in zip(p,y))/len(y)
for r,a,b,c in zip(rows,p1,p2,p3):
    print(f"{r['v']:7s}{r['wq']:>10.1f}{r['ac']:>10.1f}{r['act']:>9.1f}{r['dyn']:>9.2f}{r['api']:>11.1f}{r['off']:>9.1f}{a-r['off']:>13.1f}")
print()
print('模型1  off = %.3f + %.4f * localAPI   R2=%.3f  MAE=%.1fs'%(b1[0],b1[1],r2(p1,y),mae(p1,y)))
print('模型2  off = %.3f + %.4f*Wcalib + %.4f*Acalib + %.4f*dynQKV   R2=%.3f  MAE=%.1fs'%(b2[0],b2[1],b2[2],b2[3],r2(p2,y),mae(p2,y)))
print('模型3  + dynAct(%.4f)  R2=%.3f  MAE=%.1fs'%(b3[4],r2(p3,y),mae(p3,y)))
# 300s 门禁
print()
print('本地 API 与官方 300s 门禁的对应关系：')
for r,a in zip(rows,p3):
    flag='TIMEOUT-RISK' if a>285 else ''
    print(f"  {r['v']:6s} local={r['api']:7.1f}s  预测官方={a:6.1f}s  实测={r['off']:6.1f}s  {flag}")
