import json, os, sys, numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SP=R
files=[os.path.join(R,'bench_ensemble.jsonl'),os.path.join(R,'bench_adaptive.jsonl'),
       os.path.join(R,"bench_sampling2.jsonl"),os.path.join(R,"bench_snapshot.jsonl")]
recs=[json.loads(l) for f in files if os.path.exists(f) for l in open(f) if l.strip()]
DS=['labour','prison','tourism_small','wiki2','traffic']
v=defaultdict(dict); steps=defaultdict(list)
for r in recs:
    v[(r['head'],r['dataset'])][r['seed']]=r['scrps']['Overall']; steps[r['head']].append(r['steps_run'])
    if 'extras' in r and 'single_scrps' in r['extras']:
        v[(r['head']+'(best-snap)',r['dataset'])][r['seed']]=r['extras']['single_scrps']['Overall']
ens={}
for f in ['bench_ensemble.json','bench_ensemble_dsf.json']:
    for d,s in json.load(open(os.path.join(R,f))).items():
        for h,e in s.get('ensemble',{}).items(): ens[(h,d)]=e['scrps']['Overall']
NAME={'factor_model':'CLOVER (base)','normalizing_flow':'CLOVER + DSF','copula_spline':'CLOVER + Spline'}
VAR={'':'','+as1':' + AS loss-prioritized (β=1)','+as0':' + AS loss-prioritized (β=0)','+sp':' + AS self-paced','+rec':' + AS recency','+snap':' + Snapshot ensemble (5 cycles)','+snap(best-snap)':' + cyclic LR, best single snapshot'}
H=list(NAME)
base_steps=np.mean(steps['factor_model'])
basem={d:np.mean(list(v[('factor_model',d)].values())) for d in DS}
def row(label,vals,cost):
    deltas=[100*(vals[d]-basem[d])/basem[d] for d in DS]
    cells=[(f"{vals[d]:.5f}" if d=='labour' else f"{vals[d]:.4f}") for d in DS]
    print(f"| {label} | "+' | '.join(cells)+f" | {np.mean(deltas):+.1f}% | {cost} |")
print('| Model | '+' | '.join(DS)+' | Avg Δ vs base | Train cost |'); print('|---|'+'---|'*(len(DS)+2))
for var in VAR:
    for h in H:
        k=h+var
        if not all(v[(k,d)] for d in DS): continue
        st=steps.get(h+var.replace('(best-snap)',''),[]); cost=f"{np.mean(st)/base_steps:.1f}×" if st else ''
        row(NAME[h]+VAR[var],{d:np.mean(list(v[(k,d)].values())) for d in DS},cost)
for h in H:
    if all((h,d) in ens for d in DS): row(NAME[h]+' + Seed ensemble (K=5)',{d:ens[(h,d)] for d in DS},'5.0×')
print('\nPaired vs same head uniform/default, same seed (25 pairs):')
for var in [k for k in VAR if k]:
    for h in H:
        k=h+var
        if not all(len(v[(k,d)])==5 for d in DS): continue
        diffs=np.array([(v[(k,d)][s]-v[(h,d)][s])/v[(h,d)][s]*100 for d in DS for s in range(5)])
        print(f"{(NAME[h]+VAR[var]):<58} mean {diffs.mean():+.2f}%  wins {int((diffs<0).sum())}/25  p={wilcoxon(diffs).pvalue:.3f}")
