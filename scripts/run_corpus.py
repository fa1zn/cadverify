import sys, glob, json, time; sys.path.insert(0,"/Users/faizansyed/normal/cadverify")
import numpy as np
from cadverify.invariants import load_step, invariants
from cadverify.align import align
ROOT="/Users/faizansyed/Downloads/collection-selection"
old={x["id"]:x for x in json.load(open("corpus_run_icp.json"))}
out=[]; t0=time.time()
for p in sorted(glob.glob(ROOT+"/samples/*/*")):
    sj=json.load(open(p+"/sample.json"))
    ref=load_step(p+"/ground_truth/reference.step"); sub=load_step(p+"/submission/final.step")
    ri,si=invariants(ref),invariants(sub); r=align(ref,sub)
    rel=lambda a,b: abs(a-b)/abs(b)*100 if b else None
    out.append(dict(id=sj["id"],title=sj["title"],vol_err=rel(si.volume,ri.volume),
        area_err=rel(si.area,ri.area),
        moment_err=max(rel(a,b) for a,b in zip(si.moments,ri.moments) if b),
        faces_ref=ri.n_faces,faces_sub=si.n_faces,chamfer_pct=r["chamfer_pct_diag"],
        moment_gap=r["moment_gap"],degenerate=r["degenerate"],n_candidates=r["n_candidates"]))
el=time.time()-t0
json.dump(out, open("corpus_run_fast.json","w"), indent=1)
d=np.array([abs(x["chamfer_pct"]-old[x["id"]]["chamfer_pct"]) for x in out])
print(f"150 pairs in {el:.0f}s ({el/150:.2f} s/pair) — was 5664s (37.6 s/pair) — {5664/el:.0f}x")
print(f"max drift vs previous run: {d.max():.2e}")
print(f"pairs changed by >1e-9: {int((d>1e-9).sum())}/150")
ch=np.array([x["chamfer_pct"] for x in out]); vol=np.array([x["vol_err"] for x in out])
print(f"\nheadline unchanged: median {np.median(ch):.4f}%  p95 {np.percentile(ch,95):.4f}%  max {ch.max():.4f}%")
print(f"false accepts (vol<=3%, shape>0.5%): {int(((vol<=3)&(ch>0.5)).sum())}")
