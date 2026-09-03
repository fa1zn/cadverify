"""Goodhart curve, model-free variant.

Candidate pool is built from geometric perturbations, not model samples.
Measures how Reward A (shipped) behaves as a SELECTOR when the pool spans a
range of true quality. Does not model any particular policy's error distribution.
"""
import sys, glob, json, time, math; sys.path.insert(0,"/Users/faizansyed/normal/cadverify")
import numpy as np
from cadverify.invariants import load_step, invariants, transformed
from cadverify.align import align
from cadverify.exploit import POLICIES
from OCP.gp import gp_Trsf, gp_GTrsf, gp_Ax1, gp_Pnt, gp_Dir, gp_Vec, gp_Mat
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

ROOT="/Users/faizansyed/Downloads/collection-selection"
def dims(s):
    b=Bnd_Box(); BRepBndLib.Add_s(s,b,True); x0,y0,z0,x1,y1,z1=b.Get()
    return (x1-x0,y1-y0,z1-z0)
def pct(a,b): return abs(a-b)/abs(b)*100 if b else float('inf')

def uniform_scale(shape, f):
    t=gp_Trsf(); t.SetScale(gp_Pnt(0,0,0), f); return transformed(shape,t)
def aniso(shape, s):
    g=gp_GTrsf(); g.SetVectorialPart(gp_Mat(s,0,0, 0,1/s,0, 0,0,1))
    return BRepBuilderAPI_GTransform(shape, g, True).Shape()
def rotate(shape, rng):
    ax=rng.normal(size=3); ax/=np.linalg.norm(ax)
    t=gp_Trsf(); t.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(*ax)), rng.uniform(0,2*math.pi))
    return transformed(shape,t)

paths=sorted(glob.glob(ROOT+"/samples/*/*"))
rng=np.random.default_rng(17)
sel=[paths[i] for i in rng.choice(len(paths),40,replace=False)]
others=[paths[i] for i in rng.choice(len(paths),40,replace=False)]

rows=[]; t0=time.time()
for ti,(p,op) in enumerate(zip(sel,others)):
    sj=json.load(open(p+"/sample.json")); ak=sj["answerKey"]; dens=sj["densityGPerMm3"]
    ref=load_step(p+"/ground_truth/reference.step")
    cands={}
    cands["reference"]=ref
    cands["rotated"]=rotate(ref,rng)
    cands["submission"]=load_step(p+"/submission/final.step")
    for f in (1.01,1.025,1.05): cands[f"scale_{f}"]=uniform_scale(ref,f)
    cands["aniso_1.15"]=aniso(ref,1.15)
    cands["pocketed"]=POLICIES["pocketed_block"](ak)
    cands["cube"]=POLICIES["volume_cube"](ak)
    try:
        o=load_step(op+"/ground_truth/reference.step")
        ov=invariants(o).volume
        cands["other_part"]=uniform_scale(o,(ak["targetVolumeMm3"]/ov)**(1/3))
    except Exception: pass

    for name,c in cands.items():
        try:
            inv=invariants(c); bb=dims(c)
            ve=pct(inv.volume, ak["targetVolumeMm3"])
            be=max(pct(bb[i],[ak["targetBoundingBoxMm"][k] for k in "xyz"][i]) for i in range(3))
            me=pct(inv.volume*dens, ak["targetMassGrams"]) if ak["targetMassGrams"] else 0.0
            rewardA = -max(ve,be,me)                       # continuous: higher is better
            rewardB = -align(ref,c)["chamfer_pct_diag"]    # ground truth
            rows.append(dict(task=sj["id"],cand=name,A=rewardA,B=rewardB,
                             vol=ve,bbox=be,mass=me,passA=bool(ve<=3 and be<=3 and me<=3)))
        except Exception as e:
            rows.append(dict(task=sj["id"],cand=name,A=-1e9,B=-1e9,error=str(type(e).__name__)))
    if (ti+1)%10==0: print(f"  {ti+1}/40 tasks ({time.time()-t0:.0f}s)", flush=True)

json.dump(rows, open("goodhart_pool.json","w"), indent=1)
print(f"pool built: {len(rows)} candidates over 40 tasks in {time.time()-t0:.0f}s")
