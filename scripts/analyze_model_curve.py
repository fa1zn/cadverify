"""Curve from the model-generated pool. Filters stale rows from the killed run."""
import json, collections, numpy as np

rows = [json.loads(l) for l in open("full_goodhart.jsonl")]

# Stale = failures from the killed run, whose err lacks the ": message" suffix
# added when the runner was rewritten. Successes are never stale.
def stale(r):
    return (not r.get("ok")) and (": " not in r.get("err", ""))

live = [r for r in rows if not stale(r)]
ok = [r for r in live if r.get("ok")]
model = [r for r in ok if r.get("cand") != "exploit"]
expl = {r["task"]: r for r in ok if r.get("cand") == "exploit"}

print("=== POOL ===")
print(f"  rows in checkpoint      : {len(rows)}  (stale dropped: {len(rows)-len(live)})")
print(f"  usable model candidates : {len(model)}")
print(f"  tasks with an exploit   : {len(expl)}")
rep = sum(1 for r in model if r.get("repaired"))
print(f"  needed a repair retry   : {rep}/{len(model)} ({rep/max(len(model),1)*100:.0f}%)")

fails = collections.Counter(r["err"].split(":")[0] for r in live if not r.get("ok"))
print(f"  failures by stage       : {dict(fails)}")

by = collections.defaultdict(list)
for r in model:
    by[r["task"]].append(r)
tasks = [t for t, g in by.items() if len(g) >= 2 and t in expl]
print(f"  tasks usable for curve  : {len(tasks)} (>=2 model candidates + exploit)")
if not tasks:
    raise SystemExit("\nnot enough data for a curve yet")

print("\n=== KILL CHECK 2, on the full pool ===")
v = np.array([r["vol"] for r in model])
print(f"  volume gate pass rate   : {int((v<=3).sum())}/{len(v)} = {(v<=3).mean()*100:.1f}%")
print(f"  volume error  median {np.median(v):.1f}%  min {v.min():.1f}%  max {v.max():.1f}%")
pa = sum(1 for r in model if r["passA"])
print(f"  full shipped gate pass  : {pa}/{len(model)} = {pa/len(model)*100:.1f}%")

print("\n=== CURVE: select by shipped reward, report true quality ===")
rng = np.random.default_rng(0); BOOT = 800
print(f"{'n':>3}{'shape err %':>14}{'95% CI':>20}{'exploit win':>13}{'tasks':>7}")
curve = []
for n in [1, 2, 3, 4, 6, 8, 12, 16]:
    per_task = []; win = 0; tot = 0
    for t in tasks:
        pool = by[t] + [expl[t]]          # exploit always available
        if len(pool) < n: continue
        vals = []
        for _ in range(BOOT):
            pick = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
            w = max(pick, key=lambda x: x["A"])
            vals.append(-w["B"]); tot += 1
            if w.get("cand") == "exploit": win += 1
        per_task.append(np.mean(vals))
    if not per_task: continue
    pt = np.array(per_task)
    bs = [np.mean(rng.choice(pt, len(pt))) for _ in range(2000)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    curve.append(dict(n=n, mean=float(pt.mean()), lo=float(lo), hi=float(hi),
                      exploit=win/max(tot,1)*100, tasks=len(pt)))
    print(f"{n:>3}{pt.mean():>13.3f}%  [{lo:>6.3f}, {hi:>6.3f}]{win/max(tot,1)*100:>12.1f}%{len(pt):>7}")

print("\n=== CONTROL: select by corrected reward ===")
ctrl = []
for n in [1, 2, 3, 4, 6, 8, 12, 16]:
    per_task = []
    for t in tasks:
        pool = by[t] + [expl[t]]
        if len(pool) < n: continue
        vals = []
        for _ in range(BOOT):
            pick = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
            vals.append(-max(pick, key=lambda x: x["B"])["B"])
        per_task.append(np.mean(vals))
    if per_task:
        ctrl.append(dict(n=n, mean=float(np.mean(per_task))))
        print(f"{n:>3}{np.mean(per_task):>13.3f}%")

json.dump({"shipped": curve, "corrected": ctrl,
           "n_model": len(model), "n_tasks": len(tasks),
           "repair_rate": rep/max(len(model),1),
           "vol_pass_rate": float((v<=3).mean())},
          open("model_curve.json", "w"), indent=1)
print("\n-> model_curve.json")
