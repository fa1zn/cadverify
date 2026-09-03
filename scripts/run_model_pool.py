"""Goodhart curve, model-generated candidates. Parallel, checkpointed, one repair retry."""
import os, re, base64, glob, json, time, sys, tempfile, subprocess, threading
sys.path.insert(0, "/Users/faizansyed/normal/cadverify")
_src = open(os.path.expanduser("~/.zshrc")).read()
os.environ["ANTHROPIC_API_KEY"] = re.search(
    r'^\s*export\s+ANTHROPIC_API_KEY=["\']?([^"\'\s]+)', _src, re.M).group(1)

import anthropic, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from cadverify.invariants import load_step, invariants
from cadverify.align import align
from cadverify.exploit import POLICIES
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

ROOT = "/Users/faizansyed/Downloads/collection-selection/samples/generate-3d-from-2d"
VENV = "/Users/faizansyed/normal/cadverify/.venv/bin/python"
CKPT = "/Users/faizansyed/normal/cadverify/full_goodhart.jsonl"
N_TASKS, N_CAND, WORKERS = 40, 16, 8
MODEL = "claude-opus-5"

PROMPT = ("You are given an engineering drawing. Model the part in build123d (Python).\n"
          "Reply with ONLY one ```python code block. Define a module-level variable `result` "
          "holding the final build123d Part or Solid. All dimensions in millimetres. "
          "Do not print anything, do not export, no explanation.")

RUNNER = '\n'.join([
    "import sys",
    'sys.path.insert(0,"/Users/faizansyed/normal/cadverify")',
    "from build123d import *",
    "ns={}",
    "exec(open(sys.argv[1]).read(), ns)",
    'r=ns.get("result")',
    'if r is None: raise SystemExit("no result")',
    'shape=r.wrapped if hasattr(r,"wrapped") else r',
    "from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs",
    "w=STEPControl_Writer(); w.Transfer(shape, STEPControl_AsIs); w.Write(sys.argv[2])",
])

client = anthropic.Anthropic(max_retries=4)
lock = threading.Lock()
usage = {"in": 0, "out": 0}
CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def dims(s):
    b = Bnd_Box(); BRepBndLib.Add_s(s, b, True)
    x0, y0, z0, x1, y1, z1 = b.Get()
    return (x1 - x0, y1 - y0, z1 - z0)


def pct(a, b):
    return abs(a - b) / abs(b) * 100 if b else float("inf")


def ask(messages):
    with client.messages.stream(model=MODEL, max_tokens=16000,
                                thinking={"type": "adaptive"},
                                output_config={"effort": "medium"},
                                messages=messages) as s:
        m = s.get_final_message()
    with lock:
        usage["in"] += m.usage.input_tokens
        usage["out"] += m.usage.output_tokens
    txt = "".join(b.text for b in m.content if b.type == "text")
    hit = CODE_RE.search(txt)
    return hit.group(1) if hit else txt


def run_script(code, sp, rp, op):
    open(sp, "w").write(code)
    r = subprocess.run([VENV, rp, sp, op], capture_output=True, text=True, timeout=180)
    return (r.returncode == 0 and os.path.exists(op)), (r.stderr or "")


def one(p, sj, ak, pdf, ref, i, density, done):
    if (sj["id"], i) in done:
        return None
    rec = {"task": sj["id"], "title": sj["title"], "i": i}
    first = [{"role": "user", "content": [
        {"type": "document", "source": {"type": "base64",
                                        "media_type": "application/pdf", "data": pdf}},
        {"type": "text", "text": PROMPT}]}]
    try:
        code = ask(first)
    except Exception as e:
        rec.update(ok=False, err="api:%s: %s" % (type(e).__name__, str(e)[:150]))
        return rec

    sp = tempfile.mktemp(suffix=".py"); rp = tempfile.mktemp(suffix=".py")
    op = tempfile.mktemp(suffix=".step")
    open(rp, "w").write(RUNNER)
    repaired = False
    try:
        ok, err = run_script(code, sp, rp, op)
        if not ok:
            repaired = True
            followup = first + [
                {"role": "assistant", "content": "```python\n" + code + "\n```"},
                {"role": "user", "content": "That script failed with:\n\n" + err[-1500:] +
                 "\n\nReturn a corrected version. Same rules: one ```python block, "
                 "module-level `result`."}]
            try:
                code = ask(followup)
            except Exception as e:
                rec.update(ok=False, repaired=True,
                           err="repair:%s: %s" % (type(e).__name__, str(e)[:120]))
                return rec
            ok, err = run_script(code, sp, rp, op)
            if not ok:
                rec.update(ok=False, repaired=True, err="exec2:" + err[-120:])
                return rec
    except Exception as e:
        rec.update(ok=False, repaired=repaired, err="exec:%s" % type(e).__name__)
        return rec

    rec["repaired"] = repaired
    try:
        sub = load_step(op); inv = invariants(sub); bb = dims(sub)
        ve = pct(inv.volume, ak["targetVolumeMm3"])
        tb = [ak["targetBoundingBoxMm"][k] for k in "xyz"]
        be = max(pct(bb[j], tb[j]) for j in range(3))
        me = pct(inv.volume * density, ak["targetMassGrams"]) if ak["targetMassGrams"] else 0.0
        rec.update(ok=True, A=-max(ve, be, me),
                   B=-align(ref, sub)["chamfer_pct_diag"],
                   vol=ve, bbox=be, mass=me,
                   passA=bool(ve <= 3 and be <= 3 and me <= 3))
    except Exception as e:
        rec.update(ok=False, err="grade:%s" % type(e).__name__)
    return rec


def main():
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT):
            try:
                r = json.loads(line)
                if r.get("ok"):
                    done.add((r["task"], r["i"]))
            except Exception:
                pass
    print("resuming: %d usable candidates already done" % len(done), flush=True)
    out = open(CKPT, "a")

    paths = sorted(glob.glob(ROOT + "/*"))
    rng = np.random.default_rng(23)
    sel = [paths[i] for i in rng.choice(len(paths), N_TASKS, replace=False)]

    t0 = time.time(); made = 0
    for ti, p in enumerate(sel):
        sj = json.load(open(p + "/sample.json")); ak = sj["answerKey"]
        density = sj["densityGPerMm3"]
        pdf = base64.standard_b64encode(open(p + "/input/drawing.pdf", "rb").read()).decode()
        ref = load_step(p + "/ground_truth/reference.step")

        if (sj["id"], -1) not in done:
            ex = POLICIES["pocketed_block"](ak); inv = invariants(ex); bb = dims(ex)
            ve = pct(inv.volume, ak["targetVolumeMm3"])
            tb = [ak["targetBoundingBoxMm"][k] for k in "xyz"]
            be = max(pct(bb[j], tb[j]) for j in range(3))
            me = pct(inv.volume * density, ak["targetMassGrams"]) if ak["targetMassGrams"] else 0.0
            out.write(json.dumps({"task": sj["id"], "title": sj["title"], "i": -1, "ok": True,
                                  "cand": "exploit", "A": -max(ve, be, me),
                                  "B": -align(ref, ex)["chamfer_pct_diag"],
                                  "vol": ve, "bbox": be, "mass": me,
                                  "passA": bool(ve <= 3 and be <= 3 and me <= 3)}) + "\n")
            out.flush()

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = [pool.submit(one, p, sj, ak, pdf, ref, i, density, done)
                    for i in range(N_CAND)]
            for f in as_completed(futs):
                r = f.result()
                if r is None:
                    continue
                out.write(json.dumps(r) + "\n"); out.flush(); made += 1

        cost = (usage["in"] * 5 + usage["out"] * 25) / 1e6
        print("[%2d/%d] %-38s %4d gens  $%.2f  %.0fs"
              % (ti + 1, N_TASKS, sj["title"][:38], made, cost, time.time() - t0), flush=True)

    out.close()
    print("\nDONE: %d generations, $%.2f, %.0fs"
          % (made, (usage["in"] * 5 + usage["out"] * 25) / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
