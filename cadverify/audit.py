"""Pre-delivery checks for a CAD corpus.

Every check here exists because a real delivery failed it. Nothing is
hypothetical: the completeness check corresponds to 30 samples that shipped
labelled "approved" with no trajectory, the index check to 6 folders named with
an id absent from their own contents, the density check to a mass computed at
the wrong material and re-shipped unfixed after being reported, and the answer-key
check to a bounding box recorded in a different axis order than the STEP beside it.

Severities
    reject      the sample is not usable for its stated purpose
    quarantine  usable, but something is wrong that a buyer will find
    note        worth knowing, not a defect

Geometry checks need OpenCascade. Without it the structural checks still run and
the geometry ones report as skipped rather than silently passing.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field

REJECT, QUARANTINE, NOTE = "reject", "quarantine", "note"


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    sample_id: str | None = None
    detail: dict = field(default_factory=dict)


@dataclass
class Corpus:
    root: str
    samples: list = field(default_factory=list)     # (folder_path, sample.json dict)
    dataset: list = field(default_factory=list)     # DATASET.jsonl rows
    metadata: dict = field(default_factory=dict)

    @classmethod
    def load(cls, root: str) -> "Corpus":
        c = cls(root=root)
        for p in sorted(glob.glob(os.path.join(root, "samples", "*", "*"))):
            sj = os.path.join(p, "sample.json")
            if os.path.exists(sj):
                c.samples.append((p, json.load(open(sj))))
        dpath = os.path.join(root, "DATASET.jsonl")
        if os.path.exists(dpath):
            c.dataset = [json.loads(l) for l in open(dpath) if l.strip()]
        mpath = os.path.join(root, "metadata.json")
        if os.path.exists(mpath):
            c.metadata = json.load(open(mpath))
        return c


def _pct(a, b):
    return abs(a - b) / abs(b) * 100 if b else float("inf")


# ---------------------------------------------------------------- structural

def check_completeness(c: Corpus):
    """A sample whose process data is empty is not the thing that was ordered."""
    for path, sj in c.samples:
        sid = sj.get("id", os.path.basename(path))
        empty = []

        traj = os.path.join(path, "submission", "steps.json")
        if os.path.exists(traj):
            try:
                if not json.load(open(traj)):
                    empty.append("trajectory")
            except Exception:
                empty.append("trajectory (unparseable)")

        feats = os.path.join(path, "submission", "final.features.json")
        if os.path.exists(feats):
            try:
                if not json.load(open(feats)).get("features"):
                    empty.append("feature tree")
            except Exception:
                empty.append("feature tree (unparseable)")

        m = (sj.get("submission") or {}).get("measurements") or {}
        if m and m.get("volumeMm3") is None:
            empty.append("measurements")

        if empty:
            yield Finding("completeness", REJECT,
                          "empty: " + ", ".join(empty), sid,
                          {"capture_mode": (sj.get("provenance") or {}).get("captureMode")})


def check_index_integrity(c: Corpus):
    """Folder name, sample.json, DATASET.jsonl and metadata.json must agree."""
    inner = {}
    for path, sj in c.samples:
        base = os.path.basename(path)
        folder_id = base.split("--", 1)[1] if "--" in base else base
        sid = sj.get("id")
        inner[sid] = path
        if sid and folder_id != sid:
            yield Finding("index", QUARANTINE,
                          f"folder is named {folder_id[:8]} but contains {sid[:8]}",
                          sid, {"folder": base})

    if c.dataset:
        ds_ids = {d.get("id") for d in c.dataset}
        missing = ds_ids - set(inner)
        orphan = set(inner) - ds_ids
        for i in sorted(x for x in missing if x):
            yield Finding("index", REJECT, "listed in DATASET.jsonl, no sample on disk", i)
        for i in sorted(x for x in orphan if x):
            yield Finding("index", REJECT, "sample on disk, absent from DATASET.jsonl", i)

    if c.metadata.get("samples"):
        declared = c.metadata.get("sampleCount")
        if declared is not None and declared != len(c.samples):
            yield Finding("index", REJECT,
                          f"metadata declares {declared} samples, {len(c.samples)} on disk")


def check_manifest(c: Corpus):
    """metadata.json's per-sample byte and file counts must match the disk."""
    by_folder = {m.get("folder"): m for m in c.metadata.get("samples", [])}
    for path, sj in c.samples:
        m = by_folder.get(os.path.basename(path))
        if not m:
            continue
        files = [f for f in glob.glob(os.path.join(path, "**", "*"), recursive=True)
                 if os.path.isfile(f)]
        total = sum(os.path.getsize(f) for f in files)
        if m.get("fileCount") not in (None, len(files)):
            yield Finding("manifest", QUARANTINE,
                          f"fileCount {m['fileCount']} declared, {len(files)} present",
                          sj.get("id"))
        if m.get("totalBytes") not in (None, total):
            yield Finding("manifest", QUARANTINE,
                          f"totalBytes {m['totalBytes']:,} declared, {total:,} present",
                          sj.get("id"))


def check_density_consistency(c: Corpus):
    """mass must equal volume x the declared density for the stated material."""
    for _, sj in c.samples:
        m = (sj.get("submission") or {}).get("measurements") or {}
        dens = sj.get("densityGPerMm3")
        if not (m.get("massGrams") and m.get("volumeMm3") and dens):
            continue
        implied = m["massGrams"] / m["volumeMm3"]
        if _pct(implied, dens) > 1.0:
            yield Finding("density", REJECT,
                          f"mass implies density {implied:.5f} but material "
                          f"{sj.get('material')} is {dens}",
                          sj.get("id"),
                          {"implied": implied, "declared": dens,
                           "material": sj.get("material")})


def check_answer_key_precision(c: Corpus):
    """A key rounded to 4 significant figures spends part of the tolerance budget."""
    rounded = 0
    total = 0
    for _, sj in c.samples:
        v = (sj.get("answerKey") or {}).get("targetVolumeMm3")
        if not v:
            continue
        total += 1
        if float(v) == float(f"%.4g" % v):
            rounded += 1
    if total and rounded / total > 0.5:
        yield Finding("precision", NOTE,
                      f"{rounded}/{total} answer-key volumes are rounded to 4 significant "
                      "figures, which consumes up to ~0.05% of a 3% tolerance for nothing")


def check_duplicates(c: Corpus):
    """Identical inputs across a delivery are usually a re-ship, not a sample."""
    seen = defaultdict(list)
    for path, sj in c.samples:
        d = os.path.join(path, "input", "drawing.pdf")
        if os.path.exists(d):
            seen[hashlib.sha256(open(d, "rb").read()).hexdigest()].append(sj.get("id"))
    for h, ids in seen.items():
        if len(ids) > 1:
            yield Finding("duplicates", QUARANTINE,
                          f"{len(ids)} samples share one drawing", ids[0],
                          {"ids": ids})


def check_distribution(c: Corpus):
    """Composition, reported rather than gated — a buyer's spec decides what is right."""
    for field_name in ("difficulty", "material", "units", "tolerancePercent", "track"):
        counts = Counter(sj.get(field_name) for _, sj in c.samples)
        if len(counts) > 1 or field_name in ("difficulty", "track"):
            yield Finding("distribution", NOTE, f"{field_name}: {dict(counts)}")
    modes = Counter((sj.get("provenance") or {}).get("captureMode") for _, sj in c.samples)
    yield Finding("distribution", NOTE, f"captureMode: {dict(modes)}")


# ------------------------------------------------------------------ geometry

def check_answer_key_matches_reference(c: Corpus, limit=None):
    """The reference solid must satisfy its own answer key.

    This is the check that matters most and the one nobody runs. If the ground
    truth fails the key, every submission is being scored against a target the
    correct answer does not hit.
    """
    try:
        from .invariants import invariants, load_step
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
    except Exception as e:
        yield Finding("answer-key", NOTE,
                      f"skipped — OpenCascade unavailable ({type(e).__name__})")
        return

    def dims(s):
        b = Bnd_Box()
        BRepBndLib.Add_s(s, b, True)
        x0, y0, z0, x1, y1, z1 = b.Get()
        return (x1 - x0, y1 - y0, z1 - z0)

    vol_fail = box_fail = perm_ok = n = 0
    for path, sj in c.samples[:limit]:
        ak = sj.get("answerKey") or {}
        ref = os.path.join(path, "ground_truth", "reference.step")
        if not (ak.get("targetVolumeMm3") and os.path.exists(ref)):
            continue
        try:
            shape = load_step(ref)
        except Exception as e:
            yield Finding("answer-key", REJECT,
                          f"reference STEP will not load: {type(e).__name__}", sj.get("id"))
            continue
        n += 1
        tol = sj.get("tolerancePercent", 3)
        inv = invariants(shape)
        if _pct(inv.volume, ak["targetVolumeMm3"]) > tol:
            vol_fail += 1
            yield Finding("answer-key", REJECT,
                          f"reference volume misses its own key by "
                          f"{_pct(inv.volume, ak['targetVolumeMm3']):.2f}%", sj.get("id"))
        tb = ak.get("targetBoundingBoxMm")
        if tb:
            bb = dims(shape)
            t = [tb[k] for k in "xyz"]
            if max(_pct(bb[i], t[i]) for i in range(3)) > tol:
                box_fail += 1
                if min(max(_pct(bb[p[i]], t[i]) for i in range(3))
                       for p in itertools.permutations(range(3))) <= tol:
                    perm_ok += 1

    if n and box_fail:
        msg = (f"the reference part fails its own bounding-box gate on {box_fail}/{n} "
               f"samples ({box_fail/n*100:.0f}%)")
        if perm_ok:
            msg += (f"; {perm_ok} of those pass under an axis permutation, so the key "
                    "records the box in a different axis order than the STEP")
        yield Finding("answer-key", REJECT, msg, None,
                      {"bbox_fail": box_fail, "recovered_by_permutation": perm_ok, "n": n})
    if n and not box_fail and not vol_fail:
        yield Finding("answer-key", NOTE,
                      f"reference satisfies its own key on all {n} samples checked")


STRUCTURAL = [check_completeness, check_index_integrity, check_manifest,
              check_density_consistency, check_answer_key_precision,
              check_duplicates, check_distribution]
GEOMETRIC = [check_answer_key_matches_reference]


def audit(root: str, geometry: bool = True, limit=None):
    c = Corpus.load(root)
    findings = []
    for chk in STRUCTURAL:
        findings.extend(chk(c))
    if geometry:
        findings.extend(check_answer_key_matches_reference(c, limit=limit))
    return c, findings


def verdicts(c: Corpus, findings):
    """Per-sample accept / quarantine / reject."""
    worst = {}
    rank = {NOTE: 0, QUARANTINE: 1, REJECT: 2}
    for f in findings:
        if f.sample_id is None or f.severity == NOTE:
            continue
        if rank[f.severity] > rank.get(worst.get(f.sample_id, NOTE), 0):
            worst[f.sample_id] = f.severity
    out = {}
    for _, sj in c.samples:
        sid = sj.get("id")
        out[sid] = worst.get(sid, "accept")
    return out
