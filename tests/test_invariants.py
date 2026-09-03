"""Invariants come off the B-rep exactly and survive rigid motion."""

import json
import math

import numpy as np
import pytest

from cadverify.invariants import invariants, load_step, transformed
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

from .conftest import answer_key, ref_step


def random_rigid(rng):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    r = gp_Trsf()
    r.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(*ax)), rng.uniform(0, 2 * math.pi))
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(*rng.uniform(-500, 500, 3)))
    return t.Multiplied(r)


def test_rigid_motion_is_actually_applied(sample):
    """Guard against a vacuous invariance test: confirm the transform moves the part."""
    shape = load_step(ref_step(sample))
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(0, 1000, 0))
    before, after = invariants(shape), invariants(transformed(shape, t))
    assert after.com[1] - before.com[1] == pytest.approx(1000.0, abs=1e-6)


def test_invariants_survive_rigid_motion(corpus):
    rng = np.random.default_rng(11)
    for path in [corpus[i] for i in rng.choice(len(corpus), 6, replace=False)]:
        shape = load_step(ref_step(path))
        base = invariants(shape)
        for _ in range(3):
            moved = invariants(transformed(shape, random_rigid(rng)))
            assert abs(moved.volume - base.volume) / base.volume < 1e-11
            assert abs(moved.area - base.area) / base.area < 1e-11
            for a, b in zip(moved.moments, base.moments):
                assert abs(a - b) / b < 1e-9
            assert moved.n_faces == base.n_faces
            assert moved.n_edges == base.n_edges


def test_agrees_with_shipped_answer_key(corpus):
    """Disagreement with the key must fall inside the key's own 4-sig-fig rounding."""
    rng = np.random.default_rng(2)
    for path in [corpus[i] for i in rng.choice(len(corpus), 12, replace=False)]:
        ak = answer_key(path)
        inv = invariants(load_step(ref_step(path)))
        for mine, theirs in ((inv.volume, ak["targetVolumeMm3"]),
                             (inv.area, ak["targetSurfaceAreaMm2"])):
            if not theirs:
                continue
            band = 0.5 * 10 ** (math.floor(math.log10(abs(theirs))) - 3)
            assert abs(mine - theirs) <= band * 1.000001, (
                f"{path}: {mine} vs key {theirs}, outside rounding band {band}"
            )


def test_ground_truth_fails_its_own_bbox_gate(corpus):
    """Regression guard on the headline finding.

    The answer key records the bounding box in a different axis order than the
    STEP file it ships beside, so the reference part fails its own gate on most
    samples while passing on volume everywhere.
    """
    import itertools

    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    def dims(s):
        b = Bnd_Box()
        BRepBndLib.Add_s(s, b, True)
        x0, y0, z0, x1, y1, z1 = b.Get()
        return (x1 - x0, y1 - y0, z1 - z0)

    def pct(a, b):
        return abs(a - b) / abs(b) * 100 if b else float("inf")

    rng = np.random.default_rng(5)
    raw = perm = vol = n = 0
    for path in [corpus[i] for i in rng.choice(len(corpus), 25, replace=False)]:
        ak = answer_key(path)
        shape = load_step(ref_step(path))
        bb = dims(shape)
        tb = [ak["targetBoundingBoxMm"][k] for k in "xyz"]
        n += 1
        if max(pct(bb[i], tb[i]) for i in range(3)) <= 3:
            raw += 1
        if min(max(pct(bb[q[i]], tb[i]) for i in range(3))
               for q in itertools.permutations(range(3))) <= 3:
            perm += 1
        if pct(invariants(shape).volume, ak["targetVolumeMm3"]) <= 3:
            vol += 1

    assert vol == n, "volume should match the key on every sample"
    assert raw / n < 0.5, "expected the reference to fail its own bbox gate on most samples"
    assert perm / n > 0.85, "axis permutation should recover nearly all of them"
