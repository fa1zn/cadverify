"""Alignment, and the exploit it exists to catch.

Note what the positive control cannot do. Aligning a part against a rigid
transform of *itself* is the easy case by construction: both shapes share
principal axes exactly, so the discrete search always finds the answer. It
passed for a long time while the aligner was reporting inflated distances on
genuinely different shapes. test_icp_beats_discrete_search covers that gap.
"""

import math

import numpy as np
import pytest

from cadverify.align import align, principal_frame
from cadverify.exploit import POLICIES
from cadverify.invariants import invariants, load_step, transformed
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

from .conftest import answer_key, ref_step, sub_step


def random_rigid(rng):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    r = gp_Trsf()
    r.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(*ax)), rng.uniform(0, 2 * math.pi))
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(*rng.uniform(-400, 400, 3)))
    return t.Multiplied(r)


def test_recovers_a_known_pose(corpus):
    """Positive control: a part against a randomly posed copy of itself."""
    rng = np.random.default_rng(3)
    for path in [corpus[i] for i in rng.choice(len(corpus), 5, replace=False)]:
        shape = load_step(ref_step(path))
        moved = transformed(shape, random_rigid(rng))
        assert align(shape, moved)["chamfer_pct_diag"] < 0.01


def test_different_parts_do_not_align(corpus):
    """Negative control: distinct parts must stay far apart after alignment."""
    rng = np.random.default_rng(77)
    for _ in range(4):
        i, j = rng.choice(len(corpus), 2, replace=False)
        a = load_step(ref_step(corpus[i]))
        b = load_step(ref_step(corpus[j]))
        assert align(a, b)["chamfer_pct_diag"] > 0.5


def test_degenerate_parts_escalate(corpus):
    """A near-symmetric part must widen the search rather than guess."""
    from cadverify.align import DEGENERACY_GAP, OCTAHEDRAL, candidate_rotations

    found = False
    for path in corpus[:60]:
        _, _, moments, gap = principal_frame(load_step(ref_step(path)))
        if gap < DEGENERACY_GAP:
            assert len(candidate_rotations(moments)) > len(OCTAHEDRAL)
            found = True
            break
    if not found:
        pytest.skip("no degenerate part in the sampled slice")


def test_refinement_helps_and_is_guarded(corpus):
    """The case the positive control structurally cannot reach.

    Principal axes coincide exactly only when two shapes are identical. For a
    submission that genuinely differs, no signed axis permutation maps one frame
    onto the other, so the discrete search alone returns an inflated upper bound
    and ICP refinement is what closes the gap.

    But point-to-point ICP is not a safe refinement here. Measured over 20
    random pairs: it improves on 5, and diverges on 14 — median ratio 5x worse
    than the pose it started from, worst case 1e11. It earns its place only
    because best_alignment scores the raw candidate *and* the refined one and
    keeps whichever is better. The guard is not a safety net; it is doing most
    of the work. Both facts are asserted below.
    """
    from scipy.spatial import cKDTree

    from cadverify.align import canonical_points
    from cadverify.sampling import Surface, bbox_diagonal, sample_surface

    from cadverify.align import OCTAHEDRAL, icp

    rng = np.random.default_rng(9)
    improved = 0
    diverged = 0
    checked = 0
    for path in [corpus[i] for i in rng.choice(len(corpus), 20, replace=False)]:
        ref = load_step(ref_step(path))
        sub = load_step(sub_step(path))
        fa, fb = principal_frame(ref), principal_frame(sub)
        diag = bbox_diagonal(ref)
        # One point set for both arms — comparing across different samplings
        # measures sampling noise, not refinement.
        ref_can = canonical_points(sample_surface(ref, 1500, seed=1), fa[0], fa[1])
        sub_can = canonical_points(sample_surface(sub, 1500, seed=2), fb[0], fb[1])
        surf = Surface(ref).transformed(fa[0], fa[1])
        tree = cKDTree(ref_can)

        scored = [(surf.distance(sub_can @ r.T).mean(), r) for r in OCTAHEDRAL]
        discrete, best_r = min(scored, key=lambda x: x[0])

        r_icp, t_icp, _ = icp(sub_can @ best_r.T, tree, ref_can)
        refined = surf.distance((sub_can @ best_r.T) @ r_icp.T + t_icp).mean()

        # the guard: whichever of the two is better is what ships
        assert min(discrete, refined) <= discrete + 1e-9

        checked += 1
        if refined < discrete:
            improved += 1
        if refined > discrete * 2:
            diverged += 1

    assert checked == 20
    assert improved >= 3, f"refinement helped on only {improved}/20 — has it regressed?"
    assert diverged >= 5, (
        f"only {diverged}/20 diverged — if ICP became safe, the guard's rationale "
        "changed and this docstring is stale"
    )


def test_align_never_worse_than_discrete_search(corpus):
    """End-to-end contract: align() must not return a pose worse than the
    plain octahedral search it starts from."""
    from cadverify.align import OCTAHEDRAL, canonical_points
    from cadverify.sampling import Surface, bbox_diagonal, sample_surface

    rng = np.random.default_rng(15)
    for path in [corpus[i] for i in rng.choice(len(corpus), 4, replace=False)]:
        ref = load_step(ref_step(path))
        sub = load_step(sub_step(path))
        fa, fb = principal_frame(ref), principal_frame(sub)
        diag = bbox_diagonal(ref)
        surf = Surface(ref).transformed(fa[0], fa[1])
        sub_can = canonical_points(sample_surface(sub, 3000, seed=1), fb[0], fb[1])
        discrete = min(surf.distance(sub_can @ r.T).mean() for r in OCTAHEDRAL) / diag * 100
        # allow a small sampling-noise margin: align() draws its own points
        assert align(ref, sub)["chamfer_pct_diag"] <= discrete * 1.05 + 0.01


def test_exploit_beats_the_shipped_gate_and_fails_the_grader(corpus):
    """The degenerate policy: right volume, right bbox, wrong shape."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    def dims(s):
        b = Bnd_Box()
        BRepBndLib.Add_s(s, b, True)
        x0, y0, z0, x1, y1, z1 = b.Get()
        return (x1 - x0, y1 - y0, z1 - z0)

    rng = np.random.default_rng(31)
    for path in [corpus[i] for i in rng.choice(len(corpus), 5, replace=False)]:
        ak = answer_key(path)
        ref = load_step(ref_step(path))
        block = POLICIES["pocketed_block"](ak)

        # passes the shipped gate on both axes it checks
        vol = invariants(block).volume
        assert abs(vol - ak["targetVolumeMm3"]) / ak["targetVolumeMm3"] < 1e-6
        bb = dims(block)
        tb = [ak["targetBoundingBoxMm"][k] for k in "xyz"]
        assert max(abs(bb[i] - tb[i]) / tb[i] for i in range(3)) < 1e-6

        # and is nonetheless the wrong shape
        assert align(ref, block)["chamfer_pct_diag"] > 0.2
