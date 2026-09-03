"""Stage B — bring two shapes into a common frame before comparing them.

The reference and the submission are the same part in different poses. Any
frame-dependent measurement disagrees until that is undone.

Two-tier cost model: candidate poses are ranked with cheap point-to-point
distance (its sampling noise is common to every candidate, so it does not affect
the ranking), and only the winner is scored with exact point-to-surface distance.
"""

import itertools

import numpy as np
from scipy.spatial import cKDTree

from .invariants import invariants

DEGENERACY_GAP = 0.02       # relative moment gap below which principal axes are unreliable
SWEEP_STEPS = 72            # rotations sampled about a degenerate axis (5 degree steps)


def _proper_signed_permutations() -> np.ndarray:
    """The 24 rotations of the cube: signed axis permutations with determinant +1."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, col in enumerate(perm):
                m[row, col] = signs[row]
            if np.linalg.det(m) > 0:
                out.append(m)
    return np.array(out)


OCTAHEDRAL = _proper_signed_permutations()


def principal_frame(shape):
    """Return (centre_of_mass, R, moments, gap).

    R's rows are the principal axes ordered by ascending moment, with det(R)=+1.
    Mapping a point: q = R @ (p - com).
    """
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    c = g.CentreOfMass()
    pp = g.PrincipalProperties()

    moments = np.array(pp.Moments())
    axes = np.array(
        [
            [pp.FirstAxisOfInertia().X(), pp.FirstAxisOfInertia().Y(), pp.FirstAxisOfInertia().Z()],
            [pp.SecondAxisOfInertia().X(), pp.SecondAxisOfInertia().Y(), pp.SecondAxisOfInertia().Z()],
            [pp.ThirdAxisOfInertia().X(), pp.ThirdAxisOfInertia().Y(), pp.ThirdAxisOfInertia().Z()],
        ]
    )
    order = np.argsort(moments)
    moments, axes = moments[order], axes[order]
    if np.linalg.det(axes) < 0:          # never accept a reflection
        axes[2] = -axes[2]

    m = np.sort(moments)
    gap = min((m[1] - m[0]) / m[1], (m[2] - m[1]) / m[2]) if m[2] > 0 else 0.0
    return np.array([c.X(), c.Y(), c.Z()]), axes, moments, float(gap)


def _degenerate_axis(moments: np.ndarray) -> int | None:
    """Index of the axis about which the shape is rotationally ambiguous, if any."""
    m = moments
    if abs(m[1] - m[0]) / max(m[1], 1e-30) < DEGENERACY_GAP:
        return 2      # the two smallest match -> spin about the largest-moment axis
    if abs(m[2] - m[1]) / max(m[2], 1e-30) < DEGENERACY_GAP:
        return 0      # the two largest match -> spin about the smallest-moment axis
    return None


def _axis_rotations(axis: int, steps: int) -> np.ndarray:
    out = []
    for th in np.linspace(0, 2 * np.pi, steps, endpoint=False):
        c, s = np.cos(th), np.sin(th)
        r = np.eye(3)
        i, j = [k for k in range(3) if k != axis]
        r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
        out.append(r)
    return np.array(out)


def candidate_rotations(moments: np.ndarray) -> np.ndarray:
    """Every rotation that could map one canonical frame onto the other."""
    cands = OCTAHEDRAL
    ax = _degenerate_axis(moments)
    if ax is not None:
        sweep = _axis_rotations(ax, SWEEP_STEPS)
        cands = np.einsum("aij,bjk->abik", cands, sweep).reshape(-1, 3, 3)
    return cands


def canonical_points(pts: np.ndarray, com: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return (pts - com) @ axes.T


TOP_N = 4   # candidates re-scored exactly. Measured: the exact winner never ranked
            # worse than 2nd under cheap scoring across 31 trials. 4 is margin.


def best_alignment(ref_pts, ref_frame, sub_pts, sub_frame, ref_surface_canonical, top_n=TOP_N):
    """Search candidate poses, then re-score the leaders exactly.

    Cheap point-to-point distance ranks every candidate — its sampling noise is
    common to all of them, so it orders them well but does not pick reliably.
    On 2 of 31 trials the cheap winner was the wrong pose, costing up to 0.066%
    of the bbox diagonal. Re-scoring the top few with exact point-to-surface
    distance fixes that for ~1s per candidate.

    Returns (rotation, exact_score, n_candidates).
    """
    ref_com, ref_axes, ref_moments, _ = ref_frame
    sub_com, sub_axes, _, _ = sub_frame

    ref_can = canonical_points(ref_pts, ref_com, ref_axes)
    sub_can = canonical_points(sub_pts, sub_com, sub_axes)
    tree = cKDTree(ref_can)

    cands = candidate_rotations(ref_moments)
    cheap = np.empty(len(cands))
    for i, r in enumerate(cands):
        d, _ = tree.query(sub_can @ r.T, workers=-1)
        cheap[i] = d.mean()

    leaders = np.argsort(cheap)[: min(top_n, len(cands))]
    best_r, best_t, best_s = None, np.zeros(3), np.inf
    for i in leaders:
        start = sub_can @ cands[i].T
        # Refine continuously. The discrete search only gets you to the right
        # basin: principal axes coincide exactly only when the two shapes are
        # identical, so for a submission that genuinely differs, no signed
        # permutation maps one frame onto the other. Without this step every
        # distance is an inflated upper bound.
        r_icp, t_icp, _ = icp(start, tree, ref_can)
        for rot, trans in ((cands[i], np.zeros(3)), (r_icp @ cands[i], t_icp)):
            s = float(ref_surface_canonical.distance(sub_can @ rot.T + trans).mean())
            if s < best_s:
                best_r, best_t, best_s = rot, trans, s
    return best_r, best_t, best_s, len(cands)


def align(ref_shape, sub_shape, n_points: int = 3000, seed: int = 0):
    """Align a submission to a reference. Returns a dict of alignment results."""
    from .sampling import Surface, bbox_diagonal, sample_surface

    ref_frame = principal_frame(ref_shape)
    sub_frame = principal_frame(sub_shape)
    diag = bbox_diagonal(ref_shape)

    ref_pts = sample_surface(ref_shape, n_points, seed=seed)
    sub_pts = sample_surface(sub_shape, n_points, seed=seed + 1)
    ref_surf_can = Surface(ref_shape).transformed(ref_frame[0], ref_frame[1])

    rot, trans, score, n_cands = best_alignment(
        ref_pts, ref_frame, sub_pts, sub_frame, ref_surf_can
    )
    return {
        "rotation": rot,
        "translation": trans,
        "chamfer_mm": score,
        "chamfer_pct_diag": score / diag * 100 if diag else float("nan"),
        "n_candidates": n_cands,
        "moment_gap": ref_frame[3],
        "degenerate": ref_frame[3] < DEGENERACY_GAP,
        "bbox_diagonal": diag,
        "ref_frame": ref_frame,
        "sub_frame": sub_frame,
    }


def icp(src: np.ndarray, dst_tree: cKDTree, dst: np.ndarray, iters: int = 20, tol: float = 1e-9):
    """Point-to-point ICP refinement. Returns (rotation, translation, rmse)."""
    r_total, t_total = np.eye(3), np.zeros(3)
    cur = src
    prev = np.inf
    for _ in range(iters):
        d, idx = dst_tree.query(cur, workers=-1)
        target = dst[idx]
        # Kabsch
        cs, ct = cur.mean(0), target.mean(0)
        h = (cur - cs).T @ (target - ct)
        u, _, vt = np.linalg.svd(h)
        sgn = np.sign(np.linalg.det(vt.T @ u.T))
        r = vt.T @ np.diag([1, 1, sgn]) @ u.T
        t = ct - r @ cs
        cur = cur @ r.T + t
        r_total, t_total = r @ r_total, r @ t_total + t
        rmse = float(np.sqrt((d ** 2).mean()))
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    return r_total, t_total, prev
