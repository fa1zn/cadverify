"""Standard 3D-reconstruction metrics, for comparison against this verifier.

Chamfer distance, F-score at a threshold, volumetric IoU and normal consistency
are what the CAD-generation and 3D-reconstruction literature actually reports.
None of this is novel — the point of implementing them here is to answer two
questions the project could not otherwise answer:

  1. Do the standard metrics catch the degenerate policy? (If they do, this
     verifier is standard practice correctly applied, not a new method.)
  2. Does the *standard preprocessing* — centre at the origin, scale to a unit
     box, compare — reproduce the false-reject bug found in the shipped corpus?
     If it does, the bug is not unique to one vendor.

Each metric runs under two preprocessing regimes:

  center-scale   translate to centroid, scale to unit bounding box. What most
                 benchmarks do. No rotation handling.
  full-align     the same, plus principal-axis alignment with an octahedral
                 search and ICP refinement.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .align import align, canonical_points, principal_frame
from .sampling import Surface, bbox_diagonal, sample_surface, triangles

N_POINTS = 4000
FSCORE_TAU = 0.02        # fraction of the unit-normalised diagonal
VOXEL_RES = 28


def _centroid_scale(pts, diag):
    return (pts - pts.mean(axis=0)) / diag


def _tri_normals(tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln == 0, 1, ln)


def _sample_with_normals(shape, n, seed):
    tris = triangles(shape, bbox_diagonal(shape) * 0.002)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tris), size=n, p=area / area.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[pick] + u * (b[pick] - a[pick]) + v * (c[pick] - a[pick])
    return pts, _tri_normals(tris)[pick]


def _prep(ref_shape, sub_shape, mode, seed=0):
    """Return (ref_pts, ref_nrm, sub_pts, sub_nrm) in a common normalised frame."""
    diag = bbox_diagonal(ref_shape)
    rp, rn = _sample_with_normals(ref_shape, N_POINTS, seed)
    sp, sn = _sample_with_normals(sub_shape, N_POINTS, seed + 1)

    if mode == "center-scale":
        return _centroid_scale(rp, diag), rn, _centroid_scale(sp, diag), sn

    if mode == "full-align":
        fa, fb = principal_frame(ref_shape), principal_frame(sub_shape)
        res = align(ref_shape, sub_shape, n_points=N_POINTS, seed=seed)
        rot, trans = res["rotation"], res["translation"]
        rc = canonical_points(rp, fa[0], fa[1])
        sc = canonical_points(sp, fb[0], fb[1]) @ rot.T + trans
        return rc / diag, rn @ fa[1].T, sc / diag, (sn @ fb[1].T) @ rot.T

    raise ValueError(mode)


def chamfer_distance(rp, sp):
    """Mean bidirectional nearest-neighbour distance. Lower is better."""
    d1, _ = cKDTree(sp).query(rp, workers=-1)
    d2, _ = cKDTree(rp).query(sp, workers=-1)
    return float(0.5 * (d1.mean() + d2.mean()))


def f_score(rp, sp, tau=FSCORE_TAU):
    """Precision/recall of points within tau. Higher is better."""
    d_rs, _ = cKDTree(sp).query(rp, workers=-1)
    d_sr, _ = cKDTree(rp).query(sp, workers=-1)
    recall = float((d_rs < tau).mean())
    precision = float((d_sr < tau).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def normal_consistency(rp, rn, sp, sn):
    """Mean |cos| between normals at nearest-neighbour correspondences."""
    _, i = cKDTree(sp).query(rp, workers=-1)
    _, j = cKDTree(rp).query(sp, workers=-1)
    a = np.abs(np.einsum("ij,ij->i", rn, sn[i])).mean()
    b = np.abs(np.einsum("ij,ij->i", sn, rn[j])).mean()
    return float(0.5 * (a + b))


def voxel_iou(rp, sp, res=VOXEL_RES):
    """Occupancy IoU on a shared voxel grid over the union extent.

    Surface occupancy rather than solid occupancy — a point-in-solid test per
    voxel is exact but far slower, and surface IoU is what point-cloud pipelines
    typically compute.
    """
    allp = np.vstack([rp, sp])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = np.where(hi - lo == 0, 1, hi - lo)

    def occ(p):
        idx = np.clip(((p - lo) / span * (res - 1)).astype(int), 0, res - 1)
        g = np.zeros((res, res, res), bool)
        g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return g

    a, b = occ(rp), occ(sp)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


METRICS = ("chamfer", "fscore", "normal_consistency", "iou")


def compare(ref_shape, sub_shape, mode="center-scale", seed=0) -> dict:
    rp, rn, sp, sn = _prep(ref_shape, sub_shape, mode, seed)
    return {
        "chamfer": chamfer_distance(rp, sp),
        "fscore": f_score(rp, sp),
        "normal_consistency": normal_consistency(rp, rn, sp, sn),
        "iou": voxel_iou(rp, sp),
        "mode": mode,
    }
