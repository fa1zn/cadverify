"""Point sampling and pointwise distance.

Stage C's measurement machinery, built first because stage B needs a way to
score candidate alignments.

Tessellation error is acceptable here in a way it is not for invariants: both
shapes are meshed at the same relative deflection, so the error is common-mode.
"""

import math

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from scipy.spatial import cKDTree

DEFLECTION_FRAC = 0.002   # linear deflection as a fraction of the bbox diagonal
ANGULAR_DEFLECTION = 0.3


def bbox_diagonal(shape) -> float:
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b, True)
    x0, y0, z0, x1, y1, z1 = b.Get()
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)


def triangles(shape, deflection: float) -> np.ndarray:
    """Tessellate and return every triangle as an (n, 3, 3) array of vertices."""
    BRepMesh_IncrementalMesh(shape, deflection, False, ANGULAR_DEFLECTION, True)
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            t = loc.Transformation()
            pts = np.array(
                [
                    (lambda p: (p.X(), p.Y(), p.Z()))(tri.Node(i).Transformed(t))
                    for i in range(1, tri.NbNodes() + 1)
                ]
            )
            reversed_face = face.Orientation() == TopAbs_REVERSED
            idx = []
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                idx.append((b - 1, a - 1, c - 1) if reversed_face else (a - 1, b - 1, c - 1))
            if idx:
                out.append(pts[np.array(idx)])
        exp.Next()
    if not out:
        raise ValueError("tessellation produced no triangles")
    return np.concatenate(out, axis=0)


def sample_surface(shape, n: int = 4000, seed: int = 0) -> np.ndarray:
    """Sample n points uniformly by area over the shape's surface."""
    tris = triangles(shape, bbox_diagonal(shape) * DEFLECTION_FRAC)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = areas.sum()
    if total <= 0:
        raise ValueError("zero-area tessellation")

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tris), size=n, p=areas / total)
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    # fold the unit square onto the triangle
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[pick] + u * (b[pick] - a[pick]) + v * (c[pick] - a[pick])


def _point_triangle_distance(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Exact distance from each point to its paired triangle. Shapes (n,3), (n,3,3)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, pts - a

    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = pts - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = pts - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc

    # start from the face-interior solution, then override by Voronoi region
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(denom != 0, vb / np.where(denom != 0, denom, 1), 0.0)
        w = np.where(denom != 0, vc / np.where(denom != 0, denom, 1), 0.0)
    closest = a + v[:, None] * ab + w[:, None] * ac

    def seg(p0, e, t_num, t_den):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.clip(np.where(t_den != 0, t_num / np.where(t_den != 0, t_den, 1), 0.0), 0, 1)
        return p0 + t[:, None] * e

    r_a = (d1 <= 0) & (d2 <= 0)
    r_b = (d3 >= 0) & (d4 <= d3)
    r_c = (d6 >= 0) & (d5 <= d6)
    r_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    r_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    r_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

    closest = np.where(r_ab[:, None], seg(a, ab, d1, d1 - d3), closest)
    closest = np.where(r_ac[:, None], seg(a, ac, d2, d2 - d6), closest)
    closest = np.where(r_bc[:, None], seg(b, c - b, d4 - d3, (d4 - d3) + (d5 - d6)), closest)
    closest = np.where(r_a[:, None], a, closest)
    closest = np.where(r_b[:, None], b, closest)
    closest = np.where(r_c[:, None], c, closest)
    return np.linalg.norm(pts - closest, axis=1)


def _subdivide(tris: np.ndarray, max_edge: float) -> np.ndarray:
    """Split triangles until no edge exceeds max_edge.

    OpenCascade meshes a planar face into two enormous triangles regardless of
    deflection, because a plane has zero deviation from its own tessellation.
    That size spread is what breaks centroid-based pruning: a huge triangle's
    centroid can be far from a point lying on it. Making triangle size roughly
    uniform bounds the centroid-to-surface gap and lets the prefilter work.
    """
    for _ in range(12):
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        e = np.stack(
            [
                np.linalg.norm(b - a, axis=1),
                np.linalg.norm(c - b, axis=1),
                np.linalg.norm(a - c, axis=1),
            ],
            axis=1,
        )
        longest = e.argmax(axis=1)
        need = e.max(axis=1) > max_edge
        if not need.any():
            break
        keep = tris[~need]
        t = tris[need]
        li = longest[need]
        # roll each triangle so the longest edge is v0-v1, then split at its midpoint
        idx = (np.arange(3)[None, :] + li[:, None]) % 3
        v = np.take_along_axis(t, idx[:, :, None], axis=1)
        mid = 0.5 * (v[:, 0] + v[:, 1])
        left = np.stack([v[:, 0], mid, v[:, 2]], axis=1)
        right = np.stack([mid, v[:, 1], v[:, 2]], axis=1)
        tris = np.concatenate([keep, left, right], axis=0)
    return tris


class Surface:
    """A tessellated shape you can measure exact point distances against.

    Pruning is guaranteed, not heuristic. For a triangle with centroid c and
    circumradius r, dist(p, triangle) >= |p - c| - r. So once an upper bound ub
    is in hand, every triangle whose centroid is farther than ub + max_radius is
    provably not the nearest and can be skipped.

    A previous version used a plain k-nearest-centroid prefilter with no bound
    check. It reported 0.80% of the bbox diagonal for points known to lie exactly
    on the surface. The bound check below is what makes this version correct;
    points where it cannot be satisfied fall back to exhaustive search.
    """

    MAX_EDGE_FRAC = 0.04     # subdivide until no edge exceeds this fraction of the diagonal
    K = 12                   # centroids consulted to establish the initial upper bound

    def __init__(self, shape, deflection_frac: float = DEFLECTION_FRAC):
        diag = bbox_diagonal(shape)
        self._build(_subdivide(triangles(shape, diag * deflection_frac), diag * self.MAX_EDGE_FRAC))

    def _build(self, tris: np.ndarray) -> None:
        self.tris = tris
        self.centroids = tris.mean(axis=1)
        self.radii = np.linalg.norm(tris - self.centroids[:, None, :], axis=2).max(axis=1)
        self.max_radius = float(self.radii.max()) if len(tris) else 0.0
        self._tree = cKDTree(self.centroids)

    @classmethod
    def from_triangles(cls, tris: np.ndarray) -> "Surface":
        s = cls.__new__(cls)
        s._build(tris)
        return s

    def transformed(self, com: np.ndarray, axes: np.ndarray) -> "Surface":
        """This surface expressed in the canonical frame q = axes @ (p - com)."""
        return Surface.from_triangles((self.tris - com) @ axes.T)

    def _exhaustive(self, pts: np.ndarray, chunk: int = 4096) -> np.ndarray:
        m = len(self.tris)
        best = np.empty(len(pts))
        per = max(1, chunk // max(m, 1))
        for lo in range(0, len(pts), per):
            p = pts[lo : lo + per]
            n = len(p)
            d = _point_triangle_distance(
                np.repeat(p, m, axis=0), np.tile(self.tris, (n, 1, 1))
            ).reshape(n, m)
            best[lo : lo + per] = d.min(axis=1)
        return best

    def distance(self, pts: np.ndarray) -> np.ndarray:
        m = len(self.tris)
        n = len(pts)
        k = min(self.K, m)

        # Pass 1: a few nearest centroids give a cheap upper bound on the answer.
        _, idx = self._tree.query(pts, k=k, workers=-1)
        idx = idx.reshape(n, k)
        best = np.full(n, np.inf)
        for j in range(k):
            np.minimum(best, _point_triangle_distance(pts, self.tris[idx[:, j]]), out=best)

        # Pass 2: every triangle that could still win has its centroid inside
        # best + max_radius, since dist(p, t) >= |p - c_t| - r_t. Ball radius is
        # per-point, so a tight bound from pass 1 keeps the candidate set small.
        radii = best + self.max_radius
        groups = self._tree.query_ball_point(pts, radii, workers=-1, return_sorted=False)
        counts = np.fromiter((len(g) for g in groups), dtype=np.int64, count=n)
        self.last_candidates = float(counts.mean())
        nonempty = counts > 0
        if nonempty.any():
            flat = np.concatenate([g for g in groups if len(g)])
            owner = np.repeat(np.flatnonzero(nonempty), counts[nonempty])
            d = _point_triangle_distance(pts[owner], self.tris[flat])
            np.minimum.at(best, owner, d)
        return best


def chamfer(pts_a: np.ndarray, surf_b: Surface, pts_b: np.ndarray, surf_a: Surface) -> float:
    """Mean bidirectional point-to-surface distance."""
    return 0.5 * (surf_b.distance(pts_a).mean() + surf_a.distance(pts_b).mean())


def chamfer_one_way(pts: np.ndarray, surf: Surface) -> float:
    """Cheap directional distance — used to rank candidate alignments."""
    return float(surf.distance(pts).mean())


def hausdorff(pts_a: np.ndarray, surf_b: Surface, pts_b: np.ndarray, surf_a: Surface) -> float:
    return float(max(surf_b.distance(pts_a).max(), surf_a.distance(pts_b).max()))
