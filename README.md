# cadverify

A pose-invariant verifier for 2D-drawing-to-3D-CAD generation, and the evidence that
the shipped one is wrong in both directions.

The corpus is 150 approved samples from a `generate-3d-from-2d` track: an engineering
drawing in, a STEP solid out, with a reference solid and an answer key giving target
volume, bounding box, mass and centre of mass at 3% tolerance.

## Write-ups

Four pages, in the order they were written. The first two cover the verifier; the
second two cover the selection experiment built on it.

| | |
|---|---|
| [Pose-Invariant CAD Verifier](https://claude.ai/code/artifact/171a2761-127c-42b3-a80a-2c788d030e84) | Design spec — the frame-dependent/invariant split, the four-stage pipeline, the design decisions and why each one is there |
| [First Corpus Run](https://claude.ai/code/artifact/e4604fcb-8f44-4073-887b-b1df013c2180) | Measured results across all 150 pairs, including the two claims that were retracted mid-run and what replaced them |
| [The Goodhart Curve](https://claude.ai/code/artifact/e29869b9-4ee2-4f35-9eed-ccf1962aa58d) | Experiment spec, preserved as written beforehand, plus a section recording where the run diverged from it — a failed kill check, a 2× cost overrun, and a partial model arm |
| [Reward Selection Curve](https://claude.ai/code/artifact/0d9b6184-8c5f-46ba-9fec-8f26f90a2cb3) | The measured curve, both arms, with the pool composition that explains its shape |

These are private by default. Sharing one requires opening it and using its share
menu — the links will 404 for anyone else until then.

## Three findings

**1. The reference part fails its own answer key.**

Scoring each ground-truth STEP against the key that ships beside it:

| | n=150 | |
|---|---|---|
| volume within 3% | 150 / 150 | 100% |
| bounding box within 3% | 36 / 150 | **24%** |
| bounding box, best axis permutation | 144 / 150 | 96% |

The key records the bounding box in a different axis order than the file it ships
with, so the literally correct answer is marked wrong 76% of the time. Permuting the
three dimensions recovers 96%. Every other finding here follows from this one
mismatch. It reproduces in five lines and is asserted in
`tests/test_invariants.py::test_ground_truth_fails_its_own_bbox_gate`.

**2. The reward is exploitable by a policy that never reads the drawing.**

`cadverify/exploit.py` builds a rectangular block matching the target bounding box
with a centred pocket sized so the remaining volume matches exactly. Twenty-four
lines, no geometry understanding, no input.

| shipped-metric pass rate | volume | bbox | mass | all three |
|---|---|---|---|---|
| real submissions | 100.0% | 14.0% | 99.3% | **13.3%** |
| pocketed block | 100.0% | 100.0% | 100.0% | **100.0%** |

Under the corrected verifier the same block sits at 3.40% median shape distance
against 0.0115% for real submissions — a factor of 296.

**3. Optimising the shipped reward makes the selected part worse.**

Best-of-n selection over a fixed candidate pool, varying only which reward does the
selecting. Shape error is % of bounding-box diagonal, lower is better.

| n | select by shipped | select by corrected | exploit wins |
|---|---|---|---|
| 1 | 2.230% | 2.242% | 19.8% |
| 2 | 1.785% | 0.423% | 31.2% |
| 5 | 2.949% | 0.021% | 61.0% |
| 10 | **4.183%** | **0.000%** | 93.8% |

At `n=1` both arms are identical by construction — one candidate is selected
regardless of the reward — and they agree to within noise. That is the harness
sanity check. From `n=2` the confidence intervals separate and never re-cross.

## The mental model

Every measurement is taken in a coordinate frame, and a submission is the same shape
in a different pose than the reference. So measurements split in two:

- **Frame-dependent** — bounding box as an ordered triple, centre-of-mass
  coordinates, vertex positions. Meaningless until both shapes are aligned.
- **Frame-invariant** — volume, surface area, inertia eigenvalues, topology counts.
  Safe to compare directly.

**Rule one:** never compare a frame-dependent quantity without aligning first. That
erases the 86% false-reject rate.

**Rule two, the one people miss:** invariants alone are necessary and not sufficient.
Volume is a single scalar and many shapes share it, so an invariant-only grader is
immune to pose and trivially exploitable by shape. You need invariants to admit the
right parts and alignment plus a pointwise distance to reject the wrong ones.

An axis-aligned bounding box is *not* rescued by sorting its three numbers. Only 90°
axis-aligned rotations permute a box; a general rotation changes its size. An earlier
pass of this analysis got that wrong and reported three false accepts that were
actually correct parts — the correlation between "fails bbox under best permutation"
and real shape distance is 0.076.

## Pipeline

| Stage | What it does |
|---|---|
| **A** `invariants.py` | Volume, centre of mass, inertia tensor and surface area straight off the B-rep via `BRepGProp` — exact, no tessellation. Plus topology counts and OpenCascade's symmetry flags. |
| **B** `align.py` | Translate to centre of mass, rotate onto principal axes, search the 24 proper signed axis permutations, escalate to a 72-step sweep about a degenerate axis when two moments are within 2%, refine with ICP, keep the best. |
| **C** `sampling.py` | Tessellate, sample points uniformly by area, measure exact point-to-surface distance with guaranteed pruning. |
| **D** | Two gates: invariants within tolerance, and normalised shape distance under threshold. Every metric returns a number, never a bare verdict. |

## Install and run

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Point the tests at the corpus and run them:

```bash
CADVERIFY_CORPUS=~/Downloads/collection-selection ./.venv/bin/python -m pytest
```

16 tests, about 1m45s. They skip cleanly if the corpus is absent.

Score a whole corpus:

```bash
./.venv/bin/python scripts/run_corpus.py
```

150 pairs in ~150 seconds.

## What is validated, and how

The verifier is tested in both directions. A grader checked only against things that
should pass has an unmeasured false-accept rate, which is how the shipped one ended
up here.

| Test | Asserts |
|---|---|
| `test_point_triangle_matches_brute_force` | Analytic distance never exceeds a 40-step sampling of the triangle; degenerate and zero-area triangles handled |
| `test_points_on_surface_measure_zero` | Points sampled from a surface measure exactly 0 against it — not "small" |
| `test_fast_path_equals_exhaustive` | Pruned search is bit-identical to exhaustive |
| `test_rigid_motion_is_actually_applied` | Guards the invariance test from passing vacuously |
| `test_invariants_survive_rigid_motion` | Volume, area, moments, topology stable to 1e-11 |
| `test_agrees_with_shipped_answer_key` | Every disagreement falls inside the key's own 4-sig-fig rounding |
| `test_recovers_a_known_pose` | Positive control |
| `test_different_parts_do_not_align` | Negative control |
| `test_refinement_helps_and_is_guarded` | ICP helps on 5/20 pairs and **diverges on 14** — the `min()` guard is load-bearing, not a safety net |
| `test_exploit_beats_the_shipped_gate_and_fails_the_grader` | Degenerate-policy probe |

## Performance

The first working version ran at 37.6 s per pair, 94 minutes for a corpus pass. It is
now 1.01 s per pair, 151 seconds, with **zero** pairs changing by more than 1e-9.

The hot loop was an exhaustive point-to-triangle search. It is now bounded:
subdivide until triangle size is roughly uniform, take an upper bound from the
nearest few centroids, then consult only triangles whose centroid falls inside
`bound + max_radius`. Since `dist(p, t) >= |p - c| - r`, nothing outside that ball
can win, so the pruning is exact.

A plain k-nearest-centroid prefilter was tried first and was silently wrong —
OpenCascade meshes a planar face into two enormous triangles, so a point near a big
triangle's edge is nearer to dozens of small triangles' centroids than to its own. At
k=24 it reported 0.80% of the bbox diagonal for points known to lie exactly on the
surface. That is why `test_fast_path_equals_exhaustive` exists.

## Limitations

- **The 0.5% correctness threshold is a judgement call**, not a derived value. The
  false-reject rate is insensitive to it (86–87% at every threshold from 0.1% to 1%);
  the false-accept count is not.
- **Distances are upper bounds.** ICP finds a local optimum, and the guard means the
  reported figure is the better of two candidates, not a proven global minimum.
- **The curve's candidate pool is synthetic.** Candidates are geometric perturbations,
  so it measures what the reward *selects for*, not what a policy produces. A partial
  model-generated run (99 candidates from Claude Opus 5 over 7 usable tasks)
  reproduces the direction — shipped 2.306% → 4.562%, corrected 2.341% → 0.799%,
  exploit win rate 7.1% → 94.5% — with wide intervals. It stopped early on cost.
- **The corpus is one track and one difficulty.** 131 easy, 19 medium, no hard. Nothing
  here generalises past `generate-3d-from-2d` without re-measurement.
- **Topology is reported, never gated on.** 148 of 150 submissions have a different
  face count from their reference; CAD reaches the same solid through different
  feature trees.

## Layout

```
cadverify/     invariants, sampling, align, exploit
tests/         the validation suite above
scripts/       corpus runs and the selection-curve experiments
results/       measured outputs as JSON
```
