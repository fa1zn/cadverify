"""cadverify — corpus audit and scoring.

    cadverify audit <corpus>     check a delivery before it ships
    cadverify score <corpus>     score submissions against references
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .audit import NOTE, QUARANTINE, REJECT, audit, verdicts


def _audit(a):
    c, findings = audit(a.corpus, geometry=not a.no_geometry, limit=a.limit)
    v = verdicts(c, findings)
    counts = Counter(v.values())
    by_check = Counter((f.check, f.severity) for f in findings)

    print(f"corpus: {a.corpus}")
    print(f"samples: {len(c.samples)}\n")

    order = {REJECT: 0, QUARANTINE: 1, NOTE: 2}
    for sev in (REJECT, QUARANTINE, NOTE):
        group = [f for f in findings if f.severity == sev]
        if not group:
            continue
        print(f"{sev.upper()}  ({len(group)})")
        shown = Counter()
        for f in sorted(group, key=lambda f: f.check):
            shown[f.check] += 1
            if shown[f.check] <= a.per_check:
                sid = f.sample_id[:8] if f.sample_id else "—"
                print(f"  [{f.check:<13}] {sid:<10} {f.message}")
        for chk, n in shown.items():
            if n > a.per_check:
                print(f"  [{chk:<13}] ... and {n - a.per_check} more")
        print()

    corpus_level = [f for f in findings if f.sample_id is None and f.severity == REJECT]

    print("verdict")
    for k in ("accept", QUARANTINE, REJECT):
        n = counts.get(k, 0)
        pct = n / max(len(c.samples), 1) * 100
        print(f"  {k:<12}{n:>5}  {pct:>5.1f}%")
    if corpus_level:
        print(f"  {'corpus-wide':<12}{len(corpus_level):>5}  "
              f"{'issue' if len(corpus_level) == 1 else 'issues'} affecting every sample")

    if a.json:
        json.dump({"verdicts": v,
                   "findings": [f.__dict__ for f in findings],
                   "summary": {k: counts.get(k, 0) for k in
                               ("accept", QUARANTINE, REJECT)}},
                  open(a.json, "w"), indent=1)
        print(f"\n-> {a.json}")

    n_rej = counts.get(REJECT, 0)
    if n_rej or corpus_level:
        parts = []
        if n_rej:
            parts.append(f"{n_rej} sample{'' if n_rej == 1 else 's'} not usable as delivered")
        if corpus_level:
            parts.append(f"{len(corpus_level)} corpus-wide issue"
                         f"{'' if len(corpus_level) == 1 else 's'}")
        print("\n" + "; ".join(parts) + ".")
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cadverify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    au = sub.add_parser("audit", help="check a delivery before it ships")
    au.add_argument("corpus")
    au.add_argument("--no-geometry", action="store_true",
                    help="skip checks that need OpenCascade")
    au.add_argument("--limit", type=int, help="only check this many samples geometrically")
    au.add_argument("--per-check", type=int, default=5,
                    help="findings shown per check before collapsing (default 5)")
    au.add_argument("--json", help="write the full report here")
    au.set_defaults(func=_audit)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
