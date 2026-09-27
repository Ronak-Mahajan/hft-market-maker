"""Compare freshly regenerated result artifacts with the committed copies.

The three generators write their artifacts into the repository root:

    python evaluate.py --seeds 500                 # results.json, results_seeds.csv
    python evaluate.py --seeds 100 --gamma-sweep   # results_sweep.json
    python audit.py --seeds 200                    # audit_results.json

Run this script after them. It reads each artifact from the working tree and
the committed version from git (HEAD unless --rev says otherwise), and exits
with status 1 if any of the four differs.

What counts as a difference:

- Discrete content must match exactly: JSON structure, keys, list lengths,
  strings, booleans and integers; the win_rate and profitable_share fractions,
  which count seeds; and the seed, arm, fills, max_abs_inventory,
  quote_crossings, floored_bids, blocked_fills and ticks_at_cap columns of
  results_seeds.csv.
- Every other float must agree to a relative tolerance of 1e-9. The
  writers round floats to 10 significant digits (evaluate.SIG_DIGITS),
  which absorbs the platform noise in the last bits of exp, log and long
  sums, so a correct regeneration normally matches byte for byte. A value
  that falls within that noise of a rounding boundary comes out one unit
  different in the tenth digit, and the tolerance accepts it.

CI runs the generators and then this script on every push.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

JSON_ARTIFACTS = ("results.json", "results_sweep.json", "audit_results.json")
CSV_ARTIFACTS = ("results_seeds.csv",)
ARTIFACTS = JSON_ARTIFACTS + CSV_ARTIFACTS

RTOL = 1e-9

# JSON float fields that are fractions of a seed count: compared exactly.
EXACT_JSON_KEYS = frozenset({"win_rate", "profitable_share"})
# results_seeds.csv columns holding identifiers or counts: compared exactly,
# as text.
EXACT_CSV_COLUMNS = frozenset({"seed", "arm", "fills", "max_abs_inventory",
                               "quote_crossings", "floored_bids",
                               "blocked_fills", "ticks_at_cap"})

MAX_REPORTED = 20


class Comparison:
    """Accumulates mismatches and the largest relative float difference."""

    def __init__(self, rtol: float = RTOL):
        self.rtol = rtol
        self.errors: list[str] = []
        self.n_floats = 0
        self.max_rel = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def fail(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def floats(self, where: str, committed: float, regenerated: float) -> None:
        self.n_floats += 1
        if math.isnan(committed) or math.isnan(regenerated):
            if not (math.isnan(committed) and math.isnan(regenerated)):
                self.fail(where, f"committed {committed!r}, regenerated {regenerated!r}")
            return
        if math.isinf(committed) or math.isinf(regenerated):
            if committed != regenerated:
                self.fail(where, f"committed {committed!r}, regenerated {regenerated!r}")
            return
        diff = abs(committed - regenerated)
        scale = max(abs(committed), abs(regenerated))
        rel = diff / scale if scale > 0 else 0.0
        self.max_rel = max(self.max_rel, rel)
        if diff > self.rtol * scale:
            self.fail(where, f"committed {committed!r}, regenerated {regenerated!r} "
                             f"(relative difference {rel:.3g} > {self.rtol:g})")


def compare_json(committed, regenerated, cmp: Comparison,
                 where: str = "$", key: str | None = None) -> Comparison:
    """Recursive comparison of two parsed JSON documents."""
    if isinstance(committed, bool) or isinstance(regenerated, bool):
        if type(committed) is not type(regenerated) or committed != regenerated:
            cmp.fail(where, f"committed {committed!r}, regenerated {regenerated!r}")
    elif isinstance(committed, float) and isinstance(regenerated, float):
        if key in EXACT_JSON_KEYS:
            if committed != regenerated:
                cmp.fail(where, f"committed {committed!r}, regenerated "
                                f"{regenerated!r} (compared exactly)")
        else:
            cmp.floats(where, committed, regenerated)
    elif isinstance(committed, dict) and isinstance(regenerated, dict):
        missing = sorted(set(committed) - set(regenerated))
        extra = sorted(set(regenerated) - set(committed))
        if missing:
            cmp.fail(where, f"keys missing from the regenerated file: {missing}")
        if extra:
            cmp.fail(where, f"keys not in the committed file: {extra}")
        for k in committed:
            if k in regenerated:
                compare_json(committed[k], regenerated[k], cmp, f"{where}.{k}", k)
    elif isinstance(committed, list) and isinstance(regenerated, list):
        if len(committed) != len(regenerated):
            cmp.fail(where, f"committed length {len(committed)}, "
                            f"regenerated length {len(regenerated)}")
        for i, (c, r) in enumerate(zip(committed, regenerated)):
            compare_json(c, r, cmp, f"{where}[{i}]", key)
    elif type(committed) is not type(regenerated) or committed != regenerated:
        # int, str, None, or a type change such as int -> float
        cmp.fail(where, f"committed {committed!r}, regenerated {regenerated!r}")
    return cmp


def compare_csv(committed_text: str, regenerated_text: str,
                cmp: Comparison) -> Comparison:
    """Row-by-row comparison of two results_seeds.csv files."""
    committed = list(csv.reader(io.StringIO(committed_text)))
    regenerated = list(csv.reader(io.StringIO(regenerated_text)))
    if not committed or not regenerated:
        cmp.fail("csv", "empty file")
        return cmp
    header = committed[0]
    if regenerated[0] != header:
        cmp.fail("header", f"committed {header}, regenerated {regenerated[0]}")
        return cmp
    if len(committed) != len(regenerated):
        cmp.fail("csv", f"committed {len(committed) - 1} rows, "
                        f"regenerated {len(regenerated) - 1}")
    for line, (c_row, r_row) in enumerate(zip(committed[1:], regenerated[1:]), 2):
        if len(c_row) != len(header) or len(r_row) != len(header):
            cmp.fail(f"line {line}", "wrong number of fields")
            continue
        for col, c, r in zip(header, c_row, r_row):
            where = f"line {line} {col}"
            if col in EXACT_CSV_COLUMNS:
                if c != r:
                    cmp.fail(where, f"committed {c!r}, regenerated {r!r} "
                                    "(compared exactly)")
            else:
                try:
                    cmp.floats(where, float(c), float(r))
                except ValueError:
                    cmp.fail(where, f"not a number: committed {c!r}, regenerated {r!r}")
    return cmp


def committed_text(rev: str, name: str) -> str:
    out = subprocess.run(["git", "show", f"{rev}:{name}"], cwd=ROOT,
                         capture_output=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"cannot read {name} at {rev}: "
                         f"{out.stderr.decode(errors='replace').strip()}")
    return out.stdout.decode("utf-8")


def check(name: str, committed: str, regenerated: str) -> Comparison:
    cmp = Comparison()
    if name.endswith(".json"):
        return compare_json(json.loads(committed), json.loads(regenerated), cmp)
    return compare_csv(committed, regenerated, cmp)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rev", default="HEAD",
                   help="git revision holding the committed artifacts (default HEAD)")
    p.add_argument("--dir", default=str(ROOT),
                   help="directory holding the regenerated artifacts "
                        "(default: the repository root)")
    args = p.parse_args()

    failed = []
    for name in ARTIFACTS:
        path = Path(args.dir) / name
        if not path.is_file():
            print(f"{name}: FAIL, no regenerated file at {path}")
            failed.append(name)
            continue
        cmp = check(name, committed_text(args.rev, name),
                    path.read_text(encoding="utf-8"))
        if cmp.ok:
            print(f"{name}: ok ({cmp.n_floats} floats, largest relative "
                  f"difference {cmp.max_rel:.2g})")
            continue
        failed.append(name)
        print(f"{name}: FAIL, {len(cmp.errors)} difference(s) from {args.rev}")
        for e in cmp.errors[:MAX_REPORTED]:
            print(f"  {e}")
        if len(cmp.errors) > MAX_REPORTED:
            print(f"  ... and {len(cmp.errors) - MAX_REPORTED} more")

    if failed:
        print(f"\n{len(failed)} artifact(s) differ from {args.rev}: "
              f"{', '.join(failed)}. Regenerate them with the commands in this "
              "script's docstring and commit them with the change that moved them.")
        return 1
    print(f"\nall {len(ARTIFACTS)} artifacts match {args.rev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
