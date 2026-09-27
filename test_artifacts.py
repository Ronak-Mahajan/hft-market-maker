"""Tests for scripts/check_artifacts.py, the CI comparison of regenerated
artifacts against the committed copies."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import evaluate
from market_maker import Config

_spec = importlib.util.spec_from_file_location(
    "check_artifacts", Path(__file__).parent / "scripts" / "check_artifacts.py")
check_artifacts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_artifacts)

SUMMARY = {"label": "final_pnl", "n": 500, "mean": -514.8917704099242,
           "ci95": [-1000.25, -29.5], "win_rate": 0.43, "significant": True}


def _errors(committed, regenerated):
    return check_artifacts.compare_json(committed, regenerated,
                                        check_artifacts.Comparison()).errors


def test_json_floats_agree_to_rtol_1e9():
    assert _errors(SUMMARY, copy.deepcopy(SUMMARY)) == []
    near = copy.deepcopy(SUMMARY)
    near["mean"] *= 1 + 1e-12
    near["ci95"][0] *= 1 - 1e-11
    assert _errors(SUMMARY, near) == []
    far = copy.deepcopy(SUMMARY)
    far["ci95"][1] *= 1 + 1e-8
    assert [e.split(":")[0] for e in _errors(SUMMARY, far)] == ["$.ci95[1]"]


def test_json_discrete_fields_are_compared_exactly():
    for key, value in (("win_rate", 0.43 + 1e-15), ("significant", False),
                       ("n", 499), ("label", "t_stat"), ("n", 500.0)):
        changed = dict(SUMMARY, **{key: value})
        assert len(_errors(SUMMARY, changed)) == 1, key
    assert _errors(SUMMARY, {k: v for k, v in SUMMARY.items() if k != "n"})
    assert _errors({"x": [1.0, 2.0]}, {"x": [1.0]})


def test_csv_counts_exact_and_floats_to_rtol(tmp_path):
    path = tmp_path / "seeds.csv"
    evaluate.write_seeds_csv(str(path), evaluate.evaluate(Config(n_ticks=300), range(2)))
    committed = path.read_text()
    header, first, *rest = committed.splitlines()
    cols = header.split(",")
    assert check_artifacts.EXACT_CSV_COLUMNS <= set(cols)

    def with_first_row(field, value):
        row = first.split(",")
        row[cols.index(field)] = value
        return "\n".join([header, ",".join(row), *rest]) + "\n"

    def errors(text):
        return check_artifacts.compare_csv(committed, text,
                                           check_artifacts.Comparison()).errors

    assert errors(committed) == []
    sigma = float(first.split(",")[cols.index("inventory_sigma")])
    assert errors(with_first_row("inventory_sigma", repr(sigma * (1 + 1e-12)))) == []
    assert errors(with_first_row("inventory_sigma", repr(sigma * (1 + 1e-8))))
    fills = float(first.split(",")[cols.index("fills")])
    assert errors(with_first_row("fills", repr(fills + 1.0)))
    assert errors("\n".join([header, first]) + "\n")
