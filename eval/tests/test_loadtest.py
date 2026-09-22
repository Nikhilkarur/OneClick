"""The pure parts of loadtest.py: hashing, summaries and target verdicts. No engine, no network."""

import loadtest


def test_siis_hash_is_stable_and_order_independent():
    assert loadtest.siis_hash({"a": 1, "b": 2}) == loadtest.siis_hash({"b": 2, "a": 1})
    assert loadtest.siis_hash({"a": 1}) != loadtest.siis_hash({"a": 2})
    assert loadtest.siis_hash(None) is None


def test_summarise_flags_small_samples():
    s = loadtest.summarise([1.0, 2.0, 3.0], hits=2, n=3)
    assert s["hit_rate"] == round(2 / 3, 3)
    assert s["p50_ms"] == 2.0
    assert not s["enough_samples"]
    assert loadtest.summarise([], 0, 0)["hit_rate"] is None


def test_verdicts_against_the_spec_targets():
    section = {
        "repeat": {"p95_ms": 0.3},
        "paraphrase": {"p95_ms": 12.0, "hit_rate": 0.805},
        "near_miss": {"false_hit_rate": 0.3},
    }
    v = loadtest.verdicts(section)
    assert v["repeat_p95"] and v["paraphrase_p95"] and v["paraphrase_hit"]
    assert v["false_hits"] is False
    assert v["cold_p95"] is None  # not measured is not a pass
