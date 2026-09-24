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


def test_api_mode_records_cost_and_models(monkeypatch):
    from evalkit import client as client_mod
    from evalkit.client import CallResult
    from evalkit.sets import KitRow

    def fake_call(self, method, path, payload=None):
        if path == "/health":
            return CallResult(200, 1.0, body={"status": "ok"}, pure_json=True)
        meta = {"model": "ministral-14b-latest", "cost_usd": 0.0}
        return CallResult(
            200, 5.0, server_ms=4.0, cache_hit=False, body={"contexts": [], "meta": meta}, pure_json=True
        )

    monkeypatch.setattr(client_mod.ApiClient, "_call", fake_call)
    kit = [KitRow("r1", "q one", {"content": "a"}), KitRow("r2", "q two", {"content": "b"})]
    section = loadtest.run_api("http://x", kit, [], [], [])
    assert section["cold"]["mean_cost_usd"] == 0.0
    assert section["cold"]["models"] == {"ministral-14b-latest": 2}
    assert section["repeat"]["n"] == 4


def test_api_mode_times_cold_on_misses_and_hits_on_hits(monkeypatch):
    from evalkit import client as client_mod
    from evalkit.client import CallResult
    from evalkit.sets import KitRow

    seen: set = set()

    def fake_call(self, method, path, payload=None):
        if path == "/health":
            return CallResult(200, 1.0, body={"status": "ok"}, pure_json=True)
        # The second kit row shares an article and hits the first row's plan: 2 ms instead of 5000.
        hit = payload["query"] in seen or payload["query"] == "q two"
        seen.add(payload["query"])
        ms = 2.0 if hit else 5000.0
        body = {"contexts": [], "meta": {"model": None if hit else "ministral-14b-latest", "cost_usd": 0.0}}
        return CallResult(200, ms, server_ms=ms, cache_hit=hit, body=body, pure_json=True)

    monkeypatch.setattr(client_mod.ApiClient, "_call", fake_call)
    kit = [KitRow("r1", "q one", {"content": "a"}), KitRow("r2", "q two", {"content": "a"})]
    section = loadtest.run_api("http://x", kit, [], [], [])
    cold = section["cold"]
    assert cold["n"] == 2 and cold["timed_n"] == 1 and cold["p95_ms"] == 5000.0
    assert cold["models"] == {"ministral-14b-latest": 1}
    assert section["repeat"]["p95_ms"] == 2.0
    assert any("cold latency is timed on the misses" in n for n in section["notes"])


def test_api_mode_lists_leaked_near_misses(monkeypatch):
    from evalkit import client as client_mod
    from evalkit.client import CallResult
    from evalkit.sets import KitRow

    def fake_call(self, method, path, payload=None):
        if path == "/health":
            return CallResult(200, 1.0, body={"status": "ok"}, pure_json=True)
        hit = payload["query"] == "leak"
        return CallResult(
            200, 1.0, cache_hit=hit, cache_tier="semantic" if hit else None, body={}, pure_json=True
        )

    monkeypatch.setattr(client_mod.ApiClient, "_call", fake_call)
    kit = [KitRow("r1", "q", {"content": "a"})]
    near = [
        {"id": "nm_1", "row_id": "r1", "query": "leak", "differs_in": "intent"},
        {"id": "nm_2", "row_id": "r1", "query": "fine", "differs_in": "symptom"},
    ]
    nm = loadtest.run_api("http://x", kit, [], [], near)["near_miss"]
    assert nm["leaked"] == [{"id": "nm_1", "query": "leak", "differs_in": "intent", "tier": "semantic"}]
    assert nm["false_hit_rate"] == 0.5
