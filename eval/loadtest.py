"""Latency and cache efficacy per execution path (metrics.md sections 3 and 4). N >= 30 per path.

Two modes:

    cache  In-process, against the engine's two-tier cache. Warms it with the 20 kit queries
           only, then times the lookup for
             repeat      each kit query again (Tier 0)
             paraphrase  the 200 held-out paraphrases (Tier 1; never used to warm)
             near miss   the 60 near misses, same article, must NOT hit
           The live engine also stores 8-10 LLM variations per solved query, which this mode
           cannot, so its paraphrase hit rate is a lower bound. Times exclude HTTP.

    api    Against a running API. Cold pass first (kit + unseen, so the SIIS cache must start
           empty), then repeat, paraphrase and near-miss passes. Reports the server's
           X-Latency-Ms and the client-side time.

Usage (from the repo root):
    python eval/loadtest.py --mode cache                 # needs data/build (see evalkit/engine.py)
    python eval/loadtest.py --mode cache --sweep         # also try several similarity thresholds
    python eval/loadtest.py --mode api --api http://localhost:8000
Writes eval/results/loadtest.json (the other mode's section is kept).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evalkit.paths import RESULTS_DIR
from evalkit.sets import KitRow, load_kit, load_set
from evalkit.stats import norm_query, percentile, rate

OUT_PATH = RESULTS_DIR / "loadtest.json"
MIN_N = 30
TARGET_P95_MS = {"repeat": 300.0, "paraphrase": 300.0, "cold": 8000.0}
TARGET_PARAPHRASE_HIT = 0.80
TARGET_FALSE_HIT_MAX = 0.02


def siis_hash(siis: dict | str | None) -> str | None:
    if siis is None:
        return None
    return hashlib.sha256(json.dumps(siis, sort_keys=True).encode()).hexdigest()[:16]


def summarise(latencies: list[float], hits: int, n: int, **extra) -> dict:
    return {
        "n": n,
        "hits": hits,
        "hit_rate": None if not n else round(hits / n, 3),
        "p50_ms": None if not latencies else round(percentile(latencies, 50), 2),
        "p95_ms": None if not latencies else round(percentile(latencies, 95), 2),
        "enough_samples": n >= MIN_N,
        **extra,
    }


# ---------------------------------------------------------------------------------------------- cache


def run_cache(
    kit: list[KitRow], paraphrases: list[dict], near_misses: list[dict], threshold: float | None
) -> dict:
    from evalkit import engine

    if threshold is not None:
        engine.settings.cache_sim_threshold = threshold
    tau = engine.settings.cache_sim_threshold
    cache = engine.cache
    cache.clear()

    hashes = {row.row_id: siis_hash(row.siis) for row in kit}
    for row in kit:
        query = norm_query(row.query)
        cache.put(
            engine.CacheEntry(
                key=cache.make_key(query, hashes[row.row_id]),
                siis_hash=hashes[row.row_id],
                slots=engine.extract_slots(query),
                plan={"row_id": row.row_id},
                query_texts=[query],
                created_at=time.time(),
            )
        )

    def lookup(text: str, row_id: str):
        query = norm_query(text)
        start = time.perf_counter()
        hit = cache.lookup(query, engine.extract_slots(query), hashes[row_id])
        return hit, (time.perf_counter() - start) * 1000

    lat, hits, n = [], 0, 0
    for _ in range(2):
        for row in kit:
            hit, ms = lookup(row.query, row.row_id)
            lat.append(ms)
            n += 1
            hits += hit is not None and hit.tier == "exact"
    repeat = summarise(lat, hits, n)

    lat, hits, wrong, misses = [], 0, 0, []
    for p in paraphrases:
        hit, ms = lookup(p["query"], p["row_id"])
        lat.append(ms)
        if hit is None:
            misses.append({"id": p["id"], "register": p.get("register"), "query": p["query"]})
            continue
        hits += 1
        wrong += hit.plan.get("row_id") != p["row_id"]
    by_register: dict[str, list[int]] = {}
    for p in paraphrases:
        by_register.setdefault(p.get("register") or "?", [0, 0])[1] += 1
    for p in paraphrases:
        if not any(m["id"] == p["id"] for m in misses):
            by_register[p.get("register") or "?"][0] += 1
    paraphrase = summarise(
        lat,
        hits,
        len(paraphrases),
        wrong_plan=wrong,
        by_register={k: round(h / t, 3) for k, (h, t) in sorted(by_register.items())},
        missed=misses,
    )

    from app.cache import semantic

    lat, false_hits, would_hit, leaked = [], 0, 0, []
    for m in near_misses:
        hit, ms = lookup(m["query"], m["row_id"])
        lat.append(ms)
        _, similarity = semantic.best_match(norm_query(m["query"]))
        would_hit += similarity >= tau
        if hit is not None:
            false_hits += 1
            leaked.append({"id": m["id"], "differs_in": m.get("differs_in"), "query": m["query"]})
    near_miss = summarise(
        lat,
        false_hits,
        len(near_misses),
        false_hit_rate=rate(false_hits, len(near_misses)),
        above_threshold_without_guard=would_hit,
        leaked=leaked,
    )
    cache.clear()
    return {
        "source": "in-process cache lookup (no HTTP, no final scrub)",
        "warmed_with": f"{len(kit)} kit queries, original phrasing only (no LLM variations)",
        "threshold": tau,
        "repeat": repeat,
        "paraphrase": paraphrase,
        "near_miss": near_miss,
    }


# ------------------------------------------------------------------------------------------------ api


def run_api(
    base_url: str, kit: list[KitRow], unseen: list[dict], paraphrases: list[dict], near_misses: list[dict]
):
    from evalkit.client import ApiClient

    client = ApiClient(base_url, timeout_s=60.0)
    siis = {row.row_id: row.siis for row in kit}
    notes = []
    health = client.health()
    if health.status_code != 200:
        raise SystemExit(f"{base_url}/health returned {health.status_code}: {health.error or health.body}")

    def timed(results, hit_expected: bool | None) -> dict:
        server = [r.server_ms for r in results if r.server_ms is not None]
        client_ms = [r.client_ms for r in results]
        hits = sum(bool(r.cache_hit) for r in results)
        ok = sum(r.ok for r in results)
        out = summarise(server or client_ms, hits, len(results), ok_200=ok)
        out["client_p50_ms"] = round(percentile(client_ms, 50) or 0.0, 2)
        out["client_p95_ms"] = round(percentile(client_ms, 95) or 0.0, 2)
        out["latency_source"] = "X-Latency-Ms" if server else "client"
        if hit_expected is False:
            out["false_hit_rate"] = rate(hits, len(results))
        return out

    cold = [client.troubleshoot(row.query, row.siis) for row in kit]
    cold += [client.troubleshoot(u["query"], u.get("siis_response")) for u in unseen]
    if any(r.cache_hit for r in cold):
        notes.append(
            "some cold-pass calls reported a cache hit: the API did not start with an empty SIIS cache"
        )
    repeat = [client.troubleshoot(row.query, row.siis) for _ in range(2) for row in kit]
    para = [client.troubleshoot(p["query"], siis[p["row_id"]]) for p in paraphrases]
    near = [client.troubleshoot(m["query"], siis[m["row_id"]]) for m in near_misses]
    client.close()
    return {
        "source": f"HTTP against {base_url}",
        "cold": timed(cold, None),
        "repeat": timed(repeat, True),
        "paraphrase": timed(para, True),
        "near_miss": timed(near, False),
        "notes": notes,
    }


# ----------------------------------------------------------------------------------------------- main


def verdicts(section: dict) -> dict:
    out = {}
    for path, target in TARGET_P95_MS.items():
        p95 = (section.get(path) or {}).get("p95_ms")
        out[f"{path}_p95"] = None if p95 is None else p95 <= target
    hit = (section.get("paraphrase") or {}).get("hit_rate")
    out["paraphrase_hit"] = None if hit is None else hit >= TARGET_PARAPHRASE_HIT
    false = (section.get("near_miss") or {}).get("false_hit_rate")
    out["false_hits"] = None if false is None else false <= TARGET_FALSE_HIT_MAX
    return out


def print_section(name: str, section: dict) -> None:
    print(f"\n[{name}] {section['source']}")
    for path in ("cold", "repeat", "paraphrase", "near_miss"):
        s = section.get(path)
        if not s:
            continue
        hit = "-" if s["hit_rate"] is None else f"{s['hit_rate']:.0%}"
        print(f"  {path:<11} n={s['n']:<4} hit {hit:>5}   p50 {s['p50_ms']} ms   p95 {s['p95_ms']} ms")
    nm = section.get("near_miss") or {}
    if "above_threshold_without_guard" in nm:
        print(
            f"  slot guard: {nm['above_threshold_without_guard']}/{nm['n']} near misses cleared the similarity "
            f"threshold, {nm['hits']} got through"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["cache", "api"], required=True)
    parser.add_argument("--api", help="base URL for --mode api")
    parser.add_argument("--sweep", action="store_true", help="cache mode: also try thresholds 0.70-0.90")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    kit = load_kit()
    paraphrases, near_misses = load_set("paraphrases"), load_set("near_miss")
    report = json.loads(args.out.read_text()) if args.out.exists() else {}

    if args.mode == "cache":
        section = run_cache(kit, paraphrases, near_misses, threshold=None)
        if args.sweep:
            section["sweep"] = []
            for tau in (0.70, 0.75, 0.80, 0.85, 0.90):
                s = run_cache(kit, paraphrases, near_misses, threshold=tau)
                section["sweep"].append(
                    {
                        "threshold": tau,
                        "paraphrase_hit": s["paraphrase"]["hit_rate"],
                        "wrong_plan": s["paraphrase"]["wrong_plan"],
                        "false_hit_rate": s["near_miss"]["false_hit_rate"],
                    }
                )
    else:
        if not args.api:
            parser.error("--mode api needs --api URL")
        section = run_api(args.api, kit, load_set("unseen"), paraphrases, near_misses)

    section["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    section["verdicts"] = verdicts(section)
    report[args.mode] = section
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print_section(args.mode, section)
    for row in section.get("sweep", []):
        print(
            f"  tau {row['threshold']:.2f}: paraphrase hit {row['paraphrase_hit']:.0%}, "
            f"wrong plan {row['wrong_plan']}, false hits {row['false_hit_rate']:.0%}"
        )
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
