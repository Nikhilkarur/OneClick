"""Regenerates docs/metrics.md from eval outputs, in the Theme 2 spec's Appendix C layout.

Reads whatever exists in eval/results/ and never invents a number: a cell with no measurement
behind it says "not measured" and the reason.

    gates.json     gate_replica.py   section 1 (schema, rules, leaks, catalog validity)
    judge.json     judge.py          section 2 (step accuracy)
    ablation.json  ablation.py       sections 2 and 5 (deeplink relevance, mapping ablation)
    loadtest.json  loadtest.py       sections 3 and 4 (latency, cache hit rate, cost)

A response with empty contexts passes every format rule trivially, so section 1 is reported as
not measured until the engine returns non-empty plans.

Usage (from the repo root):
    python eval/report.py
    python eval/report.py --model "gemini-2.5-flash, fallback mistral-small" --env "2 vCPU / 4 GB / Ubuntu 22.04"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evalkit.paths import API_DIR, REPO_ROOT, RESULTS_DIR

OUT_PATH = REPO_ROOT / "docs" / "metrics.md"
NM = "not measured"

# Screens a gold step needs that the catalog has no entry for (data/gold tier "dummy"), and why
# a correct plan still carries bixby://dummy_positive or no link there. Written from the catalog
# audit in data/gold/README.md; the report adds the counts measured from the gold set.
CATALOG_GAPS = (
    "Software update, Safe mode, Dark mode, auto-rotate, font size, per-app storage and a true "
    "Factory data reset have no catalog entry, so those steps resolve to `bixby://dummy_positive` "
    "or stay manual. `DL-0022` is the *auto* factory reset, not Factory data reset."
)


def load(name: str, results_dir: Path) -> dict | None:
    path = results_dir / name
    return json.loads(path.read_text()) if path.exists() else None


def pct(value: float | None) -> str:
    return NM if value is None else f"{value * 100:.1f}%"


def ms(value: float | None) -> str:
    return NM if value is None else f"{value:.1f}"


def commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def embed_model() -> str:
    """The embedding model id as configured, read from config.py without importing the engine."""
    text = (API_DIR / "app" / "config.py").read_text()
    match = re.search(r'embed_model:\s*str\s*=\s*"([^"]+)"', text)
    return f"{match.group(1)} (ONNX via fastembed)" if match else NM


def local_env() -> str:
    ram = ""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
            ram = f" / {int(out.stdout) / 2**30:.0f} GB RAM"
        elif Path("/proc/meminfo").exists():
            kb = int(Path("/proc/meminfo").read_text().split()[1])
            ram = f" / {kb / 2**20:.0f} GB RAM"
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass
    return f"{os.cpu_count()} vCPU{ram} / {platform.system()} {platform.release()} (measurement machine)"


# ------------------------------------------------------------------------------------------ sections


def section1(gates: dict | None) -> list[str]:
    rows = [
        ("Schema-valid output lines", ">= 99%"),
        ("Rule compliance (Goal / Title / Description syntax)", ">= 95%"),
        ("Absolute URL leaks", "0"),
        ("Deeplink catalog validity (exact URI match)", "100%"),
        ("Auto actions carrying valid actionable deeplink", ">= 90%"),
    ]
    values = [NM] * len(rows)
    note = "No gate replica run found (`eval/results/gates.json`)."
    if gates:
        a1 = (gates.get("blocks", {}).get("A1") or {}).get("detail") or {}
        a2 = (gates.get("blocks", {}).get("A2") or {}).get("detail") or {}
        responses, non_empty = a1.get("responses", 0), a1.get("non_empty", 0)
        if non_empty:
            g4 = (gates["gates"].get("G4") or {}).get("value")
            g5 = (gates["gates"].get("G5") or {}).get("value")
            rules = a1.get("rule_pass_rate") or {}
            syntax = [rules[r] for r in ("goal", "title", "description") if r in rules]
            values = [
                pct(g4),
                pct(sum(syntax) / len(syntax) if syntax else None),
                NM if g5 is None else str(g5),
                pct(a2.get("catalog_validity")),
                pct(a2.get("auto_with_link")),
            ]
            note = f"{non_empty} of {responses} responses had non-empty plans (gate replica, `{gates.get('git_sha')}`)."
        else:
            note = (
                f"The gate replica checked {responses} responses and all had empty `contexts`: the engine's "
                "extraction stages are not implemented yet. Empty plans pass every format rule trivially, "
                "so these cells stay unmeasured rather than reading 100%."
            )
    out = [
        "## 1. Schema & Rule Compliance",
        "Evaluated on sample datasets and held-out validation scenarios.",
        "",
        "| Metric | Target | Measured Value |",
        "| :--- | :--- | :--- |",
    ]
    out += [f"| {name} | {target} | {value} |" for (name, target), value in zip(rows, values, strict=True)]
    return [*out, "", f"_{note}_", ""]


def ours(ablation: dict | None) -> dict | None:
    if not ablation:
        return None
    return next((v for v in ablation["variants"] if v["key"] == "screengraph" and v["measured"]), None)


def section2(judge: dict | None, ablation: dict | None) -> list[str]:
    step = NM if not judge else f"{judge['step_accuracy_mean']:.2f}"
    link = ours(ablation)
    rel = NM if not link else f"{link['all']['relevance_mean']:.2f}"
    out = [
        "## 2. Accuracy Benchmarks",
        "Evaluated against reference ground truth scenarios across Battery, Display, Camera, and Performance.",
        "",
        "| Evaluation Metric | Scale / Anchor | Score |",
        "| :--- | :--- | :--- |",
        f"| Step accuracy (completeness, correctness, ordering) | 0.0 - 3.0 | {step} |",
        f"| Deeplink relevance (exact target screen vs. parent menu) | 0.0 - 2.0 | {rel} |",
        "",
    ]
    if not judge:
        out.append("_Step accuracy: not measured yet — `eval/judge.py` needs the engine's extracted steps._")
    if link:
        a, by = link["all"], link["by_owner"]
        split = ", ".join(
            f"{owner} {g['relevance_mean']:.2f} (n={g['n']})" for owner, g in sorted(by.items()) if g.get("n")
        )
        out.append(
            f"_Deeplink relevance: the Screen Graph resolver on {a['n']} hand-labelled gold steps "
            f"(`data/gold/deeplink_gold.jsonl`), step by step rather than end to end. Precision@1 "
            f"{pct(a['p_at_1'])} on the {a['p_at_1_n']} steps with a real catalog answer. By labeller: {split}. "
            "The mapping lane tuned its thresholds on this file, so its own labels are the more optimistic half._"
        )
    return [*out, ""]


def _latency_rows(load: dict | None) -> tuple[list[tuple[str, str, str, str]], list[str]]:
    rows = [
        ("Cache hit - exact query match", "<= 300 ms", "repeat"),
        ("Cache hit - unseen semantic paraphrase", "<= 300 ms", "paraphrase"),
        ("Cold query - full pipeline extraction & mapping", "<= 8000 ms", "cold"),
    ]
    notes = []
    api = (load or {}).get("api")
    cache = (load or {}).get("cache")
    out = []
    for name, target, path in rows:
        src = api if api and api.get(path) else (cache if cache and cache.get(path) else None)
        s = (src or {}).get(path)
        if not s:
            out.append((name, target, NM, NM))
            continue
        out.append((name, target, ms(s["p50_ms"]), ms(s["p95_ms"])))
    if api:
        notes.append(
            f"Measured over HTTP ({api['source']}), server-side `X-Latency-Ms` where the API sends it."
        )
    elif cache:
        notes.append(
            "Cache rows are the in-process lookup (`eval/loadtest.py --mode cache`): Tier 0 / Tier 1 lookup "
            "time only, without HTTP or the final scrub. The cold path needs the engine and is not measured yet."
        )
    return out, notes


def section3(load: dict | None) -> list[str]:
    rows, notes = _latency_rows(load)
    sections = [s for s in ((load or {}).get("api"), (load or {}).get("cache")) if s]
    ns = []
    for s in sections[:1]:
        ns = [f"{p} n={s[p]['n']}" for p in ("repeat", "paraphrase", "cold") if s.get(p)]
    out = [
        "## 3. Latency Benchmarks (N >= 30 requests per path)",
        "",
        "| Execution Path | Target (P95) | P50 (ms) | P95 (ms) |",
        "| :--- | :--- | :--- | :--- |",
    ]
    out += [f"| {name} | {target} | {p50} | {p95} |" for name, target, p50, p95 in rows]
    out.append("")
    if ns:
        notes.append("Samples: " + ", ".join(ns) + ".")
    return [*out, *(f"_{n}_" for n in notes), ""]


def section4(load: dict | None) -> list[str]:
    src = (load or {}).get("api") or (load or {}).get("cache") or {}
    para = src.get("paraphrase")
    cold_cost = ((load or {}).get("api") or {}).get("cold", {}).get("mean_cost_usd")
    out = [
        "## 4. Operational Cost & Cache Efficacy",
        "",
        "| Metric Item | Target | Measured Value |",
        "| :--- | :--- | :--- |",
        f"| Cold query average inference cost | Tracked | {NM if cold_cost is None else f'${cold_cost:.4f}'} |",
        "| Cache hit inference cost | $0.00 | $0.00 (no LLM call on the hit path, by construction) |",
        f"| Semantic cache hit rate (on unseen paraphrases) | >= 80% | {pct(para and para['hit_rate'])} |",
        "| Cost derivation method | - | (prompt tokens + completion tokens) × rate |",
        "",
    ]
    if para:
        out.append(
            f"_Hit rate over {para['n']} held-out paraphrases (`eval/sets/paraphrases.jsonl`, never used to warm "
            f"the cache), {para.get('wrong_plan', 0)} served the wrong plan. "
            f"{src.get('warmed_with', '')}._"
        )
    nm = src.get("near_miss")
    if nm:
        out.append(
            f"_False hits on {nm['n']} near misses (same article, different problem): {pct(nm.get('false_hit_rate'))} "
            "(target <= 2%)._"
        )
    return [*out, ""]


def section5(ablation: dict | None) -> list[str]:
    out = [
        "## 5. Architectural Ablation Analysis",
        "",
        "| Architecture Variant | Step Accuracy | Latency (P95) | Cost / Query | Key Observations |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    if not ablation:
        names = (
            "Baseline: Full LLM Deeplink Mapping",
            "Variant A: Hybrid BM25 + Dense Embedding Retrieval",
            "Variant B: Pure Rules-Based Deeplink Mapping",
        )
        out += [f"| {n} | {NM} | {NM} | {NM} | {NM} |" for n in names]
        return [*out, ""]
    for v in ablation["variants"]:
        if not v["measured"]:
            out.append(f"| {v['name']} | {NM} | {NM} | {NM} | {NM}: {v['reason']} |")
            continue
        a = v["all"]
        obs = (
            f"deeplink relevance {a['relevance_mean']:.2f}/2, P@1 {pct(a['p_at_1'])}, right tier "
            f"{pct(a['tier_accuracy'])}, wrong or unsafe link on {pct(a['wrong_link_rate'])} of steps"
        )
        cost = f"${v['cost_per_step_usd']:.5f}" if v["cost_per_step_usd"] else "$0.00"
        out.append(
            f"| {v['name']} | held fixed (mapping-only) | {v['latency_ms']['p95']:.1f} ms / step | {cost} | {obs} |"
        )
    g = ablation["gold"]
    tiers = ", ".join(f"{n} {t}" for t, n in g["by_tier"].items())
    out += [
        "",
        (
            f"_Mapping-only ablation: every variant maps the same {g['n']} gold steps ({tiers}), so "
            "extraction and therefore step accuracy are held fixed; the variants differ in deeplink "
            "relevance (0-2 rubric in `eval/evalkit/relevance.py`), precision@1 and safety. "
            f"Commit `{ablation.get('commit')}`._"
        ),
        "",
    ]
    return out


def section6(ablation: dict | None, load: dict | None, gates: dict | None, judge: dict | None) -> list[str]:
    items = [f"**Catalog gaps.** {CATALOG_GAPS}"]
    link = ours(ablation)
    if ablation:
        g = ablation["gold"]["by_tier"]
        items.append(
            f"**Dummy rate by catalog gap.** {g.get('dummy', 0)} of {ablation['gold']['n']} gold steps need a "
            f"Settings screen the catalog lacks, and {g.get('manual', 0)} are physical steps that take no link."
        )
    if link:
        wrong_dummy = [m for m in link["misses"] if m["gold_tier"] == "catalog" and m["got_tier"] == "dummy"]
        if wrong_dummy:
            screens = "; ".join(sorted({m["screen"] for m in wrong_dummy}))
            items.append(
                f"**Over-cautious links.** {len(wrong_dummy)} steps with a real catalog entry fell back to the "
                f"placeholder because the match scored under the catalog floor: {screens}."
            )
    cache = (load or {}).get("cache") or {}
    nm = cache.get("near_miss")
    if nm and nm.get("leaked"):
        axes: dict[str, int] = {}
        for leak in nm["leaked"]:
            axes[leak.get("differs_in") or "?"] = axes.get(leak.get("differs_in") or "?", 0) + 1
        items.append(
            f"**Near-miss cache hits.** {nm['hits']} of {nm['n']} near misses were served a cached plan "
            f"({', '.join(f'{k}: {v}' for k, v in sorted(axes.items()))}). The slot guard compares component "
            'and symptom only, so a how-do-I request ("I want my screen to go black after a minute") matches '
            "the cached fault report with the same words; an empty cached symptom also matches any symptom."
        )
    para = cache.get("paraphrase")
    if para and para.get("by_register"):
        weak = {k: v for k, v in para["by_register"].items() if v < 0.8}
        if weak:
            items.append(
                "**Weak paraphrase registers.** "
                + ", ".join(f"{k} {v * 100:.0f}%" for k, v in weak.items())
                + " hit rate with the cache warmed on the original phrasing only."
            )
    pending = []
    if not judge:
        pending.append("step accuracy (`judge.py`)")
    if not (load or {}).get("api"):
        pending.append("cold-path latency and cost over HTTP (`loadtest.py --mode api`)")
    if not gates or not ((gates.get("blocks", {}).get("A1") or {}).get("detail") or {}).get("non_empty"):
        pending.append("schema and rule compliance on non-empty plans (`gate_replica.py`)")
    if ablation and any(not v["measured"] for v in ablation["variants"]):
        pending.append("the LLM mapping baseline (`ablation.py`, needs `MISTRAL_API_KEY`)")
    if pending:
        items.append("**Not measured yet:** " + "; ".join(pending) + ".")
    return ["## 6. Known Edge Cases & System Limitations", "", *(f"* {i}" for i in items), ""]


def build(args) -> str:
    rd = args.results_dir
    gates, judge = load("gates.json", rd), load("judge.json", rd)
    ablation, load_ = load("ablation.json", rd), load("loadtest.json", rd)
    header = [
        "# System Performance Metrics & Evaluation Report",
        f"**Model(s):** {args.model or NM}",
        f"**Embeddings:** {args.embeddings or embed_model()}",
        f"**Environment:** {args.env or local_env()}",
        "",
        (
            f"_Generated by `eval/report.py` at commit `{commit_sha()}` on "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Template: Theme 2 spec, "
            "Appendix C. Every number comes from a run in `eval/results/`; anything without one says "
            "“not measured”._"
        ),
        "",
    ]
    body = [
        *section1(gates),
        *section2(judge, ablation),
        *section3(load_),
        *section4(load_),
        *section5(ablation),
        *section6(ablation, load_, gates, judge),
    ]
    return "\n".join([*header, *body]).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", help="provider/model id(s) of the LLM stages")
    parser.add_argument("--embeddings", help="embedding model id (default: read from api/app/config.py)")
    parser.add_argument("--env", help="vCPU / RAM / OS of the serving machine (default: this machine)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    args.out.write_text(build(args))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
