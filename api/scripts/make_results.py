"""Offline: run cold pipeline over data/kit -> results.jsonl (meta beside response).

Every kit query runs on an empty cache, so each line is a genuinely cold answer. One JSON line per
query (Appendix B): {query, query_variations, response, meta}.

    python scripts/make_results.py                                   # -> <repo>/results.jsonl
    python scripts/make_results.py --extract-model gemini-3.5-flash-lite \\
        --out ../eval/results/bakeoff_gemini-3.5-flash-lite.jsonl     # model bake-off (Phase 3)
    python scripts/make_results.py --pause 4                         # free-tier keys: pace the calls
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A measurement run must never read or write the API's real cache.sqlite.
os.environ.setdefault(
    "ONECLICK_SQLITE", str(Path(tempfile.mkdtemp(prefix="oneclick-results-")) / "cache.sqlite")
)

from app import cache
from app.config import settings
from app.obs import readiness
from app.pipeline.run import run_with_variations

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kit", type=Path, default=Path(settings.data_dir) / "kit" / "siis_responses.json")
    parser.add_argument("--out", type=Path, default=REPO / "results.jsonl")
    parser.add_argument("--enrich-model", help="override settings.enrich_model for this run")
    parser.add_argument("--extract-model", help="override settings.extract_model for this run")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between queries (free-tier limits)")
    args = parser.parse_args()

    if args.enrich_model:
        settings.enrich_model = args.enrich_model
    if args.extract_model:
        settings.extract_model = args.extract_model

    state = readiness.warm()
    if not state["ready"]:
        sys.exit(f"engine not ready: {state['error']} (run scripts/build_screengraph.py and build_index.py)")

    rows = json.loads(args.kit.read_text(encoding="utf-8"))["responses"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    empty, latencies = 0, []
    with args.out.open("w", encoding="utf-8", newline="\n") as out:
        for n, row in enumerate(rows):
            if n and args.pause:
                time.sleep(args.pause)
            cache.clear()  # every line is a cold answer
            body, variations = run_with_variations(row["original_query"], row.get("siis_response"))
            meta = body.get("meta", {})
            line = {
                "query": row["original_query"],
                "query_variations": variations,
                "response": {"contexts": body.get("contexts", [])},
                "meta": meta,
            }
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            empty += not line["response"]["contexts"]
            latencies.append(meta.get("latency_ms", 0.0))
            print(
                f"{row['id']:>7}  {meta.get('latency_ms', 0):7.0f} ms  goals={len(line['response']['contexts'])}"
                f"  model={meta.get('model')}  vars={len(variations)}  fallback={meta.get('fallback')}"
            )
    cache.clear()
    latencies.sort()
    p95 = latencies[max(0, round(0.95 * len(latencies)) - 1)] if latencies else 0.0
    print(f"wrote {len(rows)} lines to {args.out}  ({empty} empty, cold p95 {p95:.0f} ms)")


if __name__ == "__main__":
    main()
