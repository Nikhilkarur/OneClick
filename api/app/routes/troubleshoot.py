"""POST /v1/troubleshoot — the scored endpoint. Always 200 with a schema-valid body."""

import time

from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.config import settings
from app.models import ResponseMeta
from app.pipeline import run as pipeline_run

router = APIRouter()


class TroubleshootRequest(BaseModel):
    query: str
    siis_response: dict | str | None = None


@router.post("/v1/troubleshoot")
def troubleshoot(req: TroubleshootRequest, response: Response) -> dict:
    start = time.perf_counter()
    try:
        body = pipeline_run.run(req.query, req.siis_response)
    except Exception:  # noqa: BLE001 - run() never raises; this is the last line of hard rule 4
        fallback = "no_siis_context" if req.siis_response is None else "no_match"
        body = {"contexts": [], "meta": ResponseMeta(fallback=fallback).model_dump(mode="json")}
    meta = body.get("meta") or {}
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Latency-Ms"] = str(latency_ms)
    response.headers["X-Cache-Hit"] = "true" if meta.get("cache_hit") else "false"
    response.headers["X-Cache-Tier"] = meta.get("cache_tier") or "none"
    response.headers["X-Cost-Usd"] = f"{float(meta.get('cost_usd') or 0.0):.6f}"
    out = {"contexts": body.get("contexts", [])}
    if settings.include_meta:
        out["meta"] = {**meta, "latency_ms": latency_ms}
    return out
