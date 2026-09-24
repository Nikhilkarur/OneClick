"""LLM router (ADR-002) and the LLM stages, against a stubbed HTTP transport: no quota is spent.

Covers the fallback matrix (timeout, 429, bad JSON, missing key, both down), the quality/fast race,
answer rejection, cooldowns, cost accounting, and what the pipeline does with an LLM answer: select
mode turns sentence ids into the article's own steps, invented ids are dropped, background variations
reach the cache, and a failed LLM run is capped and never cached.
"""

import json
import time
from pathlib import Path

import httpx
import pytest

from app import cache
from app.config import settings
from app.llm import router
from app.llm.errors import LLMCallError, parse_json
from app.models import Intent, SiisSentence, Slots
from app.pipeline import enrich as enrich_module
from app.pipeline import extract as extract_module
from app.pipeline.run import run_stream, run_with_variations

ROOT = Path(__file__).resolve().parents[2]
REQUEST = json.loads((ROOT / "data/fixtures/touch_lag/request.json").read_text(encoding="utf-8"))
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def _gemini_ok(payload: dict, tokens=(100, 50, 20)):
    body = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": tokens[0],
            "candidatesTokenCount": tokens[1],
            "thoughtsTokenCount": tokens[2],
        },
    }
    return httpx.Response(200, json=body)


def _mistral_ok(payload: dict):
    body = {
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 40},
    }
    return httpx.Response(200, json=body)


def _clients(gemini_handler, mistral_handler):
    return {
        "gemini": httpx.Client(transport=httpx.MockTransport(gemini_handler)),
        "mistral": httpx.Client(transport=httpx.MockTransport(mistral_handler)),
    }


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral")
    router.reset_cooldowns()
    yield
    router.reset_cooldowns()


@pytest.fixture
def gemini_first(monkeypatch, keys):
    """The paid-tier shape: Gemini primary for call A, Mistral fallback."""
    monkeypatch.setattr(settings, "enrich_model", "gemini-3.1-flash-lite")
    monkeypatch.setattr(settings, "fallback_model", "mistral-small-2603")


def _complete(clients, info, **kwargs):
    return router.complete_json(
        "enrich", {"query": "screen is black"}, SCHEMA, stage="enrich", info=info, clients=clients, **kwargs
    )


# ---- router: fallback matrix ------------------------------------------------------------------------
def test_primary_answers_and_cost_counts_thinking_tokens(gemini_first):
    seen = {}

    def gemini(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return _gemini_ok({"ok": True})

    info = {}
    assert _complete(_clients(gemini, lambda r: pytest.fail("no fallback")), info) == {"ok": True}
    assert settings.enrich_model in seen["url"]
    assert seen["body"]["generationConfig"]["responseJsonSchema"] == SCHEMA
    assert seen["body"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": settings.enrich_thinking}
    assert "screen is black" in seen["body"]["contents"][0]["parts"][0]["text"]
    assert info["model"] == settings.enrich_model and info["tokens_in"] == 100 and info["tokens_out"] == 70
    price_in, price_out = settings.llm_prices[settings.enrich_model]
    assert info["cost_usd"] == pytest.approx((100 * price_in + 70 * price_out) / 1e6)
    assert info["fallback_used"] is False


@pytest.mark.parametrize(
    "failure",
    [
        lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=r)),
        lambda r: httpx.Response(429, headers={"retry-after": "7"}, text="quota"),
        lambda r: httpx.Response(503, text="down"),
        lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}),
    ],
    ids=["timeout", "429", "503", "bad_json"],
)
def test_any_primary_failure_falls_back(gemini_first, failure):
    info = {}
    assert _complete(_clients(failure, lambda r: _mistral_ok({"ok": True})), info) == {"ok": True}
    assert info["model"] == settings.fallback_model and info["fallback_used"] is True
    assert [a["ok"] for a in info["attempts"]] == [False, True]


def test_capacity_errors_put_a_model_on_cooldown_but_timeouts_do_not(gemini_first):
    calls = []

    def gemini(request):
        calls.append(1)
        return httpx.Response(503, text="busy")

    clients = _clients(gemini, lambda r: _mistral_ok({"ok": True}))
    _complete(clients, {})
    info = {}
    _complete(clients, info)  # second request: Gemini is skipped, no call made
    assert len(calls) == 1 and info["attempts"][0]["error"] == "cooldown"
    router.reset_cooldowns()

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    _complete(_clients(slow, lambda r: _mistral_ok({"ok": True})), {})
    assert not router.cooling_down(settings.enrich_model)


def test_rejected_thinking_level_is_retried_without_it(gemini_first):
    bodies = []

    def gemini(request):
        bodies.append(json.loads(request.content))
        if "thinkingConfig" in bodies[-1]["generationConfig"]:
            return httpx.Response(400, text="Unknown name thinkingLevel")
        return _gemini_ok({"ok": True})

    assert _complete(_clients(gemini, lambda r: pytest.fail("no fallback")), {}) == {"ok": True}
    assert len(bodies) == 2


def test_an_answer_the_caller_rejects_counts_as_a_failure(gemini_first):
    info = {}
    clients = _clients(lambda r: _gemini_ok({"ok": False}), lambda r: _mistral_ok({"ok": True}))
    assert _complete(clients, info, accept=lambda a: a["ok"]) == {"ok": True}
    assert info["attempts"][0]["error"] == "rejected"


def test_both_down_raises_llm_error_with_both_attempts(gemini_first):
    with pytest.raises(router.LLMError) as err:
        _complete(_clients(lambda r: httpx.Response(500), lambda r: httpx.Response(500)), {})
    assert [a["error"] for a in err.value.attempts] == ["http_500", "http_500"]


def test_missing_primary_key_uses_the_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral")
    monkeypatch.setattr(settings, "enrich_model", "gemini-3.1-flash-lite")
    monkeypatch.setattr(settings, "fallback_model", "mistral-small-2603")
    router.reset_cooldowns()
    info = {}
    assert _complete(_clients(lambda r: pytest.fail("no call"), lambda r: _mistral_ok({"ok": True})), info)
    assert info["attempts"][0]["error"] == "no_key" and router.available()


def test_provider_is_chosen_by_model_family():
    assert router.provider("gemini-3.8-flash") == "gemini"
    for model in ("mistral-small-2603", "ministral-14b-latest", "magistral-small-latest"):
        assert router.provider(model) == "mistral"


# ---- router: the quality / fast race (extract stage) ------------------------------------------------
def _race_clients(delays: dict[str, float], answers: dict[str, dict]):
    def mistral(request):
        model = json.loads(request.content)["model"]
        time.sleep(delays.get(model, 0.0))
        return _mistral_ok(answers[model])

    return _clients(lambda r: httpx.Response(503), mistral)


@pytest.fixture
def race(monkeypatch, keys):
    monkeypatch.setattr(settings, "extract_model", "ministral-14b-latest")
    monkeypatch.setattr(settings, "extract_fast_model", "ministral-8b-latest")
    monkeypatch.setattr(settings, "extract_prefer_deadline_s", 0.4)
    monkeypatch.setattr(settings, "extract_primary_timeout_s", 2.0)
    monkeypatch.setattr(settings, "extract_budget_s", 2.5)
    monkeypatch.setattr(settings, "fallback_model", "gemini-3-flash-preview")


def _extract(clients, info, accept=None):
    return router.complete_json(
        "extract",
        {"query": "q", "sentences": "s"},
        SCHEMA,
        stage="extract",
        info=info,
        clients=clients,
        accept=accept,
    )


def test_race_keeps_the_quality_answer_when_it_is_back_by_the_deadline(race):
    info = {}
    answers = {"ministral-14b-latest": {"ok": True}, "ministral-8b-latest": {"ok": False}}
    assert _extract(_race_clients({"ministral-14b-latest": 0.1}, answers), info) == {"ok": True}
    assert info["model"] == "ministral-14b-latest"


def test_race_takes_the_fast_answer_when_the_quality_one_is_late(race):
    info = {}
    answers = {"ministral-14b-latest": {"ok": True}, "ministral-8b-latest": {"ok": False}}
    started = time.perf_counter()
    assert _extract(_race_clients({"ministral-14b-latest": 1.5}, answers), info) == {"ok": False}
    assert info["model"] == "ministral-8b-latest" and time.perf_counter() - started < 1.2


def test_race_skips_a_rejected_quality_answer(race):
    info = {}
    answers = {"ministral-14b-latest": {"ok": False}, "ministral-8b-latest": {"ok": True}}
    result = _extract(_race_clients({"ministral-8b-latest": 0.2}, answers), info, accept=lambda a: a["ok"])
    assert result == {"ok": True} and info["model"] == "ministral-8b-latest"


def test_parse_json_tolerates_fences_and_rejects_non_objects():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    for bad in ("[1, 2]", "nothing here"):
        with pytest.raises(LLMCallError):
            parse_json(bad)


def test_prompts_render_every_placeholder():
    for name, variables in (
        ("enrich", {"query": "q"}),
        ("extract", {"query": "q", "intents": "i", "sentences": "s"}),
        ("variations", {"query": "q"}),
    ):
        text = router.render(router.load_prompt(name, router.prompt_version(name)), variables)
        assert "{{" not in text and "TODO" not in text


# ---- the LLM stages inside the pipeline -------------------------------------------------------------
VARIATIONS = [
    "My S22 display reacts slowly whenever I touch it.",
    "galaxy s22 touch lag delay",
    "Why is my phone so slow to register taps?! Super annoying.",
    "touchscren lagg on my galxy",
    "I am experiencing a noticeable latency when interacting with the display.",
    "the screen takes ages to respond when i swipe",
    "S22 unresponsive touch input issue",
    "Taps register late on my Samsung, what can I do?",
    "My phone ignores my fingers for a second before reacting.",
    "touch response delayed samsung s22 fix",
    "Whenever I type, letters appear a moment later on the screen.",
    "Display feels sluggish to touch, any help?",
]
SELECT_ANSWER = {
    "goals": [
        {
            "problem": "Touchscreen input is delayed and laggy",
            "title": "Touchscreen Input Lag Problem",
            "topic": "Touchscreen Issues",
            "domain": "Display",
            "actions": [
                {
                    "src_ids": ["S31", "S32"],
                    "name": "Turn Off Full Screen Gestures",
                    "description": "It will switch to navigation buttons",
                    "screen_path": "Settings > Display > Navigation bar",
                    "intent_verb": "open",
                },
                {
                    "src_ids": ["S999"],
                    "name": "Recalibrate Touchscreen",
                    "description": "It will recalibrate the touch panel",
                    "screen_path": "",
                    "intent_verb": "none",
                },
            ],
        }
    ]
}


@pytest.fixture
def fake_llm(monkeypatch, keys):
    calls = []

    def complete_json(prompt_name, variables, schema, *, stage=None, info=None, clients=None, accept=None):
        calls.append(prompt_name)
        answer = {"variations": VARIATIONS} if prompt_name == "variations" else SELECT_ANSWER
        if accept is not None:
            assert accept(answer)
        if info is not None:
            info.update(model=f"fake-{prompt_name}", tokens_in=10, tokens_out=5, cost_usd=0.0001, attempts=[])
        return answer

    monkeypatch.setattr(router, "complete_json", complete_json)
    return calls


def test_select_mode_uses_the_articles_own_steps_and_the_models_intents(fake_llm):
    cache.clear()
    events = {e.stage.value: e.detail for e in run_stream(REQUEST["query"], REQUEST["siis_response"])}
    assert events["extract"]["source"] == "llm" and events["extract"]["mode"] == "select"
    assert events["ground"]["dropped_steps"] == []  # the invented S999 never became a step
    body = events["done"]
    (goal,) = body["contexts"]
    assert goal["goal"] == "Follow these steps to perform this Touchscreen Issues Troubleshooting."
    assert goal["title"] == "Touchscreen input lag"
    (action,) = goal["actions"]
    assert action["actionName"] == "Turn Off Full Screen Gestures" and action["category"] == "auto"
    assert action["stepGroups"][0]["steps"] == [
        "Go to Settings.",
        "Tap Display.",
        "Tap Navigation bar.",
        "Select Buttons to turn off full screen gestures.",
    ]
    assert body["meta"]["model"] == "fake-extract"
    assert sorted(fake_llm) == ["extract", "variations"]
    cache.clear()


def test_background_variations_reach_results_and_the_cache(fake_llm):
    cache.clear()
    body, variations = run_with_variations(REQUEST["query"], REQUEST["siis_response"])
    assert body["contexts"] and settings.variation_min <= len(variations) <= settings.variation_max
    assert set(variations) & set(VARIATIONS), "the model's variations, not only templates"
    paraphrase = next(v for v in variations if v in VARIATIONS)
    events = [e.stage.value for e in run_stream(paraphrase, REQUEST["siis_response"])]
    assert events == ["cache", "done"]  # a stored variation now answers from the cache
    cache.clear()


def test_llm_failure_degrades_to_rules_capped_and_repeats_identically(monkeypatch, keys):
    def down(*args, **kwargs):
        raise router.LLMError([{"model": "x", "ok": False, "error": "http_500", "ms": 1}])

    monkeypatch.setattr(router, "complete_json", down)
    cache.clear()
    events = {e.stage.value: e.detail for e in run_stream(REQUEST["query"], REQUEST["siis_response"])}
    assert events["extract"]["source"] == "rules"
    body = events["done"]
    assert body["contexts"] and all(g["score"] <= settings.rules_only_score_cap for g in body["contexts"])
    repeat = {e.stage.value: e.detail for e in run_stream(REQUEST["query"], REQUEST["siis_response"])}
    assert repeat["done"]["meta"]["cache_tier"] == "exact" and repeat["done"]["contexts"] == body["contexts"]
    cache.clear()


def test_selection_keeps_only_real_sentences_and_splits_instructions():
    sentences = [
        SiisSentence(id="S1", section="s", text="Go to Settings, tap Display, and then tap Dark mode."),
        SiisSentence(id="S2", section="s", text="Dark mode is easier on the eyes."),
    ]
    action = {
        "src_ids": ["S1", "S2", "S9"],
        "name": "A",
        "description": "",
        "screen_path": "",
        "intent_verb": "x",
    }
    goal = {"problem": "Screen too bright", "title": "bright screen", "topic": "", "domain": "Nope"}
    answer = {"goals": [{**goal, "actions": [action]}]}
    actions, topics, intents = extract_module.actions_from_selection(answer, sentences)
    assert [s.text for s in actions[0].steps] == ["Go to Settings.", "Tap Display.", "Tap Dark mode."]
    assert all(s.src_ids == ["S1"] for s in actions[0].steps) and actions[0].intent_verb is None
    assert intents == [Intent(text="Screen too bright", title="Bright screen", domain="Other")]
    assert topics == ["Bright screen"]
    assert not extract_module.usable_selection(answer, {"S7"})


def test_enrich_llm_on_the_critical_path_still_works(monkeypatch, keys):
    def complete_json(prompt_name, variables, schema, *, stage=None, info=None, clients=None, accept=None):
        return {
            "canonical_query": "Touchscreen input is delayed",
            "intents": [{"text": "Touch input lags", "domain": "Display", "title": "Touch input lag"}],
            "variations": VARIATIONS,
        }

    monkeypatch.setattr(router, "complete_json", complete_json)
    monkeypatch.setattr(settings, "enrich_llm_on_critical_path", True)
    intents, variations, detail = enrich_module.enrich_with_report(
        REQUEST["query"], Slots(component="screen")
    )
    assert detail["source"] == "llm" and intents[0].title == "Touch input lag"
    assert settings.variation_min <= len(variations) <= settings.variation_max
