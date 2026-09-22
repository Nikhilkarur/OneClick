"""Rules and BM25 mappers, and parsing of the LLM mapper's reply. No network."""

import pytest

from evalkit.mappers import LlmMapper, RulesMapper, is_physical, leaf, parse_llm_choice, screen_name
from evalkit.relevance import CATALOG, DUMMY, MANUAL, Prediction


@pytest.fixture(scope="module")
def rules() -> RulesMapper:
    return RulesMapper()


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("View Touch Sensitivity Settings", "touch sensitivity"),
        ("Enable Wi-Fi", "wi fi"),
        ("Touch sensitivity", "touch sensitivity"),
        ("Navigation bar menu", "navigation bar"),
    ],
)
def test_screen_name_strips_verbs_and_suffixes(text, name):
    assert screen_name(text) == name


def test_leaf_is_the_last_path_segment():
    assert leaf("Settings > Display > Touch sensitivity") == "touch sensitivity"


def test_rules_pick_the_entry_type_the_verb_asks_for(rules):
    screen = "Settings > Display > Touch sensitivity"
    assert rules("Turn it on.", screen, "enable") == Prediction(CATALOG, "DL-0126")
    assert rules("Turn it off.", screen, "disable") == Prediction(CATALOG, "DL-0125")


def test_rules_refuse_physical_steps(rules):
    assert rules("Visit a service centre.", "Samsung service centre", "visit") == Prediction(MANUAL)
    assert is_physical("Wipe it with a soft cloth.", "Charging port", "open")


def test_rules_fall_back_by_where_the_screen_lives(rules):
    assert rules("Open it.", "Settings > Display > Imaginary panel", "open") == Prediction(DUMMY)
    assert rules("Open Downloads.", "My Files > Downloads", "open") == Prediction(MANUAL)


IDS = {"DL-0126", "DL-0125"}


@pytest.mark.parametrize(
    ("content", "pred"),
    [
        ('{"id": "DL-0126"}', Prediction(CATALOG, "DL-0126")),
        ('{"id": "DUMMY"}', Prediction(DUMMY)),
        ('{"id": "manual"}', Prediction(MANUAL)),
        ('{"id": "DL-9999"}', Prediction(MANUAL)),  # hallucinated id: no link, never a made-up URI
        ("not json", Prediction(MANUAL)),
        ("[1, 2]", Prediction(MANUAL)),
    ],
)
def test_parse_llm_choice(content, pred):
    assert parse_llm_choice(content, IDS) == pred


def test_llm_mapper_is_unavailable_without_a_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    mapper = LlmMapper(model="m")
    assert not mapper.available
    assert "DL-0126" in mapper.catalog
