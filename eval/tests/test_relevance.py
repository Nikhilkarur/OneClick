"""The 0-2 deeplink relevance rubric and gold loading."""

import pytest

from evalkit.relevance import (
    CATALOG,
    DUMMY,
    MANUAL,
    GoldCase,
    Prediction,
    Screens,
    exact,
    load_gold,
    relevance,
)

# DL-0125 / DL-0126 are the off / on toggles of Touch sensitivity: one screen, two controls.
TOUCH_ON, TOUCH_OFF, WIFI = "DL-0126", "DL-0125", "DL-0313"


@pytest.fixture(scope="module")
def screens() -> Screens:
    return Screens.load()


def case(tier: str, expected: str | None = None, **kw) -> GoldCase:
    return GoldCase(
        step="s", screen="Settings > X", verb="enable", tier=tier, expected_id=expected, owner="t", **kw
    )


def test_same_screen_comes_from_the_validation_deeplink(screens):
    assert screens.same_screen(TOUCH_ON, TOUCH_OFF)
    assert not screens.same_screen(TOUCH_ON, WIFI)
    assert not screens.same_screen(None, TOUCH_ON)


@pytest.mark.parametrize(
    ("pred", "score"),
    [
        (Prediction(CATALOG, TOUCH_ON), 2),
        (Prediction(CATALOG, TOUCH_OFF), 1),  # right screen, wrong control
        (Prediction(CATALOG, WIFI), 0),
        (Prediction(DUMMY), 0),
        (Prediction(MANUAL), 0),
    ],
)
def test_catalog_gold(screens, pred, score):
    assert relevance(case(CATALOG, TOUCH_ON), pred, screens) == score


def test_acceptable_and_parent_ids(screens):
    gold = case(CATALOG, TOUCH_ON, acceptable_ids=frozenset({"DL-0001"}), parent_ids=frozenset({WIFI}))
    assert relevance(gold, Prediction(CATALOG, "DL-0001"), screens) == 2
    assert relevance(gold, Prediction(CATALOG, WIFI), screens) == 1
    assert exact(gold, Prediction(CATALOG, "DL-0001"))
    assert not exact(gold, Prediction(CATALOG, WIFI))


@pytest.mark.parametrize(
    ("pred", "score"),
    [(Prediction(DUMMY), 2), (Prediction(MANUAL), 1), (Prediction(CATALOG, WIFI), 0)],
)
def test_dummy_gold(screens, pred, score):
    assert relevance(case(DUMMY), pred, screens) == score


def test_dummy_gold_parent_menu_scores_one(screens):
    assert relevance(case(DUMMY, parent_ids=frozenset({WIFI})), Prediction(CATALOG, WIFI), screens) == 1


@pytest.mark.parametrize(
    ("pred", "score"),
    [(Prediction(MANUAL), 2), (Prediction(DUMMY), 0), (Prediction(CATALOG, WIFI), 0)],
)
def test_manual_gold_never_rewards_a_link(screens, pred, score):
    assert relevance(case(MANUAL), pred, screens) == score


def test_exact_is_only_for_catalog_gold():
    assert not exact(case(DUMMY), Prediction(DUMMY))


def test_load_gold_reads_the_team_file():
    cases = load_gold()
    assert len(cases) >= 100
    assert {c.tier for c in cases} <= {CATALOG, DUMMY, MANUAL}
    assert all(c.expected_id for c in cases if c.tier == CATALOG)
    assert all(c.expected_id is None for c in cases if c.tier != CATALOG)
