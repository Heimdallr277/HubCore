"""Interface checks: palette contrast, theme consistency, the featured decision
(future 26), displayed benchmarks and the regression counts quoted in the documentation."""

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from hubcore import allocation, annual_reference, games
from hubcore import robustness as R
from hubcore import uncertainty as U
from hubcore.costs import Assumptions, coalition_costs
from hubcore.park import EXAMPLE_PARK
from ui import charts as V
from ui import featured, style

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "streamlit_app.py")
ENSEMBLE = ROOT / "data/precomputed/voorbeeldpark__example_N200_seed42.npz"
CHECKS = json.loads((ROOT / "data/preset_lp_checks.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------- palette ----
def test_palette_tokens():
    assert (V.MIST, V.WHITE, V.POWDER) == ("#F3F7FB", "#FFFFFF", "#DCE8F0")
    assert (V.INK, V.SLATE, V.STEEL, V.AZURE) == ("#23384A", "#526778", "#376580", "#286AD6")
    assert (V.TEAL, V.CLAY, V.GOLD) == ("#38746E", "#A24E45", "#E9C66A")
    assert V.AZURE_FILL == "rgba(40,106,214,0.12)"


@pytest.mark.parametrize("fg,bg", [
    (V.INK, V.MIST), (V.INK, V.WHITE), (V.INK, V.POWDER),
    (V.SLATE, V.MIST), (V.SLATE, V.WHITE), (V.SLATE, V.POWDER),
    (V.STEEL, V.WHITE), (V.STEEL, V.MIST), (V.WHITE, V.AZURE), (V.INK, V.GOLD),
    (V.AZURE, V.WHITE), (V.AZURE, V.MIST),
    (V.INK, V.blend(V.TEAL, V.WHITE, 0.12)), (V.INK, V.blend(V.CLAY, V.WHITE, 0.12)),
    (V.INK, V.blend(V.AZURE, V.WHITE, 0.12)), (V.INK, V.blend(V.AZURE, V.MIST, 0.12)),
    (V.WHITE, V.TEAL), (V.WHITE, V.CLAY),            # amounts written inside the objection bars
])
def test_text_pairs_reach_4_5_to_1(fg, bg):
    assert V.contrast(fg, bg) >= 4.5


@pytest.mark.parametrize("bg", [V.WHITE, V.MIST])
def test_essential_marks_and_controls_reach_3_to_1(bg):
    for color in V.SERIES + [V.TEAL, V.CLAY, V.AZURE, V.STEEL, V.INK, V.AXIS, style.CONTROL]:
        assert V.contrast(color, bg) >= 3.0, color
    # status icons on their own tinted badges
    assert V.contrast(V.TEAL, V.blend(V.TEAL, bg, 0.12)) >= 3.0
    assert V.contrast(V.CLAY, V.blend(V.CLAY, bg, 0.12)) >= 3.0
    assert V.contrast(V.AZURE, V.POWDER) >= 3.0          # focus ring in the sidebar


def test_heatmap_labels_and_ramp():
    assert V.contrast(V.BLUE_RAMP[0], V.WHITE) >= 2.0     # lightest step visible on white
    lum = [V.luminance(c) for c in V.BLUE_RAMP]
    assert all(a > b for a, b in zip(lum, lum[1:]))       # light to dark
    import pandas as pd
    table = pd.DataFrame({"rule": ["a", "b"], "clause": ["x", "x"], "v": [0.0, 0.55]})
    table = table.set_index(["rule", "clause"])
    fig = V.robustness_heat(table, "v", ["a", "b"], {"a": "A", "b": "B"}, {"x": "X"})
    labels = fig.layout.annotations
    assert len(labels) == 2
    for ann in labels:                                     # ink on white whatever the cell fill
        assert ann.bgcolor == V.WHITE and ann.font.color == V.INK
    assert V.contrast(V.INK, V.WHITE) >= 4.5


def test_theme_file_uses_the_same_tokens():
    theme = tomllib.loads((ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))["theme"]
    assert theme["primaryColor"].upper() == V.AZURE
    assert theme["backgroundColor"].upper() == V.MIST
    assert theme["secondaryBackgroundColor"].upper() == V.POWDER
    assert theme["textColor"].upper() == V.INK
    assert theme["greenColor"].upper() == V.TEAL and theme["redColor"].upper() == V.CLAY
    assert [c.upper() for c in theme["chartCategoricalColors"]] == V.SERIES
    assert len(theme["chartSequentialColors"]) == 10
    assert V.contrast(theme["borderColor"], V.WHITE) >= 3.0
    for key in ("greenTextColor", "redTextColor", "blueTextColor", "yellowTextColor"):
        tint = {"green": V.TEAL, "red": V.CLAY, "blue": V.AZURE, "yellow": V.GOLD}[key[:-9]]
        assert V.contrast(theme[key], V.blend(tint, V.MIST, 0.10)) >= 4.5, key
    assert V.contrast(theme["yellowTextColor"], theme["yellowBackgroundColor"]) >= 4.5


# ------------------------------------------------------ featured decision ----
@pytest.fixture(scope="module")
def case():
    a = Assumptions()
    space = U.default_space(EXAMPLE_PARK, a)
    ens = R.load(ENSEMBLE, EXAMPLE_PARK, a, space, 200, 42)
    k, p = 26, EXAMPLE_PARK
    row = U.sample(space, 200, 42).iloc[k]
    settings, growth = U.to_future(row, a, p.n)
    loads = p.loads(growth)
    fast = coalition_costs(loads, p.contracts(), settings).cost
    lp = annual_reference.price_game(p, settings, growth)
    np.testing.assert_allclose(fast, ens.costs[k], rtol=0, atol=1e-5)
    with np.load(ROOT / "data/preset_lp_costs.npz") as z:
        assert str(z["training_design"]) == R.fingerprint(p, a, space, 200, 42)
        np.testing.assert_allclose(lp, z["training"][k], rtol=0, atol=1e-5)
    args = (p.names, p.contracts(), loads.sum(axis=1), 0.01, 500.0)
    return {"s": featured.verdict("s", fast, *args), "c": featured.verdict("c", lp, *args),
            "a": featured.attribution(fast, lp, p.names.index("Cold store"), *args), "names": p.names}


def test_future_26_screening_passes_and_lp_finds_the_cold_store_objection(case):
    s, c = case["s"], case["c"]
    cold = case["names"].index("Cold store")
    assert s.passes and not c.passes
    assert s.excess[cold] == pytest.approx(444.95, abs=0.005)
    assert c.excess[cold] == pytest.approx(605.44, abs=0.005)
    assert c.cutoff[cold] == 500.0 and s.cutoff[cold] == 500.0
    assert [g for g, _, _ in c.material] == ["Cold store", "Bakery"]   # least-core equalizes both
    assert featured.ties(c, case["names"]) == ["Cold store", "Bakery"]
    assert c.eps == pytest.approx(605.44, abs=0.005) and s.eps == pytest.approx(444.95, abs=0.005)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for v in (s, c):
        for value in (v.standalone[cold], v.allocated[cold], v.excess[cold]):
            assert f"€{value:,.2f}" in readme


def test_future_26_attribution(case):
    a = case["a"]
    assert a["standalone_change"] == pytest.approx(-201.67, abs=0.005)
    assert a["eps_only_member_repriced"] == pytest.approx(512.17, abs=0.005)
    assert not a["passes_only_member_repriced"]


def test_method_page_quotes_the_computed_case_numbers(case):
    text = (ROOT / "app_pages/method.py").read_text(encoding="utf-8")
    for value in (case["s"].excess[2], -case["a"]["standalone_change"], case["a"]["eps_only_member_repriced"],
                  case["c"].excess[2]):
        assert f"€{value:,.2f}" in text
    changed = CHECKS["memberships"]["six_firms"]["changed_future_ids"]["stable"]
    moved = changed["screening_pass_reference_fail"] + changed["screening_fail_reference_pass"]
    assert sorted(moved) == [26, 28, 152]


# --------------------------------------------------- regression counts ----
@pytest.mark.slow
def test_regression_counts_for_both_memberships_and_both_price_sets():
    """Counts reported in the README and case study (N=200, seed 42, default thresholds)."""
    a = Assumptions()
    space = U.default_space(EXAMPLE_PARK, a)
    ens = R.load(ENSEMBLE, EXAMPLE_PARK, a, space, 200, 42)
    with np.load(ROOT / "data/preset_lp_costs.npz") as z:
        lp = z["training"]
    expected = {(6, "fast"): (160, 33, 52), (6, "lp"): (160, 32, 51),
                (5, "fast"): (163, 99, 126), (5, "lp"): (163, 96, 124)}
    for members in (list(range(6)), [0, 1, 3, 4, 5]):
        for label, costs in (("fast", ens.costs), ("lp", lp)):
            viable = core = passes = 0
            for k in range(200):
                c = games.subgame(costs[k], 6, members)
                n = len(members)
                viable += c[-1] < c[[1 << i for i in range(n)]].sum()
                core += games.least_core_value(c, n) <= games.numerical_tol(c)
                x = allocation.allocate("leastcore", c, n, EXAMPLE_PARK.contracts()[members],
                                        ens.kwh[k][members])
                passes += games.stability(x, c, n, 0.01, 500.0).stable
            assert (viable, core, passes) == expected[(len(members), label)]
    six = CHECKS["memberships"]["six_firms"]["reference"]
    five = CHECKS["memberships"]["without_cold_store"]["reference"]
    assert (six["viable"], six["exact_core_exists"], six["adaptive_material_pass"]) == expected[(6, "lp")]
    assert (five["viable"], five["exact_core_exists"], five["adaptive_material_pass"]) == expected[(5, "lp")]


# ------------------------------------------------------------ interface ----
def _texts(at):
    elements = list(at.markdown) + list(at.caption) + list(at.info)
    return " ".join(el.value for el in elements if isinstance(el.value, str))


def test_overview_shows_the_featured_card_and_opens_future_26():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    text = _texts(at)
    for phrase in ("Future 26", "Screening passes", "Continuous LP finds an objection",
                   "Cold store pays €605.44/year above its standalone cost. "
                   "Its materiality cutoff is €500/year."):
        assert phrase in text, phrase
    next(b for b in at.button if b.label == "Open future 26").click().run()
    assert not at.exception
    assert at.query_params.get("case") in ("26", ["26"])
    text = _texts(at)
    assert "Material objection: Cold store, Bakery" in text
    assert at.toggle(key="featured_lp").value is True


def test_stress_link_shows_both_calculations_and_displayed_benchmarks_match_json():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["case"] = "26"
    at.run().switch_page("app_pages/stress.py").run()
    assert not at.exception
    text = _texts(at)
    assert "Fast screening" in text and "Continuous LP" in text
    assert "€444.95/yr" in text and "€605.44/yr" in text
    assert ("Repricing only that one standalone cost lifts the least-core's largest excess "
            "from €444.95 to €512.17") in text
    six = CHECKS["memberships"]["six_firms"]["reference"]
    info = " ".join(el.value for el in at.info)
    assert f"cheaper in {six['viable']}/200" in info
    assert f"core exists in {six['exact_core_exists']}/200" in info
    assert f"materiality in {six['adaptive_material_pass']}/200" in info
    # the inspector opens on the featured future
    assert any(el.label == "Future ID to inspect" and el.value == 26 for el in at.number_input)
    # turning the LP check off hides the continuous verdict again
    at.toggle(key="featured_lp").set_value(False).run()
    assert "Material objection: Cold store, Bakery" not in _texts(at)


def test_featured_explanation_and_chart_follow_the_selected_threshold():
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page("app_pages/split.py").run()
    at.slider(key="input_rel").set_value(4.25)
    at.slider(key="input_abs").set_value(800).run()
    at.switch_page("app_pages/featured.py").run()
    assert not at.exception
    text = _texts(at)
    assert "Continuous LP passes" in text and "partly repriced deal passes" in text
    assert "4.25% of the group's cost" in text
    table = next(df.value for df in at.dataframe if "Cutoff, LP (€/yr)" in df.value)
    cold = table.loc[table["Firm"] == "Cold store"].iloc[0]
    chart = next(json.loads(chart.proto.spec) for chart in at.get("plotly_chart")
                 if "Paid above standalone cost" in chart.proto.spec)
    markers = next(trace for trace in chart["data"] if trace.get("mode") == "markers+text")
    for method, cutoff in zip(("LP", "screening"), markers["x"], strict=True):
        expected = 0.0425 * cold[f"Standalone, {method} (€/yr)"]
        assert cutoff == pytest.approx(expected)
        assert cold[f"Cutoff, {method} (€/yr)"] == pytest.approx(expected)


def test_sidebar_route_reselects_the_featured_design_after_the_seed_changed():
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page("app_pages/stress.py").run()
    at.number_input(key="requested_seed").set_value(43).run()
    assert "Material objection: Cold store, Bakery" not in _texts(at)
    at.switch_page("app_pages/featured.py").run()
    assert not at.exception
    assert at.query_params.get("case") in ("26", ["26"])
    assert at.number_input(key="requested_seed").value == 42
    assert at.selectbox(key="requested_futures").value == 200
    assert "Material objection: Cold store, Bakery" in _texts(at)
    assert any(el.label == "Future ID to inspect" and el.value == 26 for el in at.number_input)


def test_featured_route_is_hidden_once_the_park_is_edited():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.session_state["assumptions"] = at.session_state["assumptions"].updated(battery_cost_kwh=150.0)
    at.run()
    assert not at.exception
    assert "Featured decision" not in _texts(at)


def test_featured_case_hidden_for_an_edited_design():
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page("app_pages/stress.py").run()
    at.number_input(key="requested_seed").set_value(43).run()
    assert not at.exception
    assert "Featured decision" not in _texts(at)


def test_method_page_benchmarks_match_json():
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page("app_pages/method.py").run()
    assert not at.exception
    verification = json.loads((ROOT / "data/verification.json").read_text(encoding="utf-8"))["all"]
    html = " ".join(el.value for el in at.markdown)
    assert f"{verification['gap_same_battery_pct']:.2f}%" in html
    assert f"{verification['gap_optimal_pct']:.2f}%" in html
