"""Headless smoke tests: every page renders without exceptions."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
PAGES = ["app_pages/home.py", "app_pages/build.py", "app_pages/split.py",
         "app_pages/stress.py", "app_pages/method.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.switch_page(page)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) >= 1


def test_uncached_selection_hides_previous_results():
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page('app_pages/stress.py').run()
    assert not at.exception
    assert any('200 futures ready' in v.value for v in at.success)
    at.number_input(key='requested_seed').set_value(51).run()
    assert not at.exception
    assert any('Run stress test' in b.label for b in at.button)
    assert not any('Robustness of each deal' in h.value for h in at.subheader)
    assert not any('200 futures ready' in v.value for v in at.success)


def test_materiality_updates_in_same_rerun_and_reference_dispatch():
    at=AppTest.from_file(APP, default_timeout=180).run().switch_page('app_pages/split.py').run()
    at.slider(key='input_abs').set_value(0)
    at.slider(key='input_rel').set_value(0.0).run()
    assert not at.exception and at.session_state['abs_tol']==0
    assert any('Threshold: max(€0, 0.00%' in el.value for el in at.markdown)
    [b for b in at.button if b.label=='Repair my proposal'][0].click().run()
    assert not at.exception and any('L1 distance' in el.value for el in at.success)
    at.switch_page('app_pages/build.py').run()
    at.toggle(key='show_dispatch').set_value(True).run()
    assert not at.exception
    assert any('residual shortfall' in el.value for el in at.markdown)


def test_future_inspection_and_export_survive_uniform_outcomes():
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page("app_pages/stress.py").run()
    policy = next(el for el in at.selectbox if el.label == "Deal to inspect")
    policy.set_value(("capacity", "fixed")).run()
    assert not at.exception
    assert any("fails in every sampled future" in el.value for el in at.warning)
    assert any(el.label == "Download these futures (CSV)" for el in at.get("download_button"))

    # A deliberately uniform design checks the other early-return branch of the UI.
    for ens in at.session_state["ensembles"].values():
        ens.costs[:] = ens.nominal_costs
        ens.kwh[:] = ens.nominal_kwh
    at.session_state["policy_tables"].clear()
    policy = next(el for el in at.selectbox if el.label == "Deal to inspect")
    policy.set_value(("leastcore", "adaptive")).run()
    assert not at.exception
    assert any("never fails" in el.value for el in at.success)
    assert any(el.label == "Download these futures (CSV)" for el in at.get("download_button"))
    assert any("passes" in el.value and "outside cost" in el.value for el in at.markdown)


def test_full_ensemble_and_selected_future_use_matching_reference_costs():
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page("app_pages/stress.py").run()
    assert any("core exists in 32/200" in el.value for el in at.info)
    future = next(el for el in at.number_input if el.label == "Future ID to inspect")
    future.set_value(26).run()
    check = next(el for el in at.toggle if el.label == "Check this future with continuous LP costs")
    check.set_value(True).run()
    assert not at.exception
    assert any("Future 26 · passes" in el.value for el in at.markdown)
    assert any("Continuous LP · future 26 fails" in el.value for el in at.markdown)
    at.number_input(key="requested_seed").set_value(77).run()
    assert not at.exception
    assert not any("core exists in 32/200" in el.value for el in at.info)
