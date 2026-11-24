"""Failure paths and diagnostics in the interface: no traceback reaches the visitor."""

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from hubcore import costs
from hubcore.lp_reference import SolverError
from hubcore.park import Firm, Park
from ui import state

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _fail(*args, **kwargs):
    raise SolverError("LP not solved: simulated failure")


def test_failed_dispatch_lp_is_reported_on_build(monkeypatch):
    monkeypatch.setattr(state, "dispatch", _fail)
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page("app_pages/build.py").run()
    at.toggle(key="show_dispatch").set_value(True).run()
    assert not at.exception
    assert any("reference LP could not be solved" in el.value for el in at.error)


def test_failed_future_repricing_is_reported_on_stress(monkeypatch):
    monkeypatch.setattr(state, "reference_future", _fail)
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page("app_pages/stress.py").run()
    check = next(el for el in at.toggle if el.label == "Check this future with continuous LP costs")
    check.set_value(True).run()
    assert not at.exception
    assert any("continuous LP check could not be completed" in el.value for el in at.error)


def test_battery_search_limit_is_flagged_on_build(monkeypatch):
    """With the upward continuation switched off, a binding right ends at the search limit."""
    monkeypatch.setattr(costs, "BATTERY_EXTENSION_PASSES", 0)
    park = Park("Binding right (test)", (Firm("Gym", "evening", 2438, "process", 262),
                                         Firm("Office", "office", 1953, "ev_workplace_day", 536, 488.0)))
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.session_state["park"] = park
    at.switch_page("app_pages/build.py").run()
    assert not at.exception
    assert any("largest size the fast search tried (Office)" in el.value for el in at.warning)


def test_ensemble_level_search_limit_warning():
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page("app_pages/stress.py").run()
    for ens in at.session_state["ensembles"].values():
        ens.battery_limit_groups = np.r_[1, np.zeros(ens.n_futures - 1, dtype=int)]
    at.run()
    assert not at.exception
    assert any("In 1 of 200 futures" in el.value for el in at.warning)
