"""Euro formatting used in KPIs, labels and the featured panels."""

import pytest

from ui.charts import eur, eur2


@pytest.mark.parametrize("value, text", [
    (0, "€0"), (49.9, "€50"), (-356.58, "-€357"), (-0.4, "€0"),
    (999.4, "€999"), (999.6, "€1.0k"), (1_234, "€1.2k"), (9_996, "€10k"),
    (68_441.23, "€68k"), (-27_168, "-€27k"), (994_999, "€995k"), (999_996, "€1.00M"), (1_250_000, "€1.25M"),
    (float("nan"), "-"),
])
def test_compact_amounts(value, text):
    assert eur(value) == text


@pytest.mark.parametrize("value, text", [
    (605.4412, "€605.44"), (500.0, "€500"), (500.001, "€500"), (-1234.5, "-€1,234.50"),
    (-0.001, "€0"), (27_937.94, "€27,937.94"), (float("inf"), "-"),
])
def test_exact_amounts(value, text):
    assert eur2(value) == text
