"""Plotly figure builders with one consistent, accessible visual system.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------------------- tokens -----
MIST = "#F3F7FB"
WHITE = "#FFFFFF"
POWDER = "#DCE8F0"
INK = "#23384A"
SLATE = "#526778"
STEEL = "#376580"
AZURE = "#286AD6"
TEAL = "#38746E"
CLAY = "#A24E45"
GOLD = "#E9C66A"
GRID = "#E1E9F0"          # recessive gridlines
AXIS = "#768B9D"          # axis lines and neutral-bar outlines (>= 3:1 on white and mist)
NEUTRAL_BAR = "#B9C7D3"   # non-essential comparison bars, always text-labeled


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


AZURE_FILL = rgba(AZURE, 0.12)   # selected rows, areas and boxes
TEAL_FILL = rgba(TEAL, 0.12)
CLAY_FILL = rgba(CLAY, 0.12)

# Firm identity: fixed order, muted hues, OKLCH L 0.47-0.63, chroma >= 0.10.
# Validated with the data-viz palette checks (adjacent CVD dE >= 8.8, normal-vision
# dE >= 21, contrast >= 3:1 on white and on mist).
SERIES = ["#21689C", "#AC7C26", "#725294", "#50955C", "#9D4E6F", "#0C8FA7", "#874814", "#AA72AD"]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "triangle-down", "star", "hexagon"]
# Ordered values: one steel hue, light to dark (light end >= 2:1 on white).
BLUE_RAMP = ["#9CB5C8", "#7F9DB2", "#63859D", "#486E88", "#2C5774", "#0D4160", "#002F4F"]
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

# Backwards-compatible aliases used by earlier modules.
GOOD = TEAL
CRITICAL = CLAY
SURFACE = WHITE
PAGE = MIST
INK_2 = SLATE
MUTED = SLATE


# --------------------------------------------------------- contrast math ---
def _channel(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    """WCAG 2.2 relative luminance of an sRGB hex color."""
    h = hex_color.lstrip("#")
    r, g, b = (_channel(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two opaque hex colors."""
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def blend(fg: str, bg: str, alpha: float) -> str:
    """Opaque hex color of ``fg`` at ``alpha`` over ``bg`` (for rendered contrast)."""
    f = [int(fg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(bg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(alpha * x + (1 - alpha) * y):02X}" for x, y in zip(f, b))


def text_on(bg: str) -> str:
    """Ink or white, whichever reads better on ``bg``."""
    return INK if contrast(INK, bg) >= contrast(WHITE, bg) else WHITE


def ramp_color(value: float, lo: float, hi: float) -> str:
    """Color of ``value`` on the steel ramp (piecewise linear in sRGB)."""
    if not np.isfinite(value) or hi <= lo:
        return BLUE_RAMP[0]
    t = min(max((value - lo) / (hi - lo), 0.0), 1.0) * (len(BLUE_RAMP) - 1)
    k = min(int(np.floor(t)), len(BLUE_RAMP) - 2)
    return blend(BLUE_RAMP[k + 1], BLUE_RAMP[k], t - k)


def firm_color(i: int) -> str:
    return SERIES[i % len(SERIES)]


def firm_symbol(i: int) -> str:
    return SYMBOLS[i % len(SYMBOLS)]


def _layout(fig: go.Figure, height: int = 360, **kw) -> go.Figure:
    margin = kw.pop("margin", dict(l=8, r=12, t=40, b=8))
    fig.update_layout(
        height=height,
        margin=margin,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family=FONT, size=13, color=INK),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=STEEL, font=dict(family=FONT, color=INK, size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None,
                    font=dict(color=INK, size=12), bgcolor="rgba(0,0,0,0)"),
        separators=".,",
        **kw,
    )
    axes = dict(automargin=True, gridcolor=GRID, linecolor=AXIS, zeroline=False,
                tickfont=dict(color=SLATE), title_font=dict(color=INK))
    fig.update_xaxes(**axes)
    fig.update_yaxes(**axes)
    return fig


def eur(v: float) -> str:
    """Compact euro amount for KPIs and labels: €950, €1.2k, €68k, €1.25M.
    """
    if not np.isfinite(v):
        return "-"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if round(v / 1e6, 2) >= 1:
        return f"{sign}€{v / 1e6:.2f}M"
    if round(v / 1e3, 1) >= 10:
        return f"{sign}€{v / 1e3:.0f}k"
    if round(v) >= 1e3:
        return f"{sign}€{v / 1e3:.1f}k"
    return f"{sign if round(v) else ''}€{v:.0f}"


def eur2(v: float) -> str:
    """Exact euro amount, e.g. €605.44, €500 or -€1,234.50 (cents omitted for whole euros)."""
    if not np.isfinite(v):
        return "-"
    sign = "-" if round(v, 2) < 0 else ""
    v = abs(v)
    if abs(v - round(v)) < 0.005:
        return f"{sign}€{round(v):,.0f}"
    return f"{sign}€{v:,.2f}"


# ---------------------------------------------------------------------------
# Build page
# ---------------------------------------------------------------------------

def peak_day(loads_day: np.ndarray, names: list[str], contract: float, capacity_right: float,
             shaved_level: float | None) -> go.Figure:
    """Stacked hourly columns per firm on the hub's peak day, with contract lines."""
    hours = np.arange(24)
    fig = go.Figure()
    for i, name in enumerate(names):
        fig.add_bar(x=hours, y=loads_day[i], name=name, marker_color=firm_color(i),
                    marker_line=dict(color=WHITE, width=0.6),
                    hovertemplate=f"{name}<br>%{{x}}:00  %{{y:,.0f}} kW<extra></extra>")
    xs = [-0.5, 23.5]
    fig.add_scatter(x=xs, y=[capacity_right] * 2, mode="lines",
                    name=f"Capacity right {capacity_right:,.0f} kW",
                    line=dict(color=SLATE, width=2, dash="dot"), hoverinfo="skip")
    fig.add_scatter(x=xs, y=[contract] * 2, mode="lines", name=f"Group contract {contract:,.0f} kW",
                    line=dict(color=INK, width=2.5), hoverinfo="skip")
    if shaved_level is not None and abs(shaved_level - contract) > 0.01 * contract:
        fig.add_scatter(x=xs, y=[shaved_level] * 2, mode="lines",
                        name=f"Battery target {shaved_level:,.0f} kW",
                        line=dict(color=AZURE, width=2, dash="dash"), hoverinfo="skip")
    fig.update_layout(barmode="stack", bargap=0.1, legend_traceorder="normal")
    fig.update_xaxes(title="Hour of day", dtick=3, range=[-0.6, 23.6])
    fig.update_yaxes(title="kW")
    return _layout(fig, 440)


def diversity_bars(sum_of_peaks: float, coincident: float, right: float, contract: float) -> go.Figure:
    labels = ["Sum of individual peaks", "Peak of the combined load", "Group capacity right",
              "Chosen group contract"]
    vals = [sum_of_peaks, coincident, right, contract]
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h", marker_color=[STEEL, STEEL, NEUTRAL_BAR, INK],
        marker_line=dict(color=[STEEL, STEEL, AXIS, INK], width=1),
        text=[f"{v:,.0f} kW" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK), hovertemplate="%{y}: %{x:,.0f} kW<extra></extra>", width=0.5,
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK))
    fig.update_xaxes(title="kW", range=[0, max(vals) * 1.42])
    return _layout(fig, 260, showlegend=False, margin=dict(l=8, r=24, t=16, b=8))


def waterfall(standalone_total: float, components_hub: dict[str, float],
              components_alone: dict[str, float]) -> go.Figure:
    """Bridge from the sum of stand-alone costs to the hub's cost."""
    order = [("c_contract", "Contracted capacity"), ("c_peak", "Monthly peaks"),
             ("c_unserved", "Demand above contract"), ("c_battery", "Battery"),
             ("c_overhead", "Hub overhead")]
    x = ["Firms on their own"]
    y = [standalone_total]
    measure = ["absolute"]
    for key, label in order:
        d = components_hub.get(key, 0.0) - components_alone.get(key, 0.0)
        if abs(d) < 1:
            continue
        x.append(label)
        y.append(d)
        measure.append("relative")
    x.append("Energy hub")
    y.append(0)
    measure.append("total")
    fig = go.Figure(go.Waterfall(
        x=x, y=y, measure=measure, connector=dict(line=dict(color=AXIS, width=1)),
        increasing=dict(marker=dict(color=CLAY)), decreasing=dict(marker=dict(color=TEAL)),
        totals=dict(marker=dict(color=STEEL)),
        text=[(("+" if v > 0 else "") + eur(v)) if m == "relative" else eur(v) if m == "absolute" else ""
              for v, m in zip(y, measure)],
        textposition="outside", textfont=dict(color=INK),
        hovertemplate="%{x}: %{y:,.0f} €/yr<extra></extra>",
    ))
    fig.update_yaxes(title="€ per year")
    return _layout(fig, 370, showlegend=False)


# ---------------------------------------------------------------------------
# Split page
# ---------------------------------------------------------------------------

def allocation_dots(savings_pct: dict[str, np.ndarray], rule_labels: dict[str, str],
                    names: list[str], stable: dict[str, bool]) -> go.Figure:
    """Savings (% of stand-alone cost) per firm (color and symbol) and rule (row).
    """
    rules = list(savings_pct)
    n = len(names)
    ylabels = [f"{'✓ passes' if stable[r] else '✗ objection'} · {rule_labels[r]}" for r in rules]
    offsets = (np.arange(n) - (n - 1) / 2) * (0.5 / max(n, 1))
    fig = go.Figure()
    for k in range(len(rules)):
        if k % 2 == 0:
            fig.add_hrect(y0=k - 0.5, y1=k + 0.5, fillcolor=MIST, line_width=0, layer="below")
    for i, name in enumerate(names):
        fig.add_scatter(
            x=[savings_pct[r][i] for r in rules], y=[k + offsets[i] for k in range(len(rules))],
            mode="markers", name=name, customdata=ylabels,
            marker=dict(size=11, color=firm_color(i), symbol=firm_symbol(i),
                        line=dict(color=WHITE, width=1)),
            hovertemplate=f"{name}<br>%{{customdata}}<br>saves %{{x:.1f}}%<extra></extra>",
        )
    fig.add_vline(x=0, line=dict(color=INK, width=1.5))
    fig.update_yaxes(autorange="reversed", tickvals=list(range(len(rules))), ticktext=ylabels,
                     tickfont=dict(color=INK, size=13), showgrid=False)
    fig.update_xaxes(title="Savings vs standalone (%)", ticksuffix="%")
    return _layout(fig, 90 + 56 * len(rules))


def excess_bars(items: list[tuple[str, float, float]]) -> go.Figure:
    """items: (coalition label, excess €, threshold €) sorted descending."""
    labels = [t[0] for t in items][::-1]
    vals = [t[1] for t in items][::-1]
    material = [t[1] > t[2] for t in items][::-1]
    colors = [CLAY if m else NEUTRAL_BAR for m in material]
    text = [f"{eur(v)} ⚠" if m else eur(v) for v, m in zip(vals, material)]
    fig = go.Figure(go.Bar(y=labels, x=vals, orientation="h", marker_color=colors, width=0.55,
                           marker_line=dict(color=[CLAY if m else AXIS for m in material], width=1),
                           text=text, textposition="outside", cliponaxis=False,
                           textfont=dict(color=INK),
                           hovertemplate="%{y}<br>excess %{x:,.0f} €/yr<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    span = max(abs(v) for v in vals) if vals else 1.0
    lo = min(0.0, min(vals)) * 1.35 if vals else 0.0
    fig.update_xaxes(title="Excess (€/yr)", tickangle=0, nticks=5,
                     range=[lo - 0.05 * span, max(0.0, max(vals)) * 1.45 + 0.1 * span])
    fig.update_yaxes(tickfont=dict(color=INK))
    return _layout(fig, 70 + 34 * len(items), showlegend=False, margin=dict(l=8, r=16, t=16, b=8))


# ---------------------------------------------------------------------------
# Stress-test page
# ---------------------------------------------------------------------------

def robustness_heat(table, value: str, rule_order: list[str], rule_labels: dict[str, str],
                    clause_labels: dict[str, str], fmt: str = "{:.1%}", reverse: bool = False,
                    zmax: float | None = None) -> go.Figure:
    """Rule x clause table on the steel ramp; values are written in ink on white labels."""
    clauses = list(clause_labels)
    z = np.array([[table.loc[(r, c), value] for c in clauses] for r in rule_order], dtype=float)
    top = zmax if zmax is not None else (1.0 if fmt.endswith("%}") else (np.nanmax(z) or 1.0))
    scale = BLUE_RAMP[::-1] if reverse else BLUE_RAMP
    xs = [clause_labels[c] for c in clauses]
    ys = [rule_labels[r] for r in rule_order]
    fig = go.Figure(go.Heatmap(
        z=z, x=xs, y=ys, text=[[fmt.format(v) for v in row] for row in z],
        colorscale=[[k / (len(scale) - 1), col] for k, col in enumerate(scale)],
        zmin=0, zmax=top, showscale=False, xgap=3, ygap=3,
        hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>",
    ))

    for r, row in enumerate(z):
        for c, v in enumerate(row):
            fig.add_annotation(x=xs[c], y=ys[r], text=f"<b>{fmt.format(v)}</b>", showarrow=False,
                               font=dict(size=14, color=INK, family=FONT), bgcolor=WHITE,
                               bordercolor=WHITE, borderpad=3, opacity=1)
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK), showgrid=False)
    fig.update_xaxes(side="top", tickfont=dict(color=INK), showgrid=False)
    return _layout(fig, 64 + 48 * len(rule_order), margin=dict(l=8, r=8, t=52, b=8))


def futures_scatter(x: np.ndarray, y: np.ndarray, fail: np.ndarray, xlabel: str, ylabel: str,
                    box: tuple[tuple[float, float], tuple[float, float]] | None,
                    fail_label: str, ok_label: str, highlight: int | None = None) -> go.Figure:
    fig = go.Figure()
    if box is not None:
        (x0, x1), (y0, y1) = box
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, layer="below",
                      line=dict(color=AZURE, width=2), fillcolor=AZURE_FILL)
    ids = np.arange(len(x))
    fig.add_scatter(x=x[~fail], y=y[~fail], mode="markers", name=f"✓ {ok_label}", customdata=ids[~fail],
                    marker=dict(size=9, color=TEAL, symbol="circle", line=dict(color=WHITE, width=1.5)),
                    hovertemplate=f"Future %{{customdata}}<br>{xlabel}: %{{x:.3g}}<br>{ylabel}: %{{y:.3g}}"
                                  f"<extra>✓ {ok_label}</extra>")
    fig.add_scatter(x=x[fail], y=y[fail], mode="markers", name=f"✗ {fail_label}", customdata=ids[fail],
                    marker=dict(size=10, color=CLAY, symbol="x", line=dict(color=WHITE, width=1)),
                    hovertemplate=f"Future %{{customdata}}<br>{xlabel}: %{{x:.3g}}<br>{ylabel}: %{{y:.3g}}"
                                  f"<extra>✗ {fail_label}</extra>")
    if highlight is not None and 0 <= highlight < len(x):
        fig.add_scatter(x=[x[highlight]], y=[y[highlight]], mode="markers+text", name=f"Future {highlight}",
                        text=[f"Future {highlight}"], textposition="top center", textfont=dict(color=INK),
                        marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=AZURE, width=3)),
                        hoverinfo="skip")
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return _layout(fig, 420)


def prim_trajectory(coverage: list[float], density: list[float], selected: int) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=coverage, y=density, mode="lines+markers", name="Peeling steps",
                    line=dict(color=STEEL, width=2), marker=dict(size=8, color=STEEL,
                                                                line=dict(color=WHITE, width=1.5)),
                    customdata=np.arange(len(coverage)),
                    hovertemplate=("Step %{customdata}<br>coverage %{x:.0%}<br>"
                                   "density %{y:.0%}<extra></extra>"))
    fig.add_scatter(x=[coverage[selected]], y=[density[selected]], mode="markers", name="Selected box",
                    marker=dict(size=16, color=AZURE, symbol="diamond", line=dict(color=INK, width=1.5)),
                    hovertemplate="Selected step<br>coverage %{x:.0%}<br>density %{y:.0%}<extra></extra>")
    fig.update_xaxes(title="Failure coverage", tickformat=".0%", range=[0, 1.05])
    fig.update_yaxes(title="Failure density", tickformat=".0%", range=[0, 1.05])
    return _layout(fig, 340, showlegend=False)


def exit_bars(names: list[str], shares: np.ndarray, firm_index: list[int] | None = None) -> go.Figure:
    idx = list(range(len(names))) if firm_index is None else list(firm_index)
    fig = go.Figure(go.Bar(
        y=names[::-1], x=list(shares[::-1]), orientation="h", width=0.55,
        marker_color=[firm_color(i) for i in idx][::-1],
        text=[f"{s:.1%}" for s in shares[::-1]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK),
        hovertemplate="%{y}: materially worse off than alone in %{x:.1%} of futures<extra></extra>",
    ))
    fig.update_yaxes(tickfont=dict(color=INK))
    fig.update_xaxes(title="Futures with a material saving alone",
                     tickformat=".0%", range=[0, max(1.0, float(shares.max()) * 1.15)])
    return _layout(fig, 64 + 36 * len(names), showlegend=False)


def membership_bars(labels: list[str], shares: list[float], baseline: float) -> go.Figure:
    """Best-deal robustness for hubs with one firm fewer, against the full hub."""
    fig = go.Figure(go.Bar(
        y=labels[::-1], x=shares[::-1], orientation="h", width=0.55, marker_color=STEEL,
        text=[f"{v:.1%}" for v in shares[::-1]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK),
        hovertemplate="%{y}: best deal holds in %{x:.1%} of futures<extra></extra>",
    ))
    fig.add_vline(x=baseline, line=dict(color=INK, width=2, dash="dot"),
                  annotation_text=f"All selected: {baseline:.1%}", annotation_position="top",
                  annotation_font_color=INK)
    fig.update_yaxes(tickfont=dict(color=INK))
    fig.update_xaxes(title="Futures passing the materiality test", tickformat=".0%", range=[0, 1.05])
    return _layout(fig, 84 + 36 * len(labels), showlegend=False, margin=dict(l=8, r=16, t=34, b=8))


def objection_meter(rows: list[dict]) -> go.Figure:
    """Compare a member's excess with the cutoff derived from each method's standalone cost."""
    labels = [f"<b>{r['label']}</b><br>{'⚠ above cutoff' if r['material'] else '✓ within cutoff'}"
              for r in rows][::-1]
    vals = [r["excess"] for r in rows][::-1]
    cutoffs = [r["cutoff"] for r in rows][::-1]
    mat = [r["material"] for r in rows][::-1]
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h", width=0.55, marker_color=[CLAY if m else TEAL for m in mat],
        # Amounts sit at the start of each bar so the cutoff line never crosses them.
        text=[eur2(v) for v in vals], textposition="inside", insidetextanchor="start",
        textfont=dict(color=WHITE, size=14), cliponaxis=False,
        hovertemplate="pays %{x:,.2f} €/yr above its standalone cost<extra></extra>",
    ))
    if len(set(cutoffs)) == 1:
        fig.add_vline(x=cutoffs[0], line=dict(color=INK, width=2, dash="dash"),
                      annotation_text=f"Cutoff {eur2(cutoffs[0])}", annotation_position="top right",
                      annotation_font=dict(color=INK, size=12))
    else:
        fig.add_scatter(y=labels, x=cutoffs, mode="markers+text",
                        marker=dict(symbol="line-ns", size=22, line=dict(width=2, color=INK)),
                        text=[f"Cutoff {eur2(t)}" for t in cutoffs], textposition="middle right",
                        textfont=dict(color=INK), cliponaxis=False, hoverinfo="skip")
    top = max(max(vals), max(cutoffs))
    fig.update_xaxes(title="Paid above standalone cost (€/yr)", range=[0, top * 1.25], tickprefix="€")
    fig.update_yaxes(tickfont=dict(color=INK, size=13))
    return _layout(fig, 110 + 60 * len(rows), showlegend=False, margin=dict(l=8, r=16, t=34, b=8))


def dispatch_day(load: np.ndarray, result, hours: slice) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=np.arange(24), y=result.imports[hours], name="Grid import", mode="lines",
                    line=dict(color=STEEL, width=2.5), fill="tozeroy", fillcolor=AZURE_FILL)
    for name, series, color, dash in [
        ("Requested load", load, INK, "dot"),
        ("Charging", result.charge[hours], SERIES[1], "dash"),
        ("Discharging", result.discharge[hours], SERIES[2], "dashdot"),
        ("Shortfall", result.unserved[hours], CLAY, "solid"),
    ]:
        fig.add_scatter(x=np.arange(24), y=series, name=name, mode="lines",
                        line=dict(color=color, dash=dash, width=2))
    fig.add_hline(y=result.contract_kw, line=dict(color=SLATE, dash="dash", width=1.5),
                  annotation_text="Contract (kW)", annotation_font_color=INK)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title="Hour of day", dtick=3)
    fig.update_yaxes(title="Power (kW)")
    return _layout(fig, 380)
