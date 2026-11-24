"""Visual system for the app: CSS, KPI cards, status badges, evidence chips and case panels.
"""

from __future__ import annotations

import html

import streamlit as st

from ui.charts import (
    AZURE,
    AZURE_FILL,
    CLAY,
    CLAY_FILL,
    GOLD,
    INK,
    MIST,
    POWDER,
    SLATE,
    STEEL,
    TEAL,
    TEAL_FILL,
    WHITE,
)

BORDER = "#C3D2DE"        # card outlines (decorative)
CONTROL = "#768B9D"       # widget outlines: >= 3:1 against white and mist

CSS = f"""
<style>
:root {{
  --mist:{MIST}; --white:{WHITE}; --powder:{POWDER}; --ink:{INK}; --slate:{SLATE}; --steel:{STEEL};
  --azure:{AZURE}; --azure-fill:{AZURE_FILL}; --teal:{TEAL}; --teal-fill:{TEAL_FILL};
  --clay:{CLAY}; --clay-fill:{CLAY_FILL}; --gold:{GOLD}; --border:{BORDER}; --control:{CONTROL};
}}
html, body, [data-testid="stAppViewContainer"] {{ color: var(--ink); }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; color: var(--ink); }}
h1 {{ font-weight: 700; }}
h2, h3 {{ color: var(--steel); }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ color: var(--ink); }}
[data-testid^="stBaseButton-primary"] p, [data-testid^="stBaseButton-primary"] span {{ color: var(--white) !important; }}
[data-testid="stSkipToContent"] {{ color: var(--ink) !important; background: var(--white) !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--slate); }}
a {{ color: var(--azure); }}

/* Keyboard focus: opaque azure ring on every interactive element */
a:focus-visible, button:focus-visible, [role="button"]:focus-visible, [role="tab"]:focus-visible,
input:focus-visible, textarea:focus-visible, select:focus-visible, summary:focus-visible,
[role="switch"]:focus-visible, [role="slider"]:focus-visible, [role="checkbox"]:focus-visible {{
  outline: 3px solid var(--azure) !important; outline-offset: 2px !important; border-radius: 6px;
}}
[data-testid="stMain"]:focus-visible, canvas:focus-visible, [tabindex]:focus-visible {{
  outline: 3px solid var(--azure) !important; outline-offset: -3px !important;
}}
[data-testid="stSidebarNav"] [data-testid="stIconMaterial"], [data-testid="stSidebar"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{
  color: var(--ink); opacity: 1;
}}

/* Cards, KPIs, chips */
.hc-eyebrow {{ text-transform: uppercase; font-size: .78rem; letter-spacing: .08em;
  color: var(--steel); font-weight: 700; margin-bottom: .2rem; }}
.hc-lead {{ font-size: 1.12rem; color: var(--slate); line-height: 1.55; max-width: 60rem; }}
.hc-card {{ background: var(--white); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 16px; height: 100%; min-height: 124px; box-shadow: 0 1px 2px rgba(35,56,74,.05); }}
.hc-kpi-label {{ font-size: .84rem; color: var(--slate); margin-bottom: 2px; }}
.hc-kpi-value {{ font-size: 1.6rem; font-weight: 700; color: var(--steel); line-height: 1.2;
  overflow-wrap: anywhere; }}
.hc-kpi-note {{ font-size: .82rem; color: var(--slate); margin-top: 2px; }}
.hc-chip {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
  border: 1px solid var(--border); color: var(--ink); background: var(--powder); margin: 0 4px 4px 0; }}
.hc-chip b {{ color: var(--ink); font-weight: 700; }}
.hc-status {{ display:inline-flex; gap:8px; align-items:center; font-weight:650; font-size:.95rem;
  padding: 6px 12px; border-radius: 8px; color: var(--ink); }}
.hc-ok {{ background: var(--teal-fill); }} .hc-ok .hc-icon {{ background: var(--teal); }}
.hc-bad {{ background: var(--clay-fill); }} .hc-bad .hc-icon-warn {{ color: var(--clay); }}
.hc-icon {{ display:inline-flex; align-items:center; justify-content:center; width: 1.25rem; height: 1.25rem;
  border-radius: 50%; color: var(--white); font-size: .78rem; font-weight: 800; flex: none; }}
.hc-icon-warn {{ font-size: 1.15rem; line-height: 1; flex: none; }}
.hc-step {{ font-size: .8rem; font-weight: 800; color: var(--steel); letter-spacing: .04em; }}
.hc-callout {{ border-left: 4px solid var(--steel); padding: 10px 14px; background: var(--azure-fill);
  border-radius: 0 8px 8px 0; color: var(--ink); margin: 6px 0 14px 0; }}
.hc-small {{ font-size: .88rem; color: var(--slate); }}
.hc-highlight {{ display:inline-block; background: var(--gold); color: var(--ink); font-weight: 700;
  font-size: .76rem; letter-spacing: .05em; text-transform: uppercase; padding: 2px 9px; border-radius: 6px; }}

/* Featured case (matches the palette specimen) */
.hc-case-title {{ font-size: 1.55rem; font-weight: 700; color: var(--ink); margin: 0 0 2px 0; }}
.hc-case-sub {{ color: var(--slate); margin: 0 0 12px 0; }}
.hc-badges {{ display:flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 12px 0; }}
.hc-strip {{ background: var(--azure-fill); border-radius: 8px; padding: 10px 14px; color: var(--ink); }}
.hc-why {{ margin: .7rem 0 .4rem 0; }}
.hc-method {{ background: var(--white); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; height: 100%; }}
.hc-method h4 {{ margin: 0 0 4px 0; font-size: 1.05rem; color: var(--steel); }}
.hc-method .hc-how {{ font-size: .84rem; color: var(--slate); margin: 0 0 10px 0; }}
.hc-rows {{ margin-top: 10px; font-size: .9rem; }}
.hc-row {{ display: flex; justify-content: space-between; gap: 12px; padding: 5px 0;
  border-bottom: 1px solid var(--powder); color: var(--ink); }}
.hc-row span:last-child {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.hc-row.hc-key {{ font-weight: 700; }}
.hc-pending {{ border: 1px dashed var(--control); border-radius: 12px; padding: 14px 16px; color: var(--slate);
  background: var(--mist); height: 100%; }}

/* Streamlit chrome */
[data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
[data-testid="stExpander"] details {{ border-color: var(--border); background: var(--white); }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--border); }}
footer {{ visibility: hidden; }}

@media (max-width: 640px) {{
  .block-container {{ padding-top: 4rem; padding-left: 1rem; padding-right: 1rem; }}
  .hc-kpi-value {{ font-size: 1.35rem; }}
  .hc-card {{ min-height: 0; }}
  .hc-case-title {{ font-size: 1.3rem; }}
  .hc-lead {{ font-size: 1.02rem; }}
}}
</style>
"""

EVIDENCE = {
    "Sourced": "Taken from a cited public source",
    "Sourced base": "Base value from a cited public source; range explored",
    "Sourced range": "Range from cited public sources",
    "Proxy": "Stand-in from a related public source",
    "Illustrative": "Plausible value chosen for exploration, not measured",
    "Scenario": "Explored as a deep uncertainty",
    "Synthetic": "Fictional example data",
}


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "") -> str:
    return (f'<div class="hc-card"><div class="hc-kpi-label">{html.escape(label)}</div>'
            f'<div class="hc-kpi-value">{html.escape(value)}</div>'
            f'<div class="hc-kpi-note">{html.escape(note)}</div></div>')


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        col.markdown(kpi(label, value, note), unsafe_allow_html=True)


def chip(status: str, text: str | None = None) -> str:
    tip = html.escape(EVIDENCE.get(status, ""))
    body = f"<b>{html.escape(status)}</b>" + (f" · {html.escape(text)}" if text else "")
    return f'<span class="hc-chip" title="{tip}">{body}</span>'


def badge(ok: bool, text: str) -> str:
    """Pass (teal, check) or objection (clay, warning sign) badge; the word carries the meaning."""
    if ok:
        return (f'<span class="hc-status hc-ok"><span class="hc-icon" aria-hidden="true">✓</span>'
                f'{html.escape(text)}</span>')
    return (f'<span class="hc-status hc-bad"><span class="hc-icon-warn" aria-hidden="true">⚠</span>'
            f'{html.escape(text)}</span>')


def status(ok: bool, ok_text: str, bad_text: str) -> str:
    return badge(ok, ok_text if ok else bad_text)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="hc-eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<div class="hc-callout">{text}</div>', unsafe_allow_html=True)


def highlight(text: str) -> str:
    return f'<span class="hc-highlight">{html.escape(text)}</span>'


def method_card(title: str, how: str, ok: bool, verdict: str, rows: list[tuple[str, str, bool]]) -> str:
    """Result of one pricing method: how it was calculated, the verdict in words, key numbers."""
    body = "".join(
        f'<div class="hc-row{" hc-key" if key else ""}"><span>{html.escape(label)}</span>'
        f'<span>{html.escape(value)}</span></div>' for label, value, key in rows)
    return (f'<div class="hc-method"><h4>{html.escape(title)}</h4><p class="hc-how">{html.escape(how)}</p>'
            f'{badge(ok, verdict)}<div class="hc-rows">{body}</div></div>')


def pending_card(title: str, text: str) -> str:
    return (f'<div class="hc-pending"><b style="color:var(--ink)">{html.escape(title)}</b>'
            f'<p class="hc-small" style="margin:.4rem 0 0 0">{html.escape(text)}</p></div>')
