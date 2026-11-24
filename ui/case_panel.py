"""Featured-decision panels for the Overview and Stress-test pages."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from ui import featured, style
from ui.charts import eur2, objection_meter

STRESS_PAGE = "app_pages/stress.py"
FEATURED_PAGE = "app_pages/featured.py"
CASE_QUERY = {"case": str(featured.FEATURED["future"])}


def open_featured() -> None:
    """Select the featured design (200 futures, seed 42, LP check on) and open future 26.
    """
    st.session_state["requested_futures"] = featured.FEATURED["n"]
    st.session_state["requested_seed"] = featured.FEATURED["seed"]
    st.session_state["featured_lp"] = True
    st.session_state["_featured_link_applied"] = True
    st.session_state["_featured_inspect"] = True
    st.switch_page(STRESS_PAGE, query_params=CASE_QUERY)


def _badges(case: dict, show_lp: bool) -> str:
    s = case["screening"]
    parts = [style.badge(s.passes, "Screening passes" if s.passes else "Screening finds an objection")]
    if show_lp and "continuous" in case:
        c = case["continuous"]
        parts.append(style.badge(c.passes, "Continuous LP passes" if c.passes
                                 else "Continuous LP finds an objection"))
    return f'<div class="hc-badges">{"".join(parts)}</div>'


def _strip(case: dict, show_lp: bool) -> str:
    i, member = case["member_index"], case["member"]
    v = case["continuous"] if show_lp and "continuous" in case else case["screening"]
    excess, cutoff = float(v.excess[i]), float(v.cutoff[i])
    if excess > 0:
        text = (f"{member} pays {eur2(excess)}/year above its standalone cost. "
                f"Its materiality cutoff is {eur2(cutoff)}/year.")
    else:
        text = f"{member} pays {eur2(-excess)}/year less than its standalone cost."
    if v is case["screening"] and show_lp is False:
        text = "Screening prices: " + text
    return f'<div class="hc-strip">{html.escape(text)}</div>'


def why_lines(case: dict) -> list[str]:
    """Attribute the price difference under the selected materiality thresholds."""
    if "continuous" not in case:
        return []
    member, names = case["member"], case["names"]
    s, c, a = case["screening"], case["continuous"], case["attribution"]
    partial_verdict = "passes" if a["passes_only_member_repriced"] else "fails"
    lines = [
        "Both columns apply the same rule (adaptive least-core) to the same future. Only the prices of "
        "the alternatives differ.",
        f"The continuous LP finds a standalone design for the {member.lower()} that costs "
        f"{eur2(a['standalone_continuous'])}/year, {eur2(-a['standalone_change'])} less than the screening "
        f"grid ({eur2(a['standalone_screening'])}). A cheaper outside option means a stronger objection.",
        f"Repricing only that one standalone cost lifts the least-core's largest excess from "
        f"{eur2(s.eps)} to {eur2(a['eps_only_member_repriced'])}. This partly repriced deal "
        f"{partial_verdict} the selected materiality test. Repricing the remaining coalitions raises "
        f"the largest excess to {eur2(c.eps)}.",
    ]
    others = [n for n in featured.ties(c, names) if n != member]
    if others:
        lines.append(
            f"The least-core spreads the largest objection evenly, so {' and '.join(others)} also "
            f"pay{'s' if len(others) == 1 else ''} {eur2(float(c.excess.max()))}/year above standalone. "
            f"Under screening both paid {eur2(float(s.excess.max()))} above standalone.")
    lines.append("Neither price set is market data: both describe the same synthetic park under "
                 "hypothetical capacity rights.")
    return lines


def overview_card(case: dict) -> None:
    """Show the featured case on the Overview page."""
    with st.container(border=True):
        top, action = st.columns([3, 1.4], vertical_alignment="center")
        top.markdown(
            f'{style.highlight("Featured decision")}'
            f'<div class="hc-case-title" style="margin-top:6px">Future {case["future"]}</div>'
            f'<p class="hc-case-sub">Six firms · adaptive least-core allocation · '
            f'200 futures, seed 42</p>', unsafe_allow_html=True)
        if action.button("Open future 26", type="primary", icon=":material/arrow_forward:",
                         key="open_featured", width="stretch"):
            open_featured()
        st.markdown(_badges(case, True) + _strip(case, True), unsafe_allow_html=True)
        lines = why_lines(case)
        if lines:
            st.markdown(f'<p class="hc-small hc-why"><b style="color:var(--ink)">'
                        f'Why the prices differ.</b> {html.escape(lines[1])}</p>', unsafe_allow_html=True)


def stress_panel(case: dict, default_lp: bool) -> None:
    """Full side-by-side comparison on the Stress-test page."""
    i, member, names = case["member_index"], case["member"], case["names"]
    with st.container(border=True, key="featured_case"):
        head, toggle = st.columns([3, 1.3], vertical_alignment="center")
        head.markdown(
            f'{style.highlight("Featured decision")}'
            f'<div class="hc-case-title" style="margin-top:6px">Future {case["future"]}</div>'
            f'<p class="hc-case-sub">Six firms · adaptive least-core allocation · '
            f'one of the 200 seed-42 futures</p>', unsafe_allow_html=True)
        st.session_state.setdefault("featured_lp", default_lp)
        show_lp = toggle.toggle("Check continuous LP", key="featured_lp",
                                help="Reprice all 63 coalitions with continuous contract and battery sizes. "
                                     "Bundled for this design; solved on demand otherwise.")
        if show_lp and "continuous" not in case:
            from ui import state
            try:
                with st.spinner("Repricing all 63 coalitions with continuous LPs…"):
                    case = state.featured_case(with_lp=True) or case
            except state.SOLVER_ERRORS as exc:
                st.error(f"The continuous LP repricing could not be completed ({exc}).")
        st.markdown(_badges(case, show_lp) + _strip(case, show_lp), unsafe_allow_html=True)
        st.write("")

        def rows(v):
            return [
                ("Hub cost, all six firms", f"{eur2(v.hub_cost)}/yr", False),
                (f"{member}: standalone cost", f"{eur2(float(v.standalone[i]))}/yr", False),
                (f"{member}: share of hub cost", f"{eur2(float(v.allocated[i]))}/yr", False),
                (f"{member}: paid above standalone", f"{eur2(float(v.excess[i]))}/yr", True),
                ("Materiality cutoff", f"{eur2(float(v.cutoff[i]))}/yr", False),
                ("Largest excess, any subgroup (ε*)", f"{eur2(v.eps)}/yr", False),
            ]

        left, right = st.columns(2)
        s = case["screening"]
        left.markdown(style.method_card(
            featured.SCREENING, featured.SCREENING_NOTE, s.passes,
            "Passes: no material objection" if s.passes else f"{len(s.material)} material objection(s)",
            rows(s)), unsafe_allow_html=True)
        if show_lp and "continuous" in case:
            c = case["continuous"]
            right.markdown(style.method_card(
                featured.CONTINUOUS, featured.CONTINUOUS_NOTE + (" Bundled with this release."
                                                                if case.get("bundled") else ""),
                c.passes, "Passes: no material objection" if c.passes
                else f"Material objection: {', '.join(g for g, _, _ in c.material)}", rows(c)),
                unsafe_allow_html=True)
        else:
            right.markdown(style.pending_card(
                featured.CONTINUOUS, "Turn on “Check continuous LP” to reprice every subgroup in this "
                "future and apply the same allocation rule."), unsafe_allow_html=True)

        methods = [s, case["continuous"]] if show_lp and "continuous" in case else [s]
        meter = [{"label": v.method, "excess": float(v.excess[i]), "cutoff": float(v.cutoff[i]),
                  "material": any(group == member for group, _, _ in v.material)} for v in methods]
        st.markdown(f"**{member}: amount paid above its standalone cost**")
        st.plotly_chart(objection_meter(meter), width="stretch",
                        config={"displayModeBar": False}, key="featured_meter")
        if show_lp and "continuous" in case:
            st.markdown("**Why the prices differ**")
            st.markdown("\n".join(f"- {line}" for line in why_lines(case)))
            with st.expander("Numbers for researchers · every member, both price sets"):
                c = case["continuous"]
                table = pd.DataFrame({
                    "Firm": names,
                    "Standalone, screening (€/yr)": s.standalone, "Share, screening (€/yr)": s.allocated,
                    "Excess, screening (€/yr)": s.excess,
                    "Standalone, LP (€/yr)": c.standalone, "Share, LP (€/yr)": c.allocated,
                    "Excess, LP (€/yr)": c.excess, "Cutoff, screening (€/yr)": s.cutoff,
                    "Cutoff, LP (€/yr)": c.cutoff,
                })
                st.dataframe(table.style.format({col: "{:,.2f}" for col in table.columns if "€" in col}),
                             hide_index=True, width="stretch")
                st.caption(
                    f"Excess = allocated share - standalone cost. Materiality: an objection must exceed "
                    f"max(€{case['abs']:,.0f}, {case['rel'] * 100:g}% of the group's cost). ε* is the signed "
                    f"least-core value over all 62 proper subgroups. Material objections under LP: "
                    + ("; ".join(f"{g} {eur2(e)} > {eur2(t)}" for g, e, t in c.material) or "none") + ".")
                inputs = case["inputs"]
                st.dataframe(pd.DataFrame({"Uncertainty": inputs.index, "Value in future 26": inputs.values}),
                             hide_index=True, width="stretch")
                st.download_button("Download future 26 comparison (CSV)", table.to_csv(index=False).encode(),
                                   "hubcore_future26_screening_vs_lp.csv", "text/csv")
