import json
from pathlib import Path

import pandas as pd
import streamlit as st

from hubcore.uncertainty import default_space
from ui import state, style

ROOT = Path(__file__).resolve().parents[1]
style.eyebrow("Method & assumptions")
st.title("What the model can tell you")
st.markdown("HubCore prices hypothetical agreements among 2-8 firms. Monetary results cover capacity, batteries, shortfalls and coordination. All example firms and electrification plans are synthetic.")

st.subheader("Grid access and the cost boundary")
st.markdown("The Dutch grid operator assesses group eligibility, coincident use, historical or forecast profiles and substantiated plans. Members keep individual connections. A group can agree a transition of up to three years for recovery of prior rights. Later withdrawal with capacity requires a documented division and an operator check. During 2026 availability depends on the operator. The conditional entitlement starts on January 1, 2027.")
st.markdown("A hypothetical group with two or more firms has a ceiling κ times its members' summed rights. A singleton retains its own ceiling. Actual topology, connection-level limits, applications and exit terms are outside the model. The two power prices use Liander's 2026 MS reference row. One tariff category is assumed for all firms. Hourly averages can conceal peaks visible in quarter-hour meter readings. Liander also identifies individual kWh and connection charges. Purchased energy and volumetric network costs can change with curtailed load and storage losses.")

st.subheader("Hourly operating cost")
st.latex(r"c(S)=\min\;12\tau_c P+\tau_m\sum_m M_m+\mathrm{CRF}\,[k_E E+K_0\mathbf{1}_{E>0}]+c_u\sum_t u_t+F(S)")
st.latex(r"i_t-c_t+d_t+u_t=\ell_t,\quad s_t=s_{t-1}+\eta c_t-d_t,\quad i_t\leq\min(P,M_{m(t)}),\;P\leq\mathrm{Cap}(S)")
st.markdown("A time step is one hour. Imports *i*, charge *c*, discharge *d* and shortfall *u* are in kW, and end-of-hour charge *s* and installed size *E* are in kWh. Charge and discharge power are bounded by E/2 h. Efficiency is applied on charging. A separate daily start state closes each day's cycle. Consecutive days can start at different charge levels. The optional hourly dispatch view links the midnight states across the year and solves again at the displayed design.")
st.markdown("The HiGHS reference compares no battery with a continuously sized battery plus its fixed installation cost. The fast evaluator solves the storage-free case exactly and searches a grid of battery sizes and contracts with feasible daily dispatch. When the cheapest battery is the largest size tried, the search continues upward for that group. A group still at that limit is flagged on Build. The objective prices residual shortfall as a net penalty. Hub overhead applies once per group with two or more members. In every future the physical design can change as well as the operating policy.")

st.subheader("Cost sharing and objections")
st.latex(r"v(S)=\sum_{i\in S}c(\{i\})-c(S),\quad e(S,x)=\sum_{i\in S}x_i-c(S),\quad\sum_i x_i=c(N)")
st.latex(r"\varepsilon^*=\min_x\max_{\emptyset\ne S\subsetneq N}e(S,x)\quad\mathrm{subject\ to}\quad\sum_i x_i=c(N)")
st.markdown("A budget-balanced split belongs to the core if every proper subgroup has excess at most zero. The signed least-core minimizes the largest excess and then the L1 distance to the Shapley value. A negative ε* measures the attainable cushion. Repair minimizes L1 distance to the proposal inside the core if possible, or to the least-core if the core is empty. Transfers may produce a negative member cost. The cheapest partition compares all disjoint groupings.")
st.markdown("Materiality is a separate negotiation choice. By default an objection must exceed both €500/year and 1% of the subgroup's modeled cost. The threshold controls the badge and the robustness measure. The exact-core indicator has only a solver tolerance.")

st.subheader("Robustness and scenario discovery")
st.markdown("Uncertainties are stratified one dimension at a time by a seeded Latin hypercube. Fixed keys freeze the percentages signed in Build. Adaptive keys reapply the rule to realized costs. Each comparison is a one-year snapshot with hindsight. Regret in €/yr is the largest signed excess above that future's attainable minimum. A sampling frequency describes the chosen experimental design.")
st.markdown("The single-box PRIM variant peels 5% tails, retains at least 5% of futures, suggests the step maximizing coverage-density F1, and prunes weak dimensions. The figure traces raw peeling. The rule and its metrics use the pruned box. The case study scores the selected rules on 100 fresh futures with a different seed.")
a=state.assumptions()
rows=[{"Input":u.label, "Signing value":f"{u.base:g}", "Test range":f"{u.low:g}-{u.high:g}","Unit":u.unit,"Evidence":u.status} for u in default_space(state.park(),a)]
rows += [{"Input":k, "Signing value":v, "Test range":"Fixed", "Unit":unit, "Evidence":"Illustrative"} for k,v,unit in [
    ("Battery integration",f"{a.battery_fixed:,.0f}","€/installation"),("Battery life",str(a.battery_life_y),"years"),
    ("Round-trip efficiency",f"{a.battery_rte:.0%}","fraction"),("Duration",str(a.battery_duration_h),"hours"),
    ("Real discount rate",f"{a.discount_rate:.0%}","annual"),("Default existing right","110% of base peak, rounded up","10 kW increments")]]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.subheader("Two ways to price the alternatives")
st.markdown(
    "Every allocation test compares what a firm or subgroup pays inside the hub with what it would pay on "
    "its own. HubCore can price those alternatives in two ways. **Fast screening** evaluates all 2ⁿ-1 "
    "coalitions in every future with an exact storage-free solution and a finite battery and contract "
    "search; each cost is a feasible upper bound. **Continuous LP** repricing solves a HiGHS linear "
    "program with continuous contract and battery sizes for each coalition, using the same loads, daily "
    "storage closure and future. Small price differences can move a decision across a cutoff.")
st.markdown(
    "At the default €500 and 1% thresholds, the featured case on the Stress-test page shows one such decision. In future 26 of the six-firm, "
    "seed-42 design, the adaptive least-core split passes under screening prices: the cold store and the "
    "bakery each pay €444.95/year above their standalone cost, below the €500 cutoff. Continuous LP "
    "pricing finds a €201.67/year cheaper standalone design for the cold store, which alone lifts the "
    "largest excess to €512.17; with all 63 coalitions repriced it is €605.44 for both firms, a material "
    "objection. Across the 200 futures the two price sets disagree on materiality in futures 26, 28 and "
    "152 for six firms, and 25, 26, 129 and 146 for the five firms without the cold store.")

st.subheader("Reference-solver checks")
v=json.loads((ROOT / 'data/verification.json').read_text(encoding="utf-8"))
scope=v['all']
style.kpi_row([("Coalition comparisons",str(scope['count']),"70 nominal and 16 uncertain instances"),
               ("Largest fixed-size gap",f"{scope['gap_same_battery_pct']:.2f}%","LP given the evaluator's battery size"),
               ("Largest full-design gap",f"{scope['gap_optimal_pct']:.2f}%","LP choosing both contract and storage")])
st.caption("Observed gaps cover these comparisons. Other loads, costs and futures can give different errors. A threshold crossing near zero can change when coalition costs change.")
with st.expander("Results for each coalition and future"):
    st.caption("Future -1 is the signing case. The remaining IDs identify rows of the 200-future seed-42 ensemble. Costs exclude the common overhead in both solvers.")
    st.dataframe(pd.DataFrame(v['lp_comparison']).round(4),hide_index=True,width="stretch")
with st.expander("Full-ensemble LP comparison"):
    benchmark = state.preset_reference_summary()
    if benchmark:
        st.markdown("Every nonempty coalition of the six-firm example was repriced in 200 seed-42 futures and 100 independent seed-73 futures. The LP combines identical days only within the same month and weights shortfall by the number of occurrences. It uses the same daily storage closure as the screening model.")
        st.dataframe(pd.DataFrame([{"Design": title, "Coalition-futures": group["coalition_futures"],
                                   "Median gap (%)": group["median_pct"], "95th percentile (%)": group["p95_pct"],
                                   "Largest gap (%)": group["max_pct"], "Above 2%": group["above_2pct"]}
                                  for title, group in [("200 futures · seed 42", benchmark["training_gaps"]),
                                                       ("100 futures · seed 73", benchmark["holdout_gaps"])]]
                                 ).round(3), hide_index=True, width="stretch")
        st.caption("Each gap is (screening cost - LP optimum) / LP optimum, before coordination overhead. The separate 86 cases above also test the full hourly dispatch and the fixed-battery design.")
st.caption("Xinyi Mu · HubCore · 2026")
