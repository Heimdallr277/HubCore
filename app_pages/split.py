import numpy as np
import pandas as pd
import streamlit as st

from hubcore import allocation, games
from ui import state, style
from ui.charts import allocation_dots, eur, excess_bars

style.eyebrow("Step 2 · Split the bill")
st.title("How should the firms share the hub's cost?")
st.markdown("A split passes the materiality test when no subgroup could save more than the selected threshold on a hypothetical alternative. The mathematical core is stricter: every excess must be at most zero.")

with st.expander("Settings · material objections"):
    rel = st.slider("Relative threshold (% of the group's cost)", 0.0, 5.0,
                    st.session_state["rel_tol"] * 100, 0.25, key="input_rel") / 100
    abs_ = float(st.slider("Absolute threshold (€/yr)", 0, 5000, int(st.session_state["abs_tol"]), 100,
                           key="input_abs"))
    st.session_state["rel_tol"], st.session_state["abs_tol"] = rel, abs_
    st.caption("A material objection exceeds both cutoffs. They express a negotiation preference. Numerical error and approximation error are separate.")

p = state.park()
cc = state.costs()
alloc = state.allocations()
facts = state.game_facts()
n = p.n
names = p.names
sa = cc.standalone()
grand = cc.grand
rel, abs_ = st.session_state["rel_tol"], st.session_state["abs_tol"]


def group_label(mask: int) -> str:
    mem = games.members(mask, n)
    return " + ".join(names[i] for i in mem)


def short_label(mask: int) -> str:
    mem = games.members(mask, n)
    if len(mem) == n - 1 and n > 2:
        missing = [i for i in range(n) if i not in mem][0]
        return f"All except {names[missing]}"
    if len(mem) <= 2:
        return group_label(mask)
    return " + ".join(names[i].split()[0] for i in mem)


stab = {r: games.stability(x, cc.cost, n, rel, abs_) for r, x in alloc.items()}
eps = facts["eps"]
parts = facts["partition"]
one_hub = len(parts) == 1

style.kpi_row([
    ("Is the exact model core non-empty?", "Yes" if eps <= games.numerical_tol(cc.cost) else "No",
     "The core is non-empty" if eps <= games.numerical_tol(cc.cost) else f"Best possible: some group still gains {eur(eps)}/yr"),
    ("Rules passing materiality", f"{sum(s.stable for s in stab.values())} of {len(stab)}", f"Threshold: max({eur(abs_)}, {rel:.2%} of group cost)"),
    ("Cheapest modeled partition", "One hub, all firms" if one_hub else f"{len(parts)} separate groups",
     "" if one_hub else " | ".join(group_label(s) for s in parts)),
])

st.caption("Partitions compare disjoint groups under hypothetical rights. Grid eligibility and capacity after exit require a separate operator assessment.")

# ------------------------------------------------------- rule comparison ----
st.subheader("Six ways to split the bill")
savings_pct = {r: 100 * (sa - x) / sa for r, x in alloc.items()}
st.plotly_chart(allocation_dots(savings_pct, {k: v.label for k, v in allocation.RULES.items()}, names,
                                {r: s.stable for r, s in stab.items()}), width="stretch", config={"displayModeBar": False})
st.caption("Each dot is one firm. Left of the zero line, the firm pays more than on its own. "
           "✓ passes the materiality test. ✗ marks a material objection. Above 100% savings the firm receives a net transfer.")

rows = []
for r, x in alloc.items():
    s = stab[r]
    worst = s.blocking[0] if s.blocking else None
    rows.append({
        "Rule": allocation.RULES[r].label,
        "Passes materiality?": "✓ yes" if s.stable else "✗ no",
        "In exact core?": "Yes" if s.max_excess <= games.numerical_tol(cc.cost) else "No",
        "Material objections": s.n_blocking,
        "Strongest objection": f"{group_label(worst[0])} gains {eur(worst[1])}/yr by leaving" if worst else "-",
        **{f"{nm} (€/yr)": x[i] for i, nm in enumerate(names)},
    })
table = pd.DataFrame(rows)
st.dataframe(table.style.format({c: "{:,.0f}" for c in table.columns if c.endswith("(€/yr)")}),
             hide_index=True, width="stretch")

with st.expander("What each rule does"):
    for r in allocation.RULES.values():
        st.markdown(f"**{r.label}** ({r.family}). {r.description}")
    st.markdown(
        "The practical keys use a single quantity and can miss what each firm "
        "*brings* to the hub (spare capacity at other firms' peak hours). The Shapley value rewards "
        "contributions on average but can still leave a group better off on its own. The "
        "least-core split minimizes the largest signed excess, then minimizes distance to Shapley. Negative cost shares represent transfers."
    )

st.divider()

# ------------------------------------------------------ negotiation sandbox --
st.subheader("Negotiate: propose your own split")
st.markdown('<p class="hc-small">Enter each firm\'s share of the hub\'s annual cost. Shares are '
            "rescaled to 100%. HubCore checks every one of the "
            f"{2 ** n - 2} possible breakaway groups.</p>", unsafe_allow_html=True)
start = alloc["capacity"] / cc.cost[grand] * 100
prop_df = pd.DataFrame({"Firm": names, "Share of hub cost (%)": np.round(start, 1),
                        "Cost alone (€/yr)": [f"{v:,.0f}" for v in sa]})
c1, c2 = st.columns([2, 3])
with c1:
    edited = st.data_editor(prop_df, hide_index=True, width="stretch",
                            disabled=["Firm", "Cost alone (€/yr)"], key=f"prop_{state.park_key(p)}",
                            column_config={"Share of hub cost (%)": st.column_config.NumberColumn(
                                min_value=0.0, max_value=100.0, step=0.5, format="%.1f")})
    shares = edited["Share of hub cost (%)"].fillna(0).to_numpy(dtype=float)
    if shares.sum() <= 0:
        st.error("Enter at least one positive share.")
        st.stop()
    x_prop = shares / shares.sum() * cc.cost[grand]
    s_prop = games.stability(x_prop, cc.cost, n, rel, abs_)
    st.markdown(style.status(s_prop.stable, "This split holds",
                             f"{s_prop.n_blocking} material objection(s)"),
                unsafe_allow_html=True)
with c2:
    exc = games.excess(x_prop, cc.cost, n)
    order = np.argsort(-exc[1:grand])[:8] + 1
    st.markdown("**Subgroups with the largest excess**")
    items = [(short_label(int(m)), float(exc[m]), max(abs_, rel * cc.cost[m])) for m in order]
    st.plotly_chart(excess_bars(items), width="stretch", config={"displayModeBar": False})
    st.caption("Clay bars marked ⚠ exceed that subgroup's materiality cutoff. Gray bars are below it; "
               "negative values mean the subgroup is better off inside the hub.")

if s_prop.blocking:
    m, e = s_prop.blocking[0]
    mem = games.members(m, n)
    pays = x_prop[mem].sum()
    st.markdown(
        f"**Strongest objection.** {group_label(m)} would pay {eur(pays)}/yr under this split. "
        f"The subgroup's modeled alternative costs {eur(cc.cost[m])}/yr, a difference of {eur(e)}/yr."
    )

if st.button("Repair the proposal", type="primary"):
    x_fix, eps_used = games.closest_in_least_core(cc.cost, n, x_prop)
    s_fix = games.stability(x_fix, cc.cost, n, rel, abs_)
    diff = x_fix - x_prop
    rep = pd.DataFrame({
        "Firm": names,
        "Your proposal (€/yr)": x_prop, "Repaired (€/yr)": x_fix, "Change (€/yr)": diff,
        "Saving vs alone": [f"{max((sa[i] - x_fix[i]) / sa[i], 0.0) if abs(sa[i] - x_fix[i]) < 1 else (sa[i] - x_fix[i]) / sa[i]:.0%}"
                            for i in range(n)],
    })
    st.dataframe(rep.style.format({c: "{:,.0f}" for c in rep.columns if "€" in c}),
                 hide_index=True, width="stretch")
    moved = np.abs(diff).sum() / 2
    indifferent = [names[i] for i in range(n) if abs(sa[i] - x_fix[i]) < 1]
    domain = "core" if eps_used <= games.numerical_tol(cc.cost) else "least-core"
    msg = (f"The nearest {domain} allocation moves {eur(moved)}/yr between firms. "
           "The L1 distance counts each transfer on both sides. ")
    if indifferent:
        msg += f"Within €1 of their standalone cost: {', '.join(indifferent)}. "
    if domain == "least-core":
        msg += f"The exact core is empty. Even the best split leaves a largest excess of {eur(eps_used)}/yr. "
    if s_fix.stable:
        st.success(msg + "The repaired allocation passes the selected materiality test.")
    else:
        st.warning(msg + f"{s_fix.n_blocking} material objection(s) remain.")
