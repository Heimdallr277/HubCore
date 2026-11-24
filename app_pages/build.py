import numpy as np
import pandas as pd
import streamlit as st

from hubcore import profiles
from hubcore.costs import Assumptions
from hubcore.park import MAX_FIRMS, PRESETS, Park, validate
from hubcore.uncertainty import ASSUMPTION_UNCERTAINTIES
from ui import state, style
from ui.charts import dispatch_day, diversity_bars, eur, peak_day, waterfall

style.eyebrow("Step 1 · Build the hub")
st.title("What do these firms gain by sharing one contract?")
st.markdown(
    '<p class="hc-lead">Edit the park or pick a preset. For every possible group of firms, HubCore '
    "searches for a low-cost contract and battery over a full year of hourly loads. "
    "The capacity ceiling is a scenario assumption.</p>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------- inputs -------
with st.container(border=True):
    top = st.columns([2, 3])
    names = list(PRESETS)
    choice = top[0].selectbox("Start from a preset", names,
                              index=names.index(st.session_state.get("preset", names[0])))
    if choice != st.session_state.get("preset"):
        st.session_state["preset"] = choice
        st.session_state["park"] = PRESETS[choice]
        st.rerun()
    top[1].markdown(f'<p class="hc-small" style="margin-top:1.9rem">{PRESETS[choice].description}</p>',
                    unsafe_allow_html=True)

    with st.form("park_form", border=False):
        edited = st.data_editor(
            state.park_to_frame(state.park()), num_rows="dynamic", width="stretch",
            hide_index=True, key=f"editor_{choice}",
            column_config={
                "Firm": st.column_config.TextColumn(required=True, max_chars=28, width="medium"),
                "Type": st.column_config.SelectboxColumn(options=list(state.ARCH_LABEL.values()),
                                                         required=True, width="medium"),
                "Annual use (MWh)": st.column_config.NumberColumn(min_value=1, max_value=50000,
                                                                  step=10, format="%d", width="medium"),
                "Electrification plan": st.column_config.SelectboxColumn(
                    options=list(state.ADDON_LABEL.values()), required=True, width="medium"),
                "Planned extra use (MWh/yr)": st.column_config.NumberColumn(
                    min_value=0, max_value=50000, step=10, format="%d", width="medium",
                    help="Extra annual use if the plan is realized as intended"),
                "Existing contract (kW)": st.column_config.NumberColumn(
                    min_value=1, max_value=20000, step=10, format="%d", width="medium",
                    help="Current individual contracted capacity. The scenario keeps this right fixed. "
                         "Leave empty for 110% of today's peak."),
            },
        )
        with st.expander("Economic assumptions (base future)"):
            a0 = state.assumptions()
            vals = {}
            cols = st.columns(2)
            for k, u in enumerate(ASSUMPTION_UNCERTAINTIES):
                with cols[k % 2]:
                    vals[u.key] = st.slider(
                        f"{u.label} ({u.unit})", float(u.low), float(u.high),
                        key=f"base_{u.key}_{st.session_state.get('reset_id', 0)}",
                        value=float(getattr(a0, u.key)), help=f"{u.status}: {u.note}",
                        format="%.2f" if u.high < 10 else "%.0f")
            st.markdown("".join(style.chip(u.status, u.label) for u in ASSUMPTION_UNCERTAINTIES),
                        unsafe_allow_html=True)
        b1, b2 = st.columns([1, 5])
        submitted = b1.form_submit_button("Update park", type="primary")
        reset = b2.form_submit_button("Reset assumptions")
    if submitted:
        try:
            new_park = state.frame_to_park(edited, name=choice)
            if new_park.firms != PRESETS[choice].firms:
                new_park = Park(name=f"{choice} (edited)", firms=new_park.firms)
            else:
                new_park = PRESETS[choice]
            validate(new_park)
            st.session_state["park"] = new_park
            st.session_state["assumptions"] = a0.updated(**vals)
            st.rerun()
        except (ValueError, TypeError) as e:
            st.error(f"Could not update the park: {e}")
    if reset:
        st.session_state["reset_id"] = st.session_state.get("reset_id", 0) + 1
        st.session_state["assumptions"] = Assumptions()
        st.rerun()

p = state.park()
a = state.assumptions()
if p.n > MAX_FIRMS:
    st.error(f"At most {MAX_FIRMS} firms are supported.")
    st.stop()
with st.spinner("Optimizing contracts and batteries for every group of firms…"):
    cc = state.costs()
L = state.loads()

grand = cc.grand
sa = cc.standalone()
saving = cc.savings()
style.kpi_row([
    ("Firms on their own", eur(sa.sum()) + "/yr", "Capacity-related costs incl. shortfalls"),
    ("As one energy hub", eur(cc.cost[grand]) + "/yr", "Incl. hub overhead " + eur(cc.components["c_overhead"][grand])),
    ("Saving", f"{saving / sa.sum():.0%}" if sa.sum() else "-", eur(saving) + " per year"),
    ("Group contract", f"{cc.contract_kw[grand]:,.0f} kW", f"of {cc.capacity_right_kw[grand]:,.0f} kW capacity right"),
    ("Shared battery", f"{cc.battery_kwh[grand]:,.0f} kWh" if cc.battery_kwh[grand] > 0 else "None",
     f"{cc.battery_kwh[grand] / a.battery_duration_h:,.0f} kW power" if cc.battery_kwh[grand] > 0 else "Not worth it here"),
])
if saving <= 0:
    style.callout("⚠ In this configuration the hub costs more than the firms on their own. "
                  "Check the overhead assumptions or the firm mix.")
at_limit = state.groups_at_battery_limit(cc, p.names)
if at_limit:
    st.warning(
        f"⚠ For {len(at_limit)} group(s) the cheapest battery found is the largest size the fast search "
        f"tried ({', '.join(at_limit[:3])}{', …' if len(at_limit) > 3 else ''}). Their costs are upper "
        "bounds that may be loose. Check a design with the hourly reference LP below, or a future on the "
        "Stress-test page with continuous LP costs.")

# ------------------------------------------------------------- charts -------
total = L.sum(axis=0)
peak_hour = int(np.argmax(total))
day = peak_hour // 24
ts = profiles.timestamps()[day * 24]
loads_day = L[:, day * 24:(day + 1) * 24]
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(f"**Peak day of the combined load: {ts:%A %d %B}**")
    st.plotly_chart(peak_day(loads_day, p.names, cc.contract_kw[grand], cc.capacity_right_kw[grand],
                             None), width="stretch", config={"displayModeBar": False})
    st.caption("Bars show requested demand. Storage and curtailed demand can both account for the gap above the contract.")
with c2:
    st.markdown("**The diversity dividend**")
    sum_peaks = float(L.max(axis=1).sum())
    st.plotly_chart(diversity_bars(sum_peaks, float(total.max()), cc.capacity_right_kw[grand],
                                   cc.contract_kw[grand]), width="stretch", config={"displayModeBar": False})
    st.markdown(
        f'<p class="hc-small">Individually the firms peak at {sum_peaks:,.0f} kW in total, but their '
        f"combined load never exceeds {total.max():,.0f} kW: peaks happen at different hours "
        f"(diversity factor {sum_peaks / total.max():.2f}).</p>", unsafe_allow_html=True)

st.markdown("**Where the saving comes from**")
alone = {k: float(sum(v[1 << i] for i in range(p.n))) for k, v in cc.components.items()}
hub = {k: float(v[grand]) for k, v in cc.components.items()}
st.plotly_chart(waterfall(sa.sum(), hub, alone), width="stretch", config={"displayModeBar": False})

with st.expander("Inspect hourly dispatch with the reference LP"):
    st.caption("The reference LP takes the chosen battery and contract as fixed. Hourly imports, curtailment and storage flows are optimized again. This dispatch links midnight charge across days, so it is physically continuous. Coalition screening permits separate daily starting charge.")
    result = None
    if st.toggle("Solve and show the dispatch", key="show_dispatch"):
        try:
            result = state.dispatch(state.park_key(p), state.a_key(a), cc.capacity_right_kw[grand],
                                    cc.contract_kw[grand], cc.battery_kwh[grand])
        except state.SOLVER_ERRORS as exc:
            st.error(f"The reference LP could not be solved for this design ({exc}). "
                     "The screening results above are unaffected.")
    if result is not None:
        st.caption(f"Fixed-design chronological LP operating cost €{result.cost:,.0f}/yr, excluding hub overhead. "
                   f"Screening estimate at this design €{cc.cost[grand] - cc.components['c_overhead'][grand]:,.0f}/yr.")
        selected_day = st.number_input("Day of year", 1, 365, day + 1) - 1
        h = slice(selected_day * 24, (selected_day + 1) * 24)
        st.markdown(f"**{profiles.timestamps()[h][0]:%d %B %Y}** · residual shortfall {result.unserved[h].sum():,.1f} kWh")
        st.plotly_chart(dispatch_day(total[h], result, h), width="stretch")
        dispatch_data = pd.DataFrame({"Load (kW)": total[h], "Grid import (kW)": result.imports[h],
                                      "Charge (kW)": result.charge[h], "Discharge (kW)": result.discharge[h],
                                      "End-of-hour storage (kWh)": result.soc[h], "Shortfall (kW)": result.unserved[h]},
                                     index=profiles.timestamps()[h])
        st.dataframe(dispatch_data.round(2), width="stretch")

with st.expander("Each firm on its own"):
    rows = []
    for i, f in enumerate(p.firms):
        m = 1 << i
        rows.append({
            "Firm": f.name,
            "Contract right (kW)": cc.capacity_right_kw[m],
            "Peak incl. plan (kW)": L[i].max(),
            "Chosen contract (kW)": cc.contract_kw[m],
            "Own battery (kWh)": cc.battery_kwh[m],
            "Demand above contract (MWh/yr)": cc.unserved_kwh[m] / 1000,
            "Annual cost alone (€)": cc.cost[m],
        })
    st.dataframe(pd.DataFrame(rows).style.format(precision=0, thousands=","), hide_index=True,
                 width="stretch")
    st.caption("Costs cover capacity-related grid charges, demand that does not fit under the "
               "contract (valued at the unserved-demand cost), batteries and hub overhead. "
               "Commodity and volumetric network charges are excluded. Imports differ with storage losses and curtailment.")

with st.expander("Hourly loads as data"):
    df = pd.DataFrame(L.T, columns=p.names, index=profiles.timestamps())
    st.dataframe(df.iloc[day * 24:(day + 1) * 24].round(1), width="stretch")
    st.download_button("Download full-year hourly loads (CSV)", df.round(2).to_csv().encode(),
                       file_name="hubcore_loads.csv", mime="text/csv")
