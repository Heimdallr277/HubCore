import time
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from hubcore import allocation, games, prim
from hubcore import robustness as R
from hubcore import uncertainty as U
from ui import case_panel, featured, state, style
from ui.charts import exit_bars, futures_scatter, membership_bars, prim_trajectory, robustness_heat

style.eyebrow("Step 3 · Stress-test the deal")
st.title("Which deal still holds when the future turns out differently?")
st.markdown(
    '<p class="hc-lead">Contracts are signed once, but tariffs, battery prices, overheads and each '
    "firm's electrification plans are deeply uncertain. HubCore re-optimizes every possible group "
    "of firms in hundreds of sampled futures and checks each deal in each future. Two contract "
    "clauses are compared: a cost-sharing key <b>fixed at signing</b>, or the rule <b>recomputed "
    "each year</b> on realized data.</p>",
    unsafe_allow_html=True,
)

p = state.park()
space = U.default_space(p, state.assumptions())
case_requested = st.query_params.get("case") == str(featured.FEATURED["future"])
if case_requested and not st.session_state.get("_featured_link_applied"):
    st.session_state["_featured_link_applied"] = True
    st.session_state["_featured_inspect"] = True
    st.session_state["requested_futures"] = featured.FEATURED["n"]
    st.session_state["requested_seed"] = featured.FEATURED["seed"]
    st.session_state["featured_lp"] = True
labels = {u.key: u.label for u in space}

# ------------------------------------------------------------ ensemble -----
with st.container(border=True):
    c1, c2, c3 = st.columns([2, 1, 1])
    st.session_state.setdefault("requested_futures", state.N_DEFAULT if state.is_preset_unchanged() else 50)
    st.session_state.setdefault("requested_seed", state.SEED_DEFAULT)
    n_fut = c2.selectbox("Futures", [50, 100, 200], key="requested_futures")
    seed = int(c3.number_input("Seed", 0, 10_000, step=1, key="requested_seed"))
    st.session_state["home_design"] = (n_fut, seed)
    ens = state.get_ensemble(n_fut, seed)
    if ens is None and c1.button(f"Run stress test ({n_fut} futures)", type="primary"):
        bar = st.progress(0.0, text=f"Evaluating futures… each prices all {2 ** p.n - 1} groups of firms")
        started = time.perf_counter()

        def report(done: float) -> None:
            left = (time.perf_counter() - started) * (1 - done) / done if done > 0 else 0.0
            eta = "" if done >= 1 else (f" · about {left / 60:.0f} min left" if left >= 90
                                        else f" · about {left:.0f} s left")
            bar.progress(done, text=f"Evaluating futures… {done:.0%}{eta}")

        ens = state.run_ensemble(n_fut, seed, report)
        bar.empty()
    if ens is None:
        c1.caption(f"No results yet for {n_fut} futures, seed {seed}, with these inputs.")
    else:
        c1.success(f"{ens.n_futures} futures ready · seed {ens.seed}")
        hits = ens.battery_limit_groups
        if hits is not None and hits.any():
            st.warning(f"⚠ In {int((hits > 0).sum())} of {ens.n_futures} futures, at least one group's cheapest "
                       "battery is the largest size the fast search tried. Those coalition costs are loose upper "
                       "bounds; check such futures with continuous LP costs in the inspector below.")
    st.caption("Each future can choose a new contract and battery. Fixed percentages are signed under the current Build assumptions. These one-year comparisons assume perfect information about the realized future.")

with st.expander("What is uncertain, and over which ranges"):
    st.dataframe(pd.DataFrame([{
        "Uncertainty": u.label, "Low": u.low, "Base": u.base, "High": u.high, "Unit": u.unit,
        "Evidence": u.status, "Note": u.note} for u in space]), hide_index=True, width="stretch")
    st.caption("Ranges are explored, not predicted: every combination is treated as plausible, "
               "with no probabilities attached (deep uncertainty).")

if ens is None:
    st.info("Run the stress test to see robustness results for this park.")
    st.stop()

# ------------------------------------------------------- featured decision --
case = state.featured_case(with_lp=False) if (ens.n_futures, ens.seed) == (
    featured.FEATURED["n"], featured.FEATURED["seed"]) else None
if case is not None:
    case_panel.stress_panel(case, default_lp=case_requested)
elif case_requested:
    st.info("The featured decision (future 26) uses the unchanged six-firm preset with 200 futures. "
            "Restore the preset on Build and select that design to see it.")

# ---------------------------------------------------------- membership -----
members = st.multiselect("Hub members to test", options=list(range(p.n)), default=list(range(p.n)),
                         format_func=lambda i: p.names[i],
                         help="Remove a firm to test whether a smaller hub is more robust.")
if len(members) < 2:
    st.warning("Select at least two firms.")
    st.stop()
members = tuple(sorted(members))
with st.spinner("Checking every deal in every future…"):
    df = state.policy_table(ens, members)
summary = R.summarise(df).set_index(["rule", "clause"])
rule_order = list(allocation.RULES)
rule_labels = {k: v.label for k, v in allocation.RULES.items()}

viable = df.groupby("future").hub_viable.first().mean()
core = df.groupby("future").core_exists.first().mean()
best_key = summary["stable_share"].idxmax()
best = summary.loc[best_key]
worst_practical = summary.loc[[k for k in summary.index if allocation.RULES[k[0]].family == "practical key"]]
style.kpi_row([
    ("Hub cheaper than going alone", f"{viable:.1%}", "of sampled futures"),
    ("Core exists at screening prices", f"{core:.1%}", "of the sampled futures"),
    ("Best screened deal", f"{best.stable_share:.1%}", f"{rule_labels[best_key[0]]}, {R.CLAUSES[best_key[1]].lower()}"),
    ("Best practical key", f"{worst_practical.stable_share.max():.1%}",
     rule_labels[worst_practical.stable_share.idxmax()[0]] + ", " + R.CLAUSES[worst_practical.stable_share.idxmax()[1]].lower()),
])

st.caption("These results use the fast evaluator to price coalitions. An individual future can be checked with continuous LP costs below.")
reference_summary = state.preset_reference_summary()
reference_case = None
reference_design = reference_summary.get("training_design", {})
if (R.fingerprint(p, ens.base, ens.space, n_fut, seed) == reference_design.get("fingerprint")
        and st.session_state["abs_tol"] == 500.0 and st.session_state["rel_tol"] == 0.01):
    for entry in reference_summary["memberships"].values():
        reference = entry["reference"]
        if reference["members"] == [p.names[i] for i in members]:
            reference_case = entry
            lc_share = float(summary.loc[("leastcore", "adaptive"), "stable_share"])
            st.info(
                f"Continuous LP check across the same {n_fut} futures. The hub is cheaper in "
                f"{reference['viable']}/{n_fut}. A core exists in {reference['exact_core_exists']}/{n_fut} "
                f"(screening {round(core * n_fut)}/{n_fut}). The adaptive least-core passes materiality in "
                f"{reference['adaptive_material_pass']}/{n_fut} (screening {round(lc_share * n_fut)}/{n_fut})."
            )
            break

# ---------------------------------------------------------- heat table -----
st.subheader("Robustness of each deal")
metric = st.radio("Show", ["Passes materiality", "In exact core", "No firm materially worse off",
                           "Mean regret (€/yr)"], horizontal=True)
if metric == "Passes materiality":
    fig = robustness_heat(summary, "stable_share", rule_order, rule_labels, R.CLAUSES)
elif metric == "In exact core":
    fig = robustness_heat(summary, "exact_core_share", rule_order, rule_labels, R.CLAUSES)
elif metric.startswith("No firm"):
    fig = robustness_heat(summary, "all_better_off_share", rule_order, rule_labels, R.CLAUSES)
else:
    fig = robustness_heat(summary, "mean_regret", rule_order, rule_labels, R.CLAUSES, fmt="€{:,.0f}",
                          reverse=False, zmax=float(summary["mean_regret"].max()) or 1.0)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption("Regret is the largest signed excess minus the minimum attainable in that future. Darker cells indicate higher values; lower regret is preferable." if metric.startswith("Mean regret") else
           "Darker cells indicate a larger share of the selected futures. Shares describe this sampling design. "
           f"Materiality cutoff: max(€{st.session_state['abs_tol']:,.0f}, {st.session_state['rel_tol']:.2%} of coalition cost).")
fixed_best = summary.xs("fixed", level="clause")["stable_share"]
adapt_best = summary.xs("adaptive", level="clause")["stable_share"]
style.callout(
    f"<b>Reading:</b> the most robust deal is <b>{rule_labels[best_key[0]]}</b> with the rule "
    f"<b>{R.CLAUSES[best_key[1]].lower()}</b>, holding in {best.stable_share:.1%} of futures. "
    f"Across rules, recomputing each year holds in {adapt_best.mean():.1%} of futures on average "
    f"versus {fixed_best.mean():.1%} for keys fixed at signing. The hub costs less than separate operation in {viable:.1%} of futures."
)

# ------------------------------------------------------------ membership --
st.subheader("Would a smaller hub be more robust?")
key_cmp = ("membership_cmp", R.fingerprint(ens.park, ens.base, ens.space, ens.n_futures, ens.seed), members,
           st.session_state["rel_tol"], st.session_state["abs_tol"])
compare = len(members) > 2 and st.button("Compare hubs with one firm fewer")
if len(members) > 2 and (compare or st.session_state.get("membership_cmp") == key_cmp):
    st.session_state["membership_cmp"] = key_cmp
    labels_m, shares_m = [], []
    with st.spinner("Re-evaluating every deal for each smaller hub…"):
        for drop in members:
            subset = tuple(m for m in members if m != drop)
            s_sub = R.summarise(state.policy_table(ens, subset))
            labels_m.append(f"Without {p.names[drop]}")
            shares_m.append(float(s_sub["stable_share"].max()))
    st.plotly_chart(membership_bars(labels_m, shares_m, float(best.stable_share)), width="stretch", config={"displayModeBar": False})
    k = int(np.argmax(shares_m))
    if shares_m[k] > best.stable_share + 0.05:
        style.callout(f"Leaving out <b>{escape(p.names[members[k]])}</b> makes the best deal hold in "
                      f"{shares_m[k]:.1%} of futures instead of {best.stable_share:.1%}. Use the member "
                      "selector above to explore that hub.")
    else:
        st.caption("No single omission improves the pass rate by more than five percentage points.")
elif len(members) <= 2:
    st.caption("Select at least three firms to compare smaller hubs.")

# ------------------------------------------------------------- who leaves --
st.subheader("Which agreements face objections?")
pol_options = [(r, c) for r in rule_order for c in R.CLAUSES]
c1, c2 = st.columns([1, 1])
pol = c1.selectbox("Deal to inspect", pol_options, index=pol_options.index(best_key),
                   format_func=lambda k: f"{rule_labels[k[0]]} · {R.CLAUSES[k[1]].lower()}")
target = c2.radio("Failure means", ["Some group has a material objection", "Some firm is materially worse off"],
                  horizontal=False)
names_in = [p.names[i] for i in members]
shares = R.firm_exit_frequency(df, p.n, *pol)[list(members)]
st.plotly_chart(exit_bars(names_in, shares), width="stretch", config={"displayModeBar": False})

sub = df[(df.rule == pol[0]) & (df.clause == pol[1])].sort_values("future")
fail = ~(sub["stable"] if target.startswith("Some group") else sub["all_better_off"]).to_numpy()
keys = [u.key for u in space if not u.is_growth or u.firm_index in members]
X = ens.samples[keys].reset_index(drop=True)
with st.expander("Inspect an individual future", key="future_inspector", on_change="rerun"):
    out = X.rename(columns=labels).copy()
    out.insert(0, "Future ID", np.arange(len(X)))
    out["Selected failure"] = fail
    out["Hub saving (€/yr)"] = sub["hub_savings"].to_numpy()
    out["Largest excess (€/yr)"] = sub["max_excess"].to_numpy()
    st.caption("Select a row to see costs for the group with the strongest material objection. If none qualifies, the largest signed excess is shown.")
    table_key = f"future_table_{n_fut}_{seed}_{members}_{pol}"
    id_key = f"future_id_{n_fut}_{seed}_{members}_{pol}"

    def row_selected():
        selected = st.session_state[table_key]["selection"]["rows"]
        if selected:
            st.session_state[id_key] = int(selected[0])

    st.dataframe(out.round(3), width="stretch", hide_index=True, on_select=row_selected,
                 selection_mode="single-row", key=table_key)
    if case_requested and case is not None and (
            st.session_state.pop("_featured_inspect", False) or id_key not in st.session_state):
        st.session_state[id_key] = featured.FEATURED["future"]
    fid = int(st.number_input("Future ID to inspect", 0, n_fut - 1, step=1, key=id_key,
                              help="Enter the ID directly, or select a row above."))
    c = games.subgame(ens.costs[fid], p.n, list(members))
    nominal = games.subgame(ens.nominal_costs, p.n, list(members))
    def diagnose(cost, signing):
        allocation_eur = R.policy_allocation(pol[0], pol[1], cost, p.contracts()[list(members)],
                                             ens.kwh[fid, list(members)], signing,
                                             ens.nominal_kwh[list(members)])
        report = games.stability(allocation_eur, cost, len(members), st.session_state["rel_tol"],
                                 st.session_state["abs_tol"])
        if target.startswith("Some firm"):
            candidates = report.ir_violations or list(range(len(members)))
            j = max(candidates, key=lambda i: allocation_eur[i] - cost[1 << i])
            mask = 1 << j
            passes = not report.ir_violations
        else:
            mask = report.blocking[0][0] if report.blocking else report.worst_mask
            passes = report.stable
        local = games.members(mask, len(members))
        inside = float(allocation_eur[local].sum())
        name = " + ".join(names_in[i] for i in local)
        detail = (f"{name}: allocated €{inside:,.2f}/yr, modeled outside cost €{cost[mask]:,.2f}/yr, "
                  f"excess €{inside-cost[mask]:,.2f}/yr.")
        if not target.startswith("Some firm") and len(report.blocking) > 1:
            others = [(" + ".join(names_in[i] for i in games.members(m, len(members))), e)
                      for m, e in report.blocking[1:]]
            detail += " Other material objections: " + "; ".join(f"{g} €{e:,.2f}/yr" for g, e in others) + "."
        return passes, detail

    passed, detail = diagnose(c, nominal)
    st.markdown(f"**Future {fid} · {'passes' if passed else 'fails'} the selected test.** {detail}")
    if st.toggle("Check this future with continuous LP costs", key=f"lp_{id_key}",
                 help="Reprice every subgroup and reapply the selected allocation. Fixed keys are signed at LP costs."):
        try:
            with st.spinner("Pricing the selected subgroups…"):
                lp_costs, signed = state.reference_future(ens, members, fid, fixed=pol[1] == "fixed")
            verified_pass, lp_detail = diagnose(lp_costs, signed)
            st.markdown(f"**Continuous LP · future {fid} {'passes' if verified_pass else 'fails'} the selected test.** "
                        f"{lp_detail}")
        except state.SOLVER_ERRORS as exc:
            st.error(f"The continuous LP check could not be completed for future {fid} ({exc}). "
                     "The screening verdict above is unaffected.")
    export = out.assign(seed=seed, members=" | ".join(names_in), rule=pol[0], clause=pol[1],
                        failure_test=target, coalition_costs="fast screening", abs_eur=st.session_state['abs_tol'], relative=st.session_state['rel_tol'],
                        design=R.fingerprint(p, ens.base, ens.space, n_fut, seed))
    st.download_button("Download these futures (CSV)", export.to_csv(index=False).encode(),
                       f"hubcore_{pol[0]}_{pol[1]}_N{n_fut}_seed{seed}.csv", "text/csv")

st.markdown(f"**Scenario discovery (PRIM).** The deal fails in {fail.mean():.1%} of futures. "
            "Which conditions explain most failures?")
if fail.sum() == 0:
    st.success("This deal never fails in the sampled futures.")
    st.stop()
if fail.all():
    st.warning("This deal fails in every sampled future, so there is nothing to discover. "
               "Pick a more robust deal or remove a firm.")
    st.stop()

traj = prim.prim(X, fail, alpha=0.05, min_mass=0.05)
default_step = prim.suggest(traj)
step = (st.slider("Peeling step (move right for a smaller raw box)", 0, len(traj) - 1, default_step)
        if len(traj) > 1 else 0)
box, pvals = prim.explain(traj[step], X, fail)

c1, c2 = st.columns([2, 3])
with c1:
    st.plotly_chart(prim_trajectory([b.coverage for b in traj], [b.density for b in traj], step),
                    width="stretch", config={"displayModeBar": False})
with c2:
    if box.restricted:
        conds = []
        for k in box.restricted:
            lo, hi = prim.open_limits(box, X)[k]
            label = escape(labels[k])
            conds.append(f"{label} ≤ {hi:.3g}" if not np.isfinite(lo) else
                         f"{label} ≥ {lo:.3g}" if not np.isfinite(hi) else f"{lo:.3g} ≤ {label} ≤ {hi:.3g}")
        st.markdown("**The deal tends to fail when**  \n" + "  \n".join(
            f"• {c}  <span class='hc-small'>(quasi-p = {pvals.get(k, float('nan')):.3f})</span>"
            for c, k in zip(conds, box.restricted)), unsafe_allow_html=True)
    else:
        st.markdown("**No compact set of conditions found** at this step.")
    st.markdown(f"After pruning: coverage **{box.coverage:.1%}** of all failures · density **{box.density:.1%}** "
                f"of futures inside the box fail · box holds **{box.mass:.1%}** of futures.")
    st.caption("The plot shows raw peeling steps. The reported rule is pruned; quasi-p values rank restrictions within the selected sample.")
    if (reference_case and pol == ("leastcore", "adaptive") and target.startswith("Some group")
            and step == default_step):
        fit = reference_case["reference"]
        if not fit["box"]:
            st.caption("The suggested box also retains no condition when all coalition prices come from continuous LPs.")
        elif len(fit["box"]) == 1:
            factor, bounds = next(iter(fit["box"].items()))
            if bounds[0] is None and bounds[1] is not None:
                scored = reference_case["screening_rule_on_lp_holdout"]
                st.caption(
                    f"Refitting PRIM with continuous LP prices selects {labels[factor]} ≤ {bounds[1]:.2f} times plan. "
                    f"That refitted rule has {fit['holdout']['density']:.1%} failure density on the independent LP holdout. "
                    f"Applying the screening rule to the same holdout gives {scored['failures_inside']}/{scored['inside']} failures inside. "
                    "The cutoff depends on how coalition costs are estimated."
                )
    keys = box.restricted[:2] if len(box.restricted) >= 2 else (box.restricted + [k for k in X.columns if k not in box.restricted])[:2]
    xk = st.selectbox("x-axis", list(X.columns), index=list(X.columns).index(keys[0]), format_func=labels.get)
    remaining = [k for k in X if k != xk]
    yk = st.selectbox("y-axis", remaining, index=remaining.index(keys[1]) if keys[1] in remaining else 0, format_func=labels.get)
st.plotly_chart(futures_scatter(X[xk].to_numpy(), X[yk].to_numpy(), fail, labels[xk], labels[yk],
                                (box.limits[xk], box.limits[yk]) if box.restricted else None, "deal fails", "deal holds"),
                width="stretch", config={"displayModeBar": False})
st.caption("Each point is one sampled future. The rectangle projects the rule onto these axes. Other restrictions still apply."
           if box.restricted else "Each point is one sampled future. The selected step yields no restricted condition after pruning.")

