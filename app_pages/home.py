import json

import numpy as np
import streamlit as st

from hubcore import allocation, games
from hubcore import robustness as R
from ui import case_panel, state, style
from ui.charts import eur

p = state.park()

style.eyebrow("HubCore · Business-park energy hubs")
st.title("Will this energy hub hold together?")
st.markdown('<p class="hc-lead">Six fictional firms consider pooling their transport capacity. '
            'A shared battery can cut shortfalls. Who pays, and which groups could obtain a cheaper '
            'deal of their own? Edit the park, negotiate a split and test it across uncertain futures.</p>'
            if state.is_preset_unchanged() and p.n == 6 else
            '<p class="hc-lead">Explore shared capacity, negotiate a cost split and see which '
            'groups could obtain a cheaper modeled alternative across uncertain futures.</p>', unsafe_allow_html=True)
n_fut, seed = st.session_state.get("home_design", (state.N_DEFAULT, state.SEED_DEFAULT))
ens = state.get_ensemble(n_fut, seed)
if ens is None:
    cc = state.costs()
    cost = cc.cost
    alloc = state.allocations()
else:
    cost = ens.nominal_costs
    alloc = allocation.all_rules(cost, p.n, p.contracts(), ens.nominal_kwh)
sa = np.array([cost[1 << i] for i in range(p.n)])
saving = sa.sum() - cost[-1]
stable_rules = [r for r, x in alloc.items()
                if games.stability(x, cost, p.n, st.session_state["rel_tol"],
                                   st.session_state["abs_tol"]).stable]
robust_value, robust_note = "Run step 3", f"No results for {n_fut} futures, seed {seed}"
if ens is not None:
    evidence = json.loads((state.ROOT / "data/case_results.json").read_text(encoding="utf-8"))
    fid = R.fingerprint(p, ens.base, ens.space, n_fut, seed)
    if (fid == evidence["ensemble"].get("fingerprint") and st.session_state["rel_tol"] == 0.01
            and st.session_state["abs_tol"] == 500):
        best = max(evidence["six_firms"]["policies"], key=lambda rec: rec["stable_share"])
    else:
        summary = R.summarise(state.policy_table(ens, tuple(range(p.n))))
        best = summary.loc[summary.stable_share.idxmax()]
    robust_value = f"{best['stable_share']:.1%} of futures"
    robust_note = f"{n_fut} futures · seed {seed} · {best['rule_label']}, {best['clause_label'].lower()}"
    if (fid == evidence["ensemble"].get("fingerprint") and st.session_state["rel_tol"] == 0.01
            and st.session_state["abs_tol"] == 500 and best["rule"] == "leastcore" and best["clause"] == "adaptive"):
        reference = state.preset_reference_summary()
        if reference.get("training_design", {}).get("fingerprint") == fid:
            passes = reference["memberships"]["six_firms"]["reference"]["adaptive_material_pass"]
            robust_note = (f"Screening {round(best['stable_share'] * n_fut)}/{n_fut} · continuous LP {passes}/{n_fut}"
                           " · adaptive least-core")


style.kpi_row([
    ("Hub saving vs. going alone", eur(saving) + "/yr", f"{saving / sa.sum():.0%} of the firms' combined modeled costs"),
    ("Cost splits that hold today", f"{len(stable_rules)} of {len(alloc)}", "No subgroup has a material cost objection"),
    ("Most robust deal holds in", robust_value, robust_note),
    ("Firms in the park", str(p.n), p.name),
])

st.write("")
case = state.featured_case(with_lp=False)
if case is not None:
    case_panel.overview_card(case)
    st.write("")

st.subheader("A one-minute path through the model")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="hc-step">STEP 1</div>', unsafe_allow_html=True)
    st.markdown("**Build the hub**  \nCompose the park and see the *diversity dividend*: how much "
                "less capacity firms need together, and a candidate contract and battery "
                "(found by optimization).")
    st.page_link("app_pages/build.py", label="Build the hub", icon=":material/arrow_forward:")
with c2:
    st.markdown('<div class="hc-step">STEP 2</div>', unsafe_allow_html=True)
    st.markdown("**Split the bill**  \nCompare six cost-sharing rules, propose your own split, "
                "identify a group with a cheaper modeled alternative and adjust the proposal to the "
                "nearest core split or minimum-objection compromise.")
    st.page_link("app_pages/split.py", label="Split the bill", icon=":material/arrow_forward:")
with c3:
    st.markdown('<div class="hc-step">STEP 3</div>', unsafe_allow_html=True)
    st.markdown("**Stress-test the deal**  \nRe-run the hub in 200 futures (tariffs, "
                "battery prices, growth), inspect future 26 under two pricing methods and get "
                "readable rules for *when* a deal breaks (scenario discovery).")
    st.page_link("app_pages/stress.py", label="Stress-test the deal", icon=":material/arrow_forward:")

st.divider()
left, right = st.columns([3, 2])
with left:
    st.subheader("Inside the model")
    st.markdown(
        "- **Optimization:** for every possible group of firms (2ⁿ-1 groups), a candidate group "
        "contract and shared battery, from hourly loads over a full year. A fast solver is checked "
        "against an exact linear program.\n"
        "- **Cooperative game theory:** Shapley value, least-core, and stability checks that ask "
        "whether any group of firms would save money by forming its own hub.\n"
        "- **Robust decision-making:** a Latin hypercube ensemble over tariffs, battery cost, "
        "overheads and each firm's electrification plans, with regret and stability metrics.\n"
        "- **Scenario discovery (PRIM):** readable conditions under which a deal fails."
    )
with right:
    st.subheader("How much to trust it")
    st.markdown(
        style.chip("Sourced", "2026 grid tariffs") + style.chip("Illustrative", "battery cost range")
        + style.chip("Proxy", "load shapes") + style.chip("Illustrative", "overheads, unserved cost")
        + style.chip("Synthetic", "example firms"),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hc-small">HubCore is an <b>exploratory</b> model. It shows how the design of a '
        "cost-sharing deal interacts with uncertainty. Modeled exits use hypothetical capacity rights. See <i>Method &amp; assumptions</i>.</p>",
        unsafe_allow_html=True,
    )

with st.expander("Why this matters in the Netherlands"):
    st.markdown(
        "Grid congestion stops many Dutch firms from expanding or electrifying. In December 2025 "
        "the regulator ACM published the decision that makes the **group transport agreement "
        "(groepstransportovereenkomst, GTO)** possible: several large consumers sign one transport "
        "agreement with the grid operator and share the group's contracted capacity. During 2026 availability depends on the operator. Consultation responses stressed that firms "
        "become dependent on each other, which makes fair and stable cost-sharing a practical "
        "question rather than a theoretical one.\n\n"
        "Sources: ACM, *Codebesluit groepstransportovereenkomst* (Stcrt. 2025, 43262, published "
        "Dec 19, 2025); Liander, *Tarieven voor aansluiting en transport elektriciteit 2026*."
    )

with st.expander("From PhD research to this tool"):
    st.markdown(
        "My PhD modeled how households reach collective energy decisions. HubCore asks "
        "a related design question for firms sharing capacity. Given the cost of each alternative, "
        "can the members agree on a split for which every subgroup pays at most its outside cost? "
        "The mathematical core identifies those splits."
    )
