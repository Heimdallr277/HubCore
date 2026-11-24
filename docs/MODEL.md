# Model, data and sources

## Inputs and scope

HubCore evaluates every nonempty subset of 2-8 firms as a possible capacity-sharing group. The six-firm Voorbeeldpark and the three-firm teaching park are fictional. Their annual demands and electrification plans are defined in [park.py](../src/hubcore/park.py). A cheaper subgroup indicates a reason to renegotiate under the assumed rights.

Each load shape has 8,760 hourly values for 2025. A shape integrates to 1,000 kWh, so multiplying it by a firm's annual MWh gives hourly demand in kW. The G0-G6 commercial shapes come from German BDEW profiles in demandlib 0.2.2. The two-shift industrial profile uses assumed working hours, and the two EV profiles use stylized charging schedules. Process electrification scales the firm's base shape. The generator and metadata record the seven selected non-working dates. [Profile generator](../scripts/make_profiles.py) · [Calendar and units](../src/hubcore/data/profiles_2025.meta.json) · [demandlib license](../src/hubcore/data/DEMANDLIB_LICENSE.txt)

By default, each firm's capacity right is 110% of its base peak rounded up to 10 kW. A singleton retains that right. A larger group has an assumed ceiling of $\kappa$ times its members' combined rights. Liander's ordinary 2026 medium-voltage rate for over 136 through 2,000 kW supplies two reference prices, €2.29/kW/month for contracted capacity and €3.57/kW/month for the monthly maximum. The model applies these to all firms and groups. Hourly averages can understate quarter-hour peaks. Commodity purchases, volumetric charges, individual fixed fees, taxes, export and trading are outside the cost calculation. [Dutch tariff and grid sources](#sources)

## Coalition costs

For a group $S$, the model chooses annual contract $P$, monthly import peaks $M_m$, battery capacity $E$ and hourly dispatch to minimize the following annual cost.

$$
c(S)=\min\left[12\tau_c P+\tau_m\sum_{m=1}^{12}M_m+\mathrm{CRF}\big(k_E E+K_0\mathbf{1}_{E>0}\big)+c_u\sum_t u_t+F(S)\right].
$$

The first two terms price capacity. The next annualizes the battery over 15 years at a 6% real discount rate. The default is €250/kWh plus €20,000 per installation. The €150-400/kWh scenario range varies the size-dependent part. The last terms value unmet demand $u_t$ at an assumed net penalty and add coordination overhead $F(S)$ once for a group of two or more. These are scenario assumptions. [Default parameters](../src/hubcore/costs.py)

In each one-hour interval, imports plus discharge and shortfall cover demand plus charging. Imports respect both the purchased contract and the monthly peak. Storage has a two-hour duration and 90% round-trip efficiency by default. Each day ends at its own starting charge, so coalition pricing omits interday storage and allows different opening levels on consecutive days. The optional dispatch view links the midnight states at a chosen design.

**Fast screening** solves the storage-free case exactly, then searches candidate battery sizes and contracts. A candidate gives a feasible upper bound on its group's cost within numerical tolerance. Differences between two such bounds can move a savings estimate in either direction. **Continuous LP** uses HiGHS to compare no storage with continuously sized storage including its fixed installation cost. Under independent daily closure, identical 24-hour load profiles within one month can share one dispatch schedule. Weighting their shortfall by the number of occurrences preserves the annual optimum. [Screening](../src/hubcore/costs.py) · [Full hourly LP](../src/hubcore/lp_reference.py) · [Repeat-day LP](../src/hubcore/annual_reference.py)

## Cost sharing and uncertainty

A split assigns each firm an annual cost $x_i$ and must sum to the full hub cost $c(N)$. A subgroup's excess is $e(S,x)=\sum_{i\in S}x_i-c(S)$. Positive excess means that subgroup could pay less separately. The **core** contains all splits with no positive excess for any nonempty proper subgroup. The **signed least-core** minimizes the largest excess, breaking ties by minimum L1 distance to the Shapley allocation. The other five rules use capacity, requested energy, equal percentage savings, equal euro savings or Shapley marginal costs. Negative shares represent transfers. The repair option finds the closest split to a proposal inside the core, or inside the least-core when the core is empty.

The app separately flags a material objection when excess exceeds both €500/year and 1% of the subgroup's alternative cost by default. A deal can pass this test even with an empty core.

A seeded Latin hypercube varies the two tariffs, battery cost, shortfall penalty, group-capacity factor, coordination costs and electrification plans. Fixed rules keep the percentages agreed at signing. Adaptive rules recompute the split in each future. Both allow the physical design to change. The design samples assumed ranges and gives no estimate of real-world frequencies.

## Benchmark results

Counts below use the same 200 futures (seed 42). The five-firm comparison removes the cold store and evaluates the remaining members' deal.

| Outcome out of 200 | Six firms, screening | Six firms, LP | Five firms, screening | Five firms, LP |
| :--- | ---: | ---: | ---: | ---: |
| Hub cheaper than separate operation | 160 | 160 | 163 | 163 |
| Core exists | 33 | 32 | 99 | 96 |
| Best fixed rule passes | 1 | 1 | 6 | 5 |
| Adaptive least-core passes | 52 | 51 | 126 | 124 |

For six firms, LP repricing changes three materiality verdicts. Futures 26 and 152 pass only under screening, and future 28 passes only under LP. Across all 63 coalitions in the 200 design and 100 holdout futures, 66 of 18,900 screening costs were more than 2% above the LP optimum, measured before coordination overhead. The repeat-day LP was also checked against an unaggregated hourly LP on 86 cases. [Screening results](../data/case_results.json) · [LP results](../data/preset_lp_checks.json) · [Hourly checks](../data/verification.json)

For scenario discovery, a single-box PRIM variant peels 5% tails, selects by coverage-density F1 and prunes weak dimensions. For six firms, neither price method identified a narrower region of failure. For five firms, the screening rule selected metalworks electrification up to 0.90 times plan. On 100 independent futures (seed 73) it captured 29 of 38 LP failures among 45 selected futures. Refitting on LP training labels moved the cutoff to 1.53 times plan and selected 77 holdout futures, 33 of which failed. These cutoffs are sensitive to the pricing method. [PRIM implementation](../src/hubcore/prim.py) · [Holdout scores](../data/preset_lp_checks.json)

## Reproduce the results

The app reads the bundled results on startup. After installing `requirements-dev.txt` in the activated environment, this test recalculates all 63 coalition prices for future 26 under both methods. [Example in the README](../README.md#what-changes-a-decision)

```bash
python -m pytest -q tests/test_interface.py -k future_26_screening_passes
```

To rebuild all bundled results, run these commands from the repository root in order. They overwrite their listed outputs.

| Command | Output |
| --- | --- |
| `python scripts/make_profiles.py` | `src/hubcore/data/profiles_2025.csv.gz` and `profiles_2025.meta.json` in the same directory |
| `python scripts/precompute.py` | The two seed-42 screening ensembles in `data/precomputed/` |
| `python scripts/verify.py` | [Hourly LP checks](../data/verification.json) |
| `python scripts/case_study.py` | [Signing case, membership comparisons and screening holdout](../data/case_results.json) |
| `python scripts/verify_preset.py --workers 4` | [LP cost arrays](../data/preset_lp_costs.npz) and [comparison results](../data/preset_lp_checks.json) |

Future IDs are zero-based row numbers. In the coalition arrays, column zero is the empty group and bit $i$ of the column number denotes firm $i$ in the park definition. Costs are euros per year. Stored fingerprints identify the inputs and scenario design. Machine timings in the hourly verification file can vary between runs.

## Sources

**Dutch grid access and tariffs.** Group eligibility, allocated capacity and withdrawal depend on operator assessment. The group-capacity factor in HubCore is an assumption. The price inputs use the ordinary MS row of table 2.1, version 2026.1.1. [ACM group transport decision](https://zoek.officielebekendmakingen.nl/stcrt-2025-43262.html) · [ACM 2026 amendment](https://zoek.officielebekendmakingen.nl/stcrt-2026-4110.html) · [Liander group agreements](https://www.liander.nl/grootzakelijk/energietransitie/samenwerken/transportovereenkomst-voor-groepen) · [Liander 2026 tariff sheet](https://www.liander.nl/-/media/files/tarieven/grootzakelijk/tarieven-2026/tarieven-voor-aansluiting-en-transport-elektriciteit-2026_2.pdf) · [Stedin metering definitions](https://www.stedin.net/-/media/project/online/files/tarieven/tarieven-2026/tarieven-en-vergoedingsregeling-2026_-elektriciteit_grootverbruik.pdf?rev=3aa494378c71488ca57ad6d0f6ec961f)

**Load profiles.** The commercial shapes serve as proxies for the synthetic park. [oemof demandlib, BDEW profiles](https://demandlib.readthedocs.io/en/stable/bdew.html)

**Cooperative games.** Marginal cost allocation, the airport-cost benchmark and the geometry of the core and related solutions. [Shapley (1953)](https://doi.org/10.1515/9781400881970-018) · [Littlechild & Owen (1973)](https://doi.org/10.1287/mnsc.20.3.370) · [Maschler, Peleg & Shapley (1979)](https://doi.org/10.1287/moor.4.4.303)

**Exploratory scenarios.** Decisions under deep uncertainty, patient rule induction and scenario discovery. [Lempert, Popper & Bankes (2003)](https://www.rand.org/content/dam/rand/pubs/monograph_reports/2007/MR1626.pdf) · [Friedman & Fisher (1999)](https://doi.org/10.1023/A:1008894516817) · [Bryant & Lempert (2010)](https://doi.org/10.1016/j.techfore.2009.08.002)
