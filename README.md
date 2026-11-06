# AI Group Assignment 

Probabilistic reasoning, automated planning, and logical inference, solved in three self-contained problems.

| # | Problem | Technique | Tools |
|---|---------|-----------|-------|
| 1 | Smart Home Energy Consumption | Bayesian Networks: inference, d-separation, sampling, parameter learning | Python, pgmpy, NumPy, pandas, matplotlib |
| 2 | Mars Rover Sample Collection | Classical planning with PDDL (`:strips`, `:typing`) | PDDL, GBFS, A* |
| 3 | Royal Archive Access Verification | FOL, clause form, Resolution Refutation | Prolog (SWI-Prolog) |

## Team

| Name | Roll No. |
|------|----------|
| Joshitha Reddy Bangaru | B24DS009 |
| Challa Sanjana | B24ME012 |
| Y.Lukesh | B24DS039 |
| D.Prem Charan | B24DS008 |
| V.Vishal | B24EC036 |

---

## Repository structure

```
.
├── README.md
├── Problem1_BayesianNetwork/
│   ├── Algo_1.ipynb          # Parts A-E: code, outputs, plots
│   └── Algo_1.pdf            # Exported notebook with all outputs
├── Problem2_MarsRover_PDDL/
│   ├── Domain.pddl
│   ├── Problem.pddl
│   ├── Problem_extended.pddl
│   ├── scenario1_gbfs.txt    # Planner output: original, satisficing
│   ├── scenario1_astar.txt   # Planner output: original, optimal
│   ├── extended_gbfs.txt     # Planner output: extended, satisficing
│   ├── extended_astar.txt    # Planner output: extended, optimal
│   └── Analysis_Report.md
└── Problem3_RoyalArchive/
    └── royal_archive_access.pl
```

---

## Problem 1: Smart Home Energy Consumption (Bayesian Network)

Models how weather, day type, occupancy, AC usage, and appliance load drive household energy consumption, then compares the expert-defined network with networks learned from synthetic data.

### Network structure

```
Weather ──────┐
              ▼
DayType ──► Occupancy ──► AC ──────┐
                │                  ▼
                └────► ApplianceLoad ──► EnergyConsumption
```

Edges: `W→A`, `D→O`, `O→A`, `O→L`, `A→E`, `L→E`. All CPTs are verified programmatically (`check_model()` is `True`, and every parent configuration sums to 1).

### Part A: Expert-model inference

| Query | Probability |
|-------|-------------|
| Q1 `P(E=High)` | 0.5806 |
| Q2 `P(E=High \| W=Hot)` | 0.7063 |
| Q3 `P(E=High \| D=Weekend)` | 0.6402 |
| Q4 `P(E=High \| W=Cold, O=High)` | 0.4843 |
| Q5 `P(W=Hot \| E=High)` | 0.6082 |
| Q6 `P(AC=On \| E=High)` | 0.6929 |
| Q7 `P(AC=On \| E=High, L=Heavy)` | 0.6499 |

**Explaining away (Q6 vs Q7).** Observing heavy appliance load already offers a competing explanation for high energy use, so the evidence for the AC being on weakens: `0.6929 → 0.6499`.

**Hand calculation of Q1.** Marginalising over AC and Load:

```
Q1 = 0.30088·0.92 + 0.18455·0.68 + 0.27112·0.55 + 0.24345·0.12 = 0.580631
```

This matches the code output exactly (verified to 1e-9).

### Part B: Conditional independence (d-separation)

| Statement | Holds? | Numeric check |
|-----------|--------|---------------|
| W ⊥ D | Yes | `P(W=Hot) = 0.5000` vs `P(W=Hot \| D=Weekend) = 0.5000` |
| W ⊥ D \| E | **No** | `0.6082` vs `0.5998`. Conditioning on the collider-descendant `E` opens the path `W→A←O←D`. |
| W ⊥ L \| O | Yes | `0.5000` vs `0.5000`. Observing `O` blocks the path `W→A←O→L`. |

### Part C: Synthetic data (ancestral sampling)

- Implemented by hand: nodes are sampled in topological order with `numpy.random.default_rng`.
- **Seed: `42`**, so results are fully reproducible.
- Dataset sizes: **500, 5,000, 20,000**.
- Weather frequencies converge to the prior as `n` grows (Hot: 51.0% → 49.74% → 49.90%).

### Part D: Parameter learning

For each dataset, the same structure is fitted with:

- **MLE** (`MaximumLikelihoodEstimator`)
- **Bayesian estimation** (`BayesianEstimator`, BDeu prior, equivalent sample size = 10)

The notebook prints the learned AC CPT side by side with the expert CPT for every dataset size, then re-runs Q1–Q7 on each learned model.

### Part E: Expert vs MLE vs Bayesian

Mean absolute difference over Q1–Q7 (vs the expert model):

| Dataset size | MLE | Bayesian (BDeu) |
|-------------:|----:|----------------:|
| 500 | 0.0207 | 0.0245 |
| 5,000 | 0.0036 | 0.0039 |
| 20,000 | 0.0017 | 0.0018 |

Takeaways:

- Error drops sharply with more data for both estimators (roughly 10× from 500 to 20,000 samples).
- At 500 samples, **Q5** has the largest error (MLE 0.0259, BDeu 0.0325).
- In this run MLE was marginally ahead at every size. The BDeu prior smooths sparse CPT rows, which matters most for the rarest parent configurations, but it also pulls estimates toward uniform. See the notebook's conclusion for the full discussion and the error-vs-size plot.

### Running it

```bash
pip install pgmpy numpy pandas matplotlib networkx jupyter
jupyter notebook Problem1_BayesianNetwork/Algo_1.ipynb
```

---

## Problem 2: Mars Rover Planning (PDDL)

The rover **Curiosity** (two bays, one drill) collects five rock samples across seven locations and must return them all to Base, finishing with empty bays and the rover at Base. Hard rocks need the drill; `SiteC → SiteD` is a one-way slope.

### Modelling

- **Requirements:** `:strips :typing` only (no negative preconditions, no numeric fluents).
- **Types:** `location`, `rock`, `bay`.
- **Actions:** `move`, `pick-drill`, `collect-soft`, `collect-hard` (requires `has-drill`), `deposit`.
- **One-way route:** encoded with directed `connected` facts. `sitec → sited` exists and `sited → sitec` does not.

### Results

| Scenario | Configuration | Plan length | States expanded | Time |
|----------|---------------|------------:|----------------:|-----:|
| Original | GBFS (h_add) | 34 | 77 | 0.17 s |
| Original | **A\* (admissible)** | **25** | 1,476 | 0.015 s |
| Original | BFS (cross-check) | 25 | 332,568 | 7.1 s |
| Extended | GBFS (h_add) | 61 | 721 | 1.4 s |
| Extended | **A\* (admissible)** | **40** | 612,873 | 13.3 s |

All actions have unit cost, so plan length = cost = number of actions.

### Highlights

- **Greedy vs optimal:** GBFS only ever uses `bay1`, which causes many extra round trips. The optimal plan fills both bays on each main trip.
- **Drill timing (optimal, original):** picked up at **SiteE** (action 11), only when the first hard rock requires it.
- **Extended scenario** (Rock6 soft at SiteE, Rock7 hard at SiteF, Base↔SiteB closed): SiteB is reachable only via SiteD, so the optimal plan goes `Base → A → C → D` using the one-way slope in its permitted direction, then `D → B`. It returns via `D → E` and never needs `D → C`.
- **Goal verification:** every plan was checked by step-by-step simulation. The final state has all rocks at Base, both bays empty, and the rover at Base.

Full discussion: [`Analysis_Report.md`](Problem2_MarsRover_PDDL/Analysis_Report.md)

### Reproducing with Fast Downward

```bash
# Satisficing
fast-downward.py Domain.pddl Problem.pddl --search "eager_greedy([add()])"
# Optimal
fast-downward.py Domain.pddl Problem.pddl --search "astar(lmcut())"
# Extended scenario: swap in Problem_extended.pddl
```

> **Note:** the statistics above come from custom Python implementations of GBFS (h_add) and A* run on the grounded domain, not from Fast Downward itself, so expansion counts and timings will differ slightly when using it. Plan lengths for the optimal configuration should match.

---

## Problem 3: Royal Archive Access Verification (Resolution Refutation)

Proves **`trusted(mira)`** from the archive's rules using resolution refutation in first-order logic, implemented in Prolog.

### Knowledge base

| Rule | Clause form |
|------|-------------|
| Archivist who verifies a record is trusted | `¬archivist(X) ∨ ¬verifies(X,R) ∨ trusted(X)` |
| Trusted person with access to a sealed record may inspect it | `¬trusted(X) ∨ ¬has_access(X,R) ∨ ¬sealed(R) ∨ permitted_inspect(X,R)` |
| Inspecting a record and finding a hidden mark means verifying it | `¬inspects(X,R) ∨ ¬hidden_mark(X,R) ∨ verifies(X,R)` |
| Records in the royal archive are sealed | `¬stored_in_royal_archive(R) ∨ sealed(R)` |

Facts: `stored_in_royal_archive(scroll_7)`, `has_access(mira, scroll_7)`, `inspects(mira, scroll_7)`, `hidden_mark(mira, scroll_7)`, `archivist(mira)`.
Negated goal: `¬trusted(mira)`. No existential quantifiers occur, so no Skolem terms are needed.

### Proof chain

1. Resolve rule 1 with `archivist(mira)`, giving `¬verifies(mira,R) ∨ trusted(mira)`.
2. Resolve with `¬trusted(mira)`, giving `¬verifies(mira,R)`.
3. Resolve rule 3 with `inspects(mira, scroll_7)`, giving `¬hidden_mark(mira,scroll_7) ∨ verifies(mira,scroll_7)`.
4. Resolve step 3 with step 2, giving `¬hidden_mark(mira,scroll_7)`.
5. Resolve with `hidden_mark(mira, scroll_7)`, giving **`[]` (empty clause)**, so the contradiction is derived and `trusted(mira)` is proved.

Rule 2 (`permitted_inspect`) and rule 4 (`sealed`) are not needed for this goal. They are distractors, as the problem statement hints.

### Running it

```bash
swipl -q -s royal_archive_access.pl -g "prove_mira_trusted,halt"
# Print all clauses:
swipl -q -s royal_archive_access.pl -g "show_cnf,halt"
```

The file follows the required eight-step layout: FOL representation → negated goal → implication removal → negation movement → standardisation → Skolemisation → CNF → resolution.

---

## Tech stack

- Python 3.10+, pgmpy, NumPy, pandas, matplotlib, networkx, Jupyter
- PDDL (STRIPS + typing), Fast Downward-style search
- SWI-Prolog 9+

## Submission contents

Source code, screenshots of outputs and visualisations, and group member names and roll numbers, as required by the assignment guidelines.
