# Modeling of Processes and Smart Systems

Python laboratory implementations and computational modeling tools for the university course **Modeling of Processes and Smart Systems** (*Моделювання процесів і смарт-систем*).

---

## 1. Course Overview

This repository provides reproducible computational implementations across fundamental system modeling and simulation paradigms:

1. **Static Black-Box Identification:** Multi-factor linear regression using Ordinary Least Squares (OLS) via normal equations and matrix decomposition.
2. **Dynamic Nonlinear Systems:** Cauchy problems for Lotka–Volterra predator–prey dynamics and Kermack–McKendrick SIR epidemic spread using the classical 4th-order Runge–Kutta (RK4) method with phase-plane analysis.
3. **Spatially Distributed Processes:** One-dimensional transient heat conduction governed by parabolic partial differential equations (PDE), solved using the method of lines (finite-difference spatial discretization with RK4 temporal integration) and validated against analytical Fourier series solutions.
4. **Discrete-Event Systems:** Python implementations for Place/Transition Petri nets, providing state-space exploration, reachability graph construction, boundedness, and liveness analysis to support work conducted in specialized modeling software.
5. **Probabilistic Graphical Models (Lab 6):** Discrete Bayesian Belief Networks with domain-specified directed acyclic graph (DAG) topology, parameter learning via Maximum Likelihood Estimation (MLE), and exact inference using Variable Elimination.

---

## 2. Laboratory Modules

| Lab | Topic | Method / Modeling Paradigm | Example Implementation | Key Verified Result |
|---|---|---|---|---|
| **1** | Black Box Modeling & OLS | Multi-factor Ordinary Least Squares | [Variant 15](labs/01_black_box_lsq/variant_15/src/lab1.py) | $R^2 \approx 0.282914$, $y(1.5, 3.0) \approx 4.413274$ |
| **2** | Dynamic Systems (LV & SIR) | RK4 Numerical Integration & Phase Portraits | [Variant 15](labs/02_dynamic_systems/variant_15/src/lab2_1.py) | Periodic orbits (LV), epidemic peak at $t = 0.8\text{ days}$ (SIR) |
| **3** | Spatially Distributed Processes | Method of Lines (RK4) vs. Analytical Fourier | [Variant 15](labs/03_spatial_processes/variant_15/src/lab3.py) | Maximum absolute deviation $< 0.057^\circ\text{C}$ ($<0.06\%$) |
| **4** | Petri Net Modeling & Analysis | Reachability Graph Generation & Analysis | [Variant 04](labs/04_petri_nets/variant_04/src/lab4.py) | 4 places, 4 transitions, finite 4-state reachability graph |
| **6** | Bayesian Belief Networks | Manually Defined DAG, MLE, Variable Elimination | [Machine Failure](labs/06_bayesian_networks/machine_failure/src/lab6.py) | AI4I 2020 dataset (10k rows), 6 inference scenarios |

> **Note on Course Curriculum:** Laboratory numbering follows the official university course curriculum (Labs 1, 2, 3, 4, and 6).

---

## 3. Repository Architecture

```text
.
├── labs/
│   ├── 01_black_box_lsq/                     # Lab 1: Least Squares Black-Box Modeling
│   │   ├── variant_15/                       # Variant 15 (OOP, residual analysis)
│   │   └── variant_04/                       # Variant 04 (Gaussian elimination)
│   ├── 02_dynamic_systems/                   # Lab 2: Lotka–Volterra and SIR Models
│   │   ├── variant_15/                       # Variant 15 (modular RK4, vector fields)
│   │   └── variant_04/                       # Variant 04 (basic RK4 integration)
│   ├── 03_spatial_processes/                 # Lab 3: Transient Heat Conduction (PDE)
│   │   ├── variant_15/                       # Variant 15 (efficient RK4 solver, Fourier series)
│   │   └── variant_04/                       # Variant 04 (procedural & structured implementations)
│   ├── 04_petri_nets/                        # Lab 4: Discrete Place/Transition Petri Nets
│   │   └── variant_04/                       # Variant 04 (reachability graph generation)
│   └── 06_bayesian_networks/                 # Lab 6: Bayesian Belief Networks
│       ├── machine_failure/                  # Machine failure (AI4I 2020 dataset, 10k rows)
│       ├── machine_failure_prototype/        # Machine failure prototype (AI4I 2020 dataset)
│       └── plant_growth/                     # Plant growth (agricultural dataset, 193 rows)
├── .gitignore                                # Git ignore configuration
└── README.md                                 # Public documentation
```

---

## 4. Installation and Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment (`venv`)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Eclipse-XS/System-Modeling-and-Simulation.git
cd System-Modeling-and-Simulation

# Create virtual environment
python -m venv .venv
```

**Activate virtual environment:**

- **Windows PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

**Install verified dependencies:**
```bash
pip install numpy scipy matplotlib networkx pgmpy tabulate pandas
```

---

## 5. Execution Examples

All scripts are self-contained and executable from the repository root:

```powershell
# Lab 1: Black Box Modeling via OLS (Variant 15)
python labs/01_black_box_lsq/variant_15/src/lab1.py

# Lab 2: Dynamic Systems - Lotka-Volterra & SIR Models (Variant 15)
python labs/02_dynamic_systems/variant_15/src/lab2_1.py
python labs/02_dynamic_systems/variant_15/src/lab2_2.py

# Lab 3: Spatial Processes - Heat Conduction PDE (Variant 15)
python labs/03_spatial_processes/variant_15/src/lab3.py

# Lab 4: Petri Nets Analysis (Variant 04)
python labs/04_petri_nets/variant_04/src/lab4.py

# Lab 6: Bayesian Network Probabilistic Inference (Machine Failure)
python labs/06_bayesian_networks/machine_failure/src/lab6.py
```

---

## 6. Scientific Methodology and Verification

### Lab 1: Least Squares Verification
For Variant 15, the empirical dataset (8 observation points) yields the following verified parameters:
- Regression coefficients: $a_0 \approx 5.905310$, $a_1 \approx 1.461947$, $a_2 \approx -1.228319$
- Linear model equation:
  $$y = 5.905310 + 1.461947 \cdot x_1 - 1.228319 \cdot x_2$$
- Sum of Squared Errors: $\text{SSE} \approx 60.249558$
- Total Sum of Squares: $\text{SST} = 84.020000$
- Coefficient of Determination: $R^2 = 1 - \frac{\text{SSE}}{\text{SST}} \approx 0.282914$
- Control point prediction: $y(1.5, 3.0) \approx 4.413274$

*Interpretation:* The linear model achieves an $R^2$ of $0.2829$, reflecting moderate fit for a first-order planar approximation over the assigned experimental data.

### Lab 4: Petri Net Modeling
The scripts in `labs/04_petri_nets/` provide programmatic analysis for Petri nets modeled during coursework in specialized software environments. The implementation:
- Represents the incidence matrix $C = O - I$.
- Computes transition firing rules under the enabling condition $M \ge I(t)$.
- Performs breadth-first reachability tree traversal to extract unique reachable markings.
- Evaluates boundedness (maximum tokens per place) and deadlocks (states with no enabled transitions).

### Lab 6: Bayesian Network Methodology
The Bayesian Network implementation in `labs/06_bayesian_networks/machine_failure/src/lab6.py`:
- **DAG Construction:** The 7-edge DAG structure is specified manually based on physical and operational failure dependencies:

  ```text
  Type -> Torque_cat
  Type -> Speed_cat
  Air_temp_cat -> Process_temp_cat
  Torque_cat -> Machine failure
  Wear_cat -> Machine failure
  Speed_cat -> Machine failure
  Process_temp_cat -> Machine failure
  ```

- **Parameter Learning:** Conditional Probability Distributions (CPDs) are estimated from 10,000 discretized records using `MaximumLikelihoodEstimator`.
- **Probabilistic Inference:** Exact inference is computed using `VariableElimination` across 6 operational scenarios.
- **Model Validation:** The DAG satisfies structural acyclicity and probability distribution axioms (`check_model()` passes).

---

## 7. Datasets and Licensing

1. **AI4I 2020 Predictive Maintenance Dataset** (`labs/06_bayesian_networks/machine_failure/data/machine_failure.csv`):
   - **Source:** UCI Machine Learning Repository (Matzka, S., 2020; Dua, D. and Graff, C., 2019).
   - **Records:** 10,000 synthetic machine operation rows reflecting real predictive maintenance telemetry.
   - **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

2. **Plant Growth Dataset** (`labs/06_bayesian_networks/plant_growth/data/plant_growth_data.csv`):
   - **Source:** Kaggle open dataset.
   - **Records:** 193 agricultural test records.
   - **License:** Public Domain / [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

---

## 8. Authorship and Provenance

- **Variant 15:** Implementations for Labs 1, 2, 3, and 6.
- **Variant 04:** Implementations for Labs 1, 2, 3, and 4 (including the Lab 4 Petri Net state-space model).
- **Machine Failure / Plant Growth:** Bayesian Network implementations based on the AI4I 2020 and Kaggle Plant Growth datasets.

---

## 9. Methodological Limitations

1. **Lab 4 Software Scope:** The Python implementations in Lab 4 serve as analytical supplements and visualizers; they do not replace dedicated graphical Petri net simulation environments used in the course curriculum.
2. **Lab 6 Data Discretization & Target Imbalance:** Machine failure is a rare event in the AI4I dataset (approx. 3.39% failures). Discretization into 3 quantile bins per continuous variable creates multi-parent conditional probability tables with some sparse configurations. Specific post-failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are intentionally excluded from predictive query evidence to avoid target leakage.
3. **Plant Growth Sample Size in Lab 6 (plant_growth):** The secondary plant growth model contains 243 parent state configurations against only 193 observation rows, causing parameter sparsity. The machine failure network provides higher sample density for probabilistic inference compared to the plant growth dataset.
