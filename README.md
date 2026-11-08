# System Modeling and Simulation

Academic laboratory collection for the **System Modeling and Simulation** university course.  
This repository contains analytical, numerical, simulation, and probabilistic models investigating the behavior of complex systems.

---

## 1. Project Overview

The repository encompasses practical implementations covering fundamental modeling paradigms:
1. **Static Black-Box Identification:** Multi-parameter linear regression via Ordinary Least Squares (OLS) with normal equations and matrix decomposition.
2. **Dynamic Nonlinear Systems:** Cauchy problems for Lotka–Volterra predator–prey dynamics and Kermack–McKendrick SIR epidemic spread using the classical 4th-order Runge–Kutta (RK4) method.
3. **Spatially Distributed Processes:** One-dimensional transient heat conduction modeled by parabolic partial differential equations (PDE) using the method of lines (finite difference spatial discretization coupled with RK4 time integration), validated against analytical Fourier series.
4. **Discrete-Event Systems:** Place/Transition Petri nets with reachability graph generation, boundedness, liveness, and deadlock analysis.
5. **Queueing Systems (GPSS World):** Discrete-event queueing simulation (**Status: Not Implemented / Specification Only**).
6. **Probabilistic Graphical Models:** Bayesian Belief Networks (BBN) implementing DAG structural learning, parameter estimation via Maximum Likelihood Estimation (MLE), Bayesian Dirichlet equivalent uniform (BDeu) scoring, and exact variable elimination inference.

---

## 2. Laboratory Status and Canonical Implementations

| # | Laboratory Assignment | Method / Paradigm | Canonical Entry Point | Implementation Status | Scientific Evaluation |
|---|---|---|---|:---:|:---:|
| **1** | Black-Box Modeling & OLS | Multi-factor OLS, Normal Equations | [`labs/01_black_box_lsq/variant_15/src/lab1.py`](labs/01_black_box_lsq/variant_15/src/lab1.py) | **Complete** | $R^2 = 0.99997$ (Variant 15) |
| **2** | Dynamic Systems (LV & SIR) | RK4 Numerical Integration | [`labs/02_dynamic_systems/variant_15/src/lab2_1.py`](labs/02_dynamic_systems/variant_15/src/lab2_1.py)<br>[`labs/02_dynamic_systems/variant_15/src/lab2_2.py`](labs/02_dynamic_systems/variant_15/src/lab2_2.py) | **Complete** | Phase portraits & peak analysis |
| **3** | Spatial Processes (Heat PDE) | Method of Lines (RK4) vs. Fourier Series | [`labs/03_spatial_processes/variant_15/src/lab3.py`](labs/03_spatial_processes/variant_15/src/lab3.py) | **Complete** | Max error $< 0.057^\circ\text{C}$ ($<0.06\%$) |
| **4** | Petri Nets | Reachability Graph, Liveness & Invariants | [`labs/04_petri_nets/variant_04/src/lab4.py`](labs/04_petri_nets/variant_04/src/lab4.py) | **Complete** | 4 places, finite reachability graph |
| **5** | Queueing Systems (GPSS) | Discrete-Event GPSS World Simulation | [`labs/05_queueing_systems/README.md`](labs/05_queueing_systems/README.md) | 🔴 **NOT IMPLEMENTED** | Detailed specification only |
| **6** | Bayesian Networks | DAG Structure Learning, MLE, Inference | [`labs/06_bayesian_networks/machine_failure/src/lab6.py`](labs/06_bayesian_networks/machine_failure/src/lab6.py) | **Complete** | AI4I dataset (10k rows), d-separation |

> **Academic Note on Readiness:** Implementations are developed for educational research and university coursework. They are not intended as production-grade enterprise simulation engines.

---

## 3. Repository Architecture

```text
.
├── labs/
│   ├── 01_black_box_lsq/                     # Lab 1: Least Squares Black-Box Modeling
│   │   ├── variant_15/ (CANONICAL)           # OOP implementation, tabulate metrics, 3D residuals
│   │   └── variant_04/                       # Procedural reference (Gaussian elimination)
│   ├── 02_dynamic_systems/                   # Lab 2: Lotka–Volterra and SIR Epidemic Models
│   │   ├── variant_15/ (CANONICAL)           # Modular architecture, quiver field, peak analysis
│   │   └── variant_04/                       # Basic RK4 implementation
│   ├── 03_spatial_processes/                 # Lab 3: Transient Heat Conduction (PDE)
│   │   ├── variant_15/ (CANONICAL)           # Efficient RK4 solver (3,284 steps, 2 s), Fourier comparison
│   │   └── variant_04/                       # Procedural & structured implementations (1.8M steps)
│   ├── 04_petri_nets/                        # Lab 4: Discrete Place/Transition Petri Nets
│   │   └── variant_04/ (CANONICAL)           # Reachability graph generation, NetworkX visualization
│   ├── 05_queueing_systems/                  # Lab 5: Queueing Systems & GPSS Simulation
│   │   └── README.md                         # Formal specification & modeling assignment (unimplemented)
│   └── 06_bayesian_networks/                 # Lab 6: Bayesian Belief Networks
│       ├── machine_failure/ (CANONICAL)      # AI4I 2020 dataset (10k rows), 6 inference scenarios
│       ├── machine_failure_prototype/        # Early exploratory prototype
│       └── plant_growth/                     # Small-sample experimental model (193 rows)
├── .gitignore                                # Environment, IDE, and private file exclusions
└── README.md                                 # Public documentation
```

---

## 4. Installation and Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment tool (`venv`)

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/Eclipse-XS/System-Modeling-and-Simulation.git
cd System-Modeling-and-Simulation

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install required dependencies
pip install numpy scipy matplotlib networkx pgmpy tabulate pandas
```

---

## 5. Execution Examples

All scripts are self-contained and can be executed either from the repository root or from their respective directories:

```bash
# 1. Lab 1: Least Squares Black Box Modeling (Variant 15)
python labs/01_black_box_lsq/variant_15/src/lab1.py

# 2. Lab 2: Dynamic Systems (Lotka-Volterra & SIR) (Variant 15)
python labs/02_dynamic_systems/variant_15/src/lab2_1.py
python labs/02_dynamic_systems/variant_15/src/lab2_2.py

# 3. Lab 3: Spatial Processes & Heat Conduction (Variant 15)
python labs/03_spatial_processes/variant_15/src/lab3.py

# 4. Lab 4: Petri Nets Analysis (Variant 04)
python labs/04_petri_nets/variant_04/src/lab4.py

# 5. Lab 6: Bayesian Network Probabilistic Inference (AI4I Machine Failure)
python labs/06_bayesian_networks/machine_failure/src/lab6.py
```

---

## 6. Datasets and Licensing

The repository includes public, open-access datasets required for reproducible modeling:

1. **AI4I 2020 Predictive Maintenance Dataset** (`labs/06_bayesian_networks/machine_failure/data/machine_failure.csv`):
   - **Source:** UCI Machine Learning Repository (Matzka, S., 2020; Dua, D. and Graff, C., 2019).
   - **Records:** 10,000 synthetic machine operation rows reflecting real predictive maintenance telemetry.
   - **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

2. **Plant Growth Dataset** (`labs/06_bayesian_networks/plant_growth/data/plant_growth_data.csv`):
   - **Source:** Kaggle open dataset.
   - **Records:** 193 agricultural test records.
   - **License:** Public Domain / [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

---

## 7. Authorship and Provenance

- **Repository Maintainer & Primary Implementations (Variant 15):**  
  The core canonical suite (Labs 1, 2, 3, and 6) was developed under academic Variant 15.
- **Alternative Implementations (Variant 04):**  
  Supplementary implementations for Variant 04 (including canonical Lab 4) are maintained for multi-variant benchmarking and completeness.
- **Third-Party Student Implementations:**  
  Historical coursework reference solutions (e.g., Variant 21 by student Ostap Seniv) were cataloged during internal code auditing and are preserved in local offline archives; they are excluded from the public git repository in compliance with academic copyright and redistribution principles.

---

## 8. Known Limitations and Future Work

1. **Lab 5 (Queueing Systems / GPSS) is Unimplemented:**  
   The project does not contain compiled `.gps` simulation models or execution logs for GPSS World Student. The directory [`labs/05_queueing_systems/README.md`](labs/05_queueing_systems/README.md) provides the formal technical specification for two systems: a multi-pump gas station and a dual-processor sensor queue.
2. **Variant Heterogeneity in Lab 4:**  
   While Labs 1, 2, 3, and 6 correspond to Variant 15, Lab 4 is implemented under Variant 4 due to the absence of a Variant 15 solution in the original source archive. The state-space generation logic in `lab4.py` is fully parameterized and can be adapted to any valid incidence matrix.
3. **Plant Growth CPD Degeneracy in Lab 6 Alternative:**  
   The secondary experimental Bayesian Network (`labs/06_bayesian_networks/plant_growth/`) has 243 conditional parent configurations against only 193 sample rows, resulting in 53% unobserved conditional probability parameters. The canonical machine failure network (`labs/06_bayesian_networks/machine_failure/`) should always be preferred for rigorous probabilistic reasoning.
