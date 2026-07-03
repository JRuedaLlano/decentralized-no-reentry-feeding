# Decentralized no-reentry access control for automated feeding in socially ranked livestock groups

This repository contains the simulation code used for the paper:

**Decentralized no-reentry access control for automated feeding in socially ranked livestock groups**

The repository is organized around the two numerical parts of the paper:

- **Simulation I**: convergence dynamics under **Policy A**
- **Simulation II**: policy comparison between **Policy A** and **Free access**

The code is split into **runner scripts** and **postprocessing scripts**.

- The **runner scripts** generate the raw simulation outputs.
- The **postprocessing scripts** read those outputs and generate the figures, processed summaries, and LaTeX tables used in the manuscript.

---

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── run_simulation_I_convergence.py
├── postprocess_simulation_I.py
├── run_simulation_II_policy_comparison.py
└── postprocess_simulation_II.py
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
numpy
matplotlib
scipy
```

---

## Reproducibility workflow

The repository has two separate pipelines.

### 1. Simulation I: convergence dynamics

Run the simulation:

```bash
python run_simulation_I_convergence.py
```

This creates the raw Simulation I outputs locally in:

```text
simulation_outputs_convergence/
├── convergence_cell_summaries.csv
└── convergence_raw_times.csv
```

Then run the postprocessing:

```bash
python postprocess_simulation_I.py
```

This creates the processed Simulation I outputs locally in:

```text
simulation_outputs_convergence_postprocessed/
├── convergence_processed_summary.csv
├── convergence_fit_summary.csv
├── tables/
│   ├── table_selected_q95.tex
│   ├── table_fit_comparison_local.tex
│   └── table_fit_comparison_global.tex
└── figures/
    ├── scope_comparison_gamma_0.25.pdf
    ├── scope_comparison_gamma_0.25.png
    ├── scope_comparison_gamma_0.75.pdf
    ├── scope_comparison_gamma_0.75.png
    ├── parameter_and_ratio_combined.pdf
    └── parameter_and_ratio_combined.png
```

### 2. Simulation II: policy comparison

Run the simulation:

```bash
python run_simulation_II_policy_comparison.py
```

This creates the raw Simulation II outputs locally in:

```text
simulation_outputs_policy_comparison/
├── policy_comparison_global_indices.csv
├── policy_comparison_rank_profiles.csv
└── policy_comparison_rank_groups.csv
```

Then run the postprocessing:

```bash
python postprocess_simulation_II.py
```

This creates the processed Simulation II outputs locally in:

```text
simulation_outputs_policy_comparison_postprocessed/
├── policy_global_compact.csv
├── tables/
│   ├── table_policy_global_outcomes_hard.tex
│   ├── table_policy_global_outcomes_favorable.tex
│   ├── table_policy_rank_groups_hard.tex
│   ├── table_policy_rank_groups_favorable.tex
│   └── table_policy_global_ci_j150.tex
└── figures/
    ├── policy_global_indices_hard.pdf
    ├── policy_global_indices_hard.png
    ├── policy_global_indices_favorable.pdf
    ├── policy_global_indices_favorable.png
    ├── policy_rank_coverage_hard.pdf
    ├── policy_rank_coverage_hard.png
    ├── policy_rank_coverage_favorable.pdf
    ├── policy_rank_coverage_favorable.png
    ├── policy_rank_queue_hard.pdf
    ├── policy_rank_queue_hard.png
    ├── policy_rank_queue_favorable.pdf
    └── policy_rank_queue_favorable.png
```

---

## Relation to the paper

### Simulation I

This pipeline reproduces the results on convergence to the favorable absorbing configuration under **Policy A**.

It corresponds to the manuscript section on:

- convergence dynamics,
- empirical \(q_{0.95}\) summaries,
- descriptive finite-range growth comparisons,
- scope and parameter effects.

### Simulation II

This pipeline reproduces the policy comparison between:

- **Policy A**: no re-entry within session,
- **Free access**: re-entry allowed within session.

It corresponds to the manuscript section on:

- feeder utilization,
- ration shortfall,
- queue burden,
- rank-level coverage profiles,
- rank-level queue profiles,
- grouped top/middle/bottom rank summaries,
- bootstrap confidence-interval summaries for the main outcomes at \(J=150\).

---

## Output conventions

### Scope labels

- `local_s1` = local scope with radius `s = 1`
- `global` = global scope

### Policy labels

- `A` = Policy A
- `free` = Free access

---

## Notes on computation

### Simulation I

Simulation I uses adaptive replication by design cell. The script stops each cell when the target precision rule is met or when the maximum replication cap is reached.

As a result:

- the number of replications may vary by cell,
- the final cell-summary file records the realized replication count for each design cell.

### Simulation II

Simulation II uses a fixed simulation design:

- `J ∈ {10, 50, 100, 150}`
- parameter regimes `(\gamma_min, p_min) ∈ {(0.25, 0.55), (0.75, 0.85)}`
- local scope (`s = 1`) and global scope
- policies `A` and `free`
- `100` independent replications per design cell
- `400` simulated days per replication
- `K = 6` sessions per day
- `100` burn-in days
- long-run averages computed over the remaining `300` days

For each reported mean outcome, uncertainty is summarized using a two-sided **95% percentile bootstrap confidence interval** across replications. The bootstrap uses:

- the replication-level sample mean as the bootstrap statistic,
- `2000` bootstrap resamples,
- the empirical `2.5%` and `97.5%` quantiles as interval endpoints.

---

## Data and outputs

The repository contains only the code required to generate the simulation outputs. No external input data are required. All CSV files, figures, and LaTeX tables are created locally when the scripts are executed.

In particular, the manuscript-level Simulation II uncertainty summary at \(J=150\) is generated automatically by the postprocessing pipeline as:

```text
tables/table_policy_global_ci_j150.tex
```

---

## Suggested execution order

For full reproduction:

```bash
python run_simulation_I_convergence.py
python postprocess_simulation_I.py
python run_simulation_II_policy_comparison.py
python postprocess_simulation_II.py
```

---

## Citation

If you use this code, please cite the associated paper.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For questions about the code or manuscript, please contact:

**José Rueda-Llano**  
Friedrich Schiller University Jena  
Email: jose.rueda.llano@uni-jena.de
