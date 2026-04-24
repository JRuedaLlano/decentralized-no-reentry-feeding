from __future__ import annotations

# ============================================================
# Simulation II postprocessing:
#   - reads policy-comparison outputs
#   - exports compact CSV summaries
#   - exports LaTeX tables used in the paper
#   - exports publication figures used in the paper
# ============================================================

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1) Utilities
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_rows(path: str | Path) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x) -> float:
    if x is None or x == "":
        return np.nan
    return float(x)


def to_int(x) -> int:
    return int(float(x))


def format_scope_label(scope: str) -> str:
    return r"Local scope ($s=1$)" if scope == "local_s1" else r"Global scope"


def format_scope_short(scope: str) -> str:
    return "Local" if scope == "local_s1" else "Global"


def format_policy_label(policy: str) -> str:
    return "Policy A" if policy == "A" else "Free access"


# ============================================================
# 2) Data containers
# ============================================================

@dataclass
class GlobalIndexRow:
    policy: str
    scope: str
    gamma_min: float
    p_min: float
    J: int
    n_rep: int
    U_feed_mean: float
    V_P4_mean: float  # kept for compatibility with runner output
    V_P1_mean: float
    V_P5_mean: float


@dataclass
class RankProfileRow:
    policy: str
    scope: str
    gamma_min: float
    p_min: float
    J: int
    rank: int
    relative_rank: float
    coverage_mean: float
    queue_fraction_mean: float


@dataclass
class RankGroupRow:
    policy: str
    scope: str
    gamma_min: float
    p_min: float
    J: int
    group_name: str
    coverage_mean: float
    queue_fraction_mean: float


# ============================================================
# 3) Load data
# ============================================================

def load_global_indices(path: str | Path) -> List[GlobalIndexRow]:
    rows = read_csv_rows(path)
    out: List[GlobalIndexRow] = []

    for r in rows:
        out.append(
            GlobalIndexRow(
                policy=r["policy"],
                scope=r["scope"],
                gamma_min=to_float(r["gamma_min"]),
                p_min=to_float(r["p_min"]),
                J=to_int(r["J"]),
                n_rep=to_int(r["n_rep"]),
                U_feed_mean=to_float(r["U_feed_mean"]),
                V_P4_mean=to_float(r["V_P4_mean"]),
                V_P1_mean=to_float(r["V_P1_mean"]),
                V_P5_mean=to_float(r["V_P5_mean"]),
            )
        )
    return out


def load_rank_profiles(path: str | Path) -> List[RankProfileRow]:
    rows = read_csv_rows(path)
    out: List[RankProfileRow] = []

    for r in rows:
        out.append(
            RankProfileRow(
                policy=r["policy"],
                scope=r["scope"],
                gamma_min=to_float(r["gamma_min"]),
                p_min=to_float(r["p_min"]),
                J=to_int(r["J"]),
                rank=to_int(r["rank"]),
                relative_rank=to_float(r["relative_rank"]),
                coverage_mean=to_float(r["coverage_mean"]),
                queue_fraction_mean=to_float(r["queue_fraction_mean"]),
            )
        )
    return out


def load_rank_groups(path: str | Path) -> List[RankGroupRow]:
    rows = read_csv_rows(path)
    out: List[RankGroupRow] = []

    for r in rows:
        out.append(
            RankGroupRow(
                policy=r["policy"],
                scope=r["scope"],
                gamma_min=to_float(r["gamma_min"]),
                p_min=to_float(r["p_min"]),
                J=to_int(r["J"]),
                group_name=r["group_name"],
                coverage_mean=to_float(r["coverage_mean"]),
                queue_fraction_mean=to_float(r["queue_fraction_mean"]),
            )
        )
    return out


# ============================================================
# 4) Helpers for subsetting
# ============================================================

def subset_global(
    rows: Sequence[GlobalIndexRow],
    scope: str,
    gamma_min: float,
    p_min: float,
    policy: str,
) -> List[GlobalIndexRow]:
    subset = [
        r for r in rows
        if r.scope == scope
        and r.gamma_min == gamma_min
        and r.p_min == p_min
        and r.policy == policy
    ]
    subset.sort(key=lambda r: r.J)
    return subset


def subset_rank_profiles(
    rows: Sequence[RankProfileRow],
    scope: str,
    gamma_min: float,
    p_min: float,
    J: int,
    policy: str,
) -> List[RankProfileRow]:
    subset = [
        r for r in rows
        if r.scope == scope
        and r.gamma_min == gamma_min
        and r.p_min == p_min
        and r.J == J
        and r.policy == policy
    ]
    subset.sort(key=lambda r: r.rank)
    return subset


def subset_rank_groups(
    rows: Sequence[RankGroupRow],
    scope: str,
    gamma_min: float,
    p_min: float,
    J: int,
    policy: str,
) -> List[RankGroupRow]:
    subset = [
        r for r in rows
        if r.scope == scope
        and r.gamma_min == gamma_min
        and r.p_min == p_min
        and r.J == J
        and r.policy == policy
    ]
    order = {"top": 0, "middle": 1, "bottom": 2}
    subset.sort(key=lambda r: order.get(r.group_name, 999))
    return subset


# ============================================================
# 5) CSV export
# ============================================================

def export_compact_global_csv(
    rows: Sequence[GlobalIndexRow],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / "policy_global_compact.csv"

    regimes = sorted({
        (r.scope, r.gamma_min, r.p_min, r.policy)
        for r in rows
    })
    index_map = {(r.scope, r.gamma_min, r.p_min, r.policy, r.J): r for r in rows}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy", "scope", "gamma_min", "p_min",
            "U_feed_J10", "U_feed_J150",
            "V_P1_J10", "V_P1_J150",
            "V_P5_J10", "V_P5_J150",
        ])

        for scope, gamma_min, p_min, policy in regimes:
            r10 = index_map[(scope, gamma_min, p_min, policy, 10)]
            r150 = index_map[(scope, gamma_min, p_min, policy, 150)]
            writer.writerow([
                policy, scope, gamma_min, p_min,
                r10.U_feed_mean, r150.U_feed_mean,
                r10.V_P1_mean, r150.V_P1_mean,
                r10.V_P5_mean, r150.V_P5_mean,
            ])


# ============================================================
# 6) LaTeX tables
# ============================================================

def export_latex_global_outcomes_table_by_regime(
    rows: Sequence[GlobalIndexRow],
    outdir: str | Path,
    gamma_min: float,
    p_min: float,
    filename: str,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / filename

    index_map = {(r.scope, r.gamma_min, r.p_min, r.policy, r.J): r for r in rows}
    scopes = ["local_s1", "global"]

    lines = []
    lines.append("\\begin{tabular}{lllrrr}")
    lines.append("\\toprule")
    lines.append("Policy & Scope & $J$ & $U_{\\text{feed}}$ & $V_{P1}$ & $V_{P5}$ \\\\")
    lines.append("\\midrule")

    for scope in scopes:
        for policy in ["A", "free"]:
            for J in [10, 150]:
                r = index_map[(scope, gamma_min, p_min, policy, J)]
                lines.append(
                    f"{format_policy_label(policy)} & {format_scope_short(scope)} & {J} & "
                    f"{r.U_feed_mean:.3f} & {r.V_P1_mean:.3f} & {r.V_P5_mean:.3f} \\\\"
                )
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_latex_rank_group_table_by_regime(
    rows: Sequence[RankGroupRow],
    outdir: str | Path,
    gamma_min: float,
    p_min: float,
    filename: str,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / filename

    lines = []
    lines.append("\\begin{tabular}{llllrr}")
    lines.append("\\toprule")
    lines.append("Policy & Scope & Group & $J$ & Coverage & Queue \\\\")
    lines.append("\\midrule")

    for scope in ["local_s1", "global"]:
        for policy in ["A", "free"]:
            subset = subset_rank_groups(rows, scope, gamma_min, p_min, 150, policy)
            for r in subset:
                lines.append(
                    f"{format_policy_label(policy)} & {format_scope_short(scope)} & "
                    f"{r.group_name} & 150 & {r.coverage_mean:.3f} & {r.queue_fraction_mean:.3f} \\\\"
                )
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 7) Figures
# ============================================================

def plot_global_indices(
    rows: Sequence[GlobalIndexRow],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)

    regimes = [
        (0.25, 0.55, "hard"),
        (0.75, 0.85, "favorable"),
    ]
    metric_info = [
        ("U_feed_mean", r"$U_{\mathrm{feed}}$"),
        ("V_P1_mean", r"$V_{P1}$"),
        ("V_P5_mean", r"$V_{P5}$"),
    ]

    for gamma_min, p_min, tag in regimes:
        fig, axes = plt.subplots(3, 2, figsize=(9.5, 8), sharex=True)

        for col, scope in enumerate(["local_s1", "global"]):
            subset_A = subset_global(rows, scope, gamma_min, p_min, "A")
            subset_F = subset_global(rows, scope, gamma_min, p_min, "free")

            J_A = np.array([r.J for r in subset_A], dtype=float)
            J_F = np.array([r.J for r in subset_F], dtype=float)

            for row_idx, (field, ylabel) in enumerate(metric_info):
                ax = axes[row_idx, col]
                yA = np.array([getattr(r, field) for r in subset_A], dtype=float)
                yF = np.array([getattr(r, field) for r in subset_F], dtype=float)

                ax.plot(J_A, yA, marker="o", label="Policy A")
                ax.plot(J_F, yF, marker="s", label="Free access")

                if row_idx == 0:
                    ax.set_title(format_scope_label(scope))
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row_idx == 2:
                    ax.set_xlabel("J")

                ax.grid(True, alpha=0.3)

        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(rf"$\gamma_{{\min}}={gamma_min:.2f},\ p_{{\min}}={p_min:.2f}$", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(outdir / f"policy_global_indices_{tag}.pdf", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(outdir / f"policy_global_indices_{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def plot_rank_profiles(
    rows: Sequence[RankProfileRow],
    outdir: str | Path,
    variable: str,
    filename_stub: str,
    ylabel: str,
) -> None:
    outdir = ensure_dir(outdir)

    regimes = [
        (0.25, 0.55, "hard"),
        (0.75, 0.85, "favorable"),
    ]

    for gamma_min, p_min, tag in regimes:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)

        for ax, scope in zip(axes, ["local_s1", "global"]):
            subset_A = subset_rank_profiles(rows, scope, gamma_min, p_min, 150, "A")
            subset_F = subset_rank_profiles(rows, scope, gamma_min, p_min, 150, "free")

            xA = np.array([r.relative_rank for r in subset_A], dtype=float)
            xF = np.array([r.relative_rank for r in subset_F], dtype=float)

            yA = np.array([getattr(r, variable) for r in subset_A], dtype=float)
            yF = np.array([getattr(r, variable) for r in subset_F], dtype=float)

            ax.plot(xA, yA, label="Policy A")
            ax.plot(xF, yF, label="Free access")
            ax.set_title(format_scope_label(scope))
            ax.set_xlabel("Relative rank")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(ylabel)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(rf"$\gamma_{{\min}}={gamma_min:.2f},\ p_{{\min}}={p_min:.2f},\ J=150$", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(outdir / f"{filename_stub}_{tag}.pdf", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(outdir / f"{filename_stub}_{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


# ============================================================
# 8) Master export
# ============================================================

def export_all_policy_postprocessed_outputs(
    global_rows: Sequence[GlobalIndexRow],
    rank_profile_rows: Sequence[RankProfileRow],
    rank_group_rows: Sequence[RankGroupRow],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)

    export_compact_global_csv(global_rows, outdir)

    table_dir = ensure_dir(outdir / "tables")
    export_latex_global_outcomes_table_by_regime(
        global_rows,
        table_dir,
        gamma_min=0.25,
        p_min=0.55,
        filename="table_policy_global_outcomes_hard.tex",
    )
    export_latex_global_outcomes_table_by_regime(
        global_rows,
        table_dir,
        gamma_min=0.75,
        p_min=0.85,
        filename="table_policy_global_outcomes_favorable.tex",
    )
    export_latex_rank_group_table_by_regime(
        rank_group_rows,
        table_dir,
        gamma_min=0.25,
        p_min=0.55,
        filename="table_policy_rank_groups_hard.tex",
    )
    export_latex_rank_group_table_by_regime(
        rank_group_rows,
        table_dir,
        gamma_min=0.75,
        p_min=0.85,
        filename="table_policy_rank_groups_favorable.tex",
    )

    fig_dir = ensure_dir(outdir / "figures")
    plot_global_indices(global_rows, fig_dir)
    plot_rank_profiles(
        rank_profile_rows,
        fig_dir,
        variable="coverage_mean",
        filename_stub="policy_rank_coverage",
        ylabel="Mean coverage",
    )
    plot_rank_profiles(
        rank_profile_rows,
        fig_dir,
        variable="queue_fraction_mean",
        filename_stub="policy_rank_queue",
        ylabel="Mean queue fraction",
    )


# ============================================================
# 9) Main
# ============================================================

if __name__ == "__main__":
    GLOBAL_PATH = "simulation_outputs_policy_comparison/policy_comparison_global_indices.csv"
    RANK_PROFILES_PATH = "simulation_outputs_policy_comparison/policy_comparison_rank_profiles.csv"
    RANK_GROUPS_PATH = "simulation_outputs_policy_comparison/policy_comparison_rank_groups.csv"
    OUTDIR = "simulation_outputs_policy_comparison_postprocessed"

    global_rows = load_global_indices(GLOBAL_PATH)
    rank_profile_rows = load_rank_profiles(RANK_PROFILES_PATH)
    rank_group_rows = load_rank_groups(RANK_GROUPS_PATH)

    export_all_policy_postprocessed_outputs(
        global_rows=global_rows,
        rank_profile_rows=rank_profile_rows,
        rank_group_rows=rank_group_rows,
        outdir=OUTDIR,
    )

    print(
        f"\nSimulation II postprocessing finished. Outputs written to ./{OUTDIR}/"
    )