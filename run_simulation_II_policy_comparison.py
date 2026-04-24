from __future__ import annotations

# ============================================================
# Simulation II runner:
#   - runs policy-comparison simulations
#   - exports aggregated global outcomes
#   - exports rank profiles
#   - exports grouped rank summaries
# No plotting, no LaTeX export, no extra design variants.
# ============================================================

import csv
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# 1) Data containers
# ============================================================

@dataclass
class PolicyCellSummary:
    policy: str
    scope_label: str
    gamma_min: float
    p_min: float
    J: int
    n_rep: int
    U_feed_mean: float
    V_P4_mean: float
    V_P1_mean: float
    V_P5_mean: float


@dataclass
class RankProfileRecord:
    policy: str
    scope_label: str
    gamma_min: float
    p_min: float
    J: int
    rank: int
    relative_rank: float
    coverage_mean: float
    queue_fraction_mean: float


@dataclass
class RankGroupSummary:
    policy: str
    scope_label: str
    gamma_min: float
    p_min: float
    J: int
    group_name: str
    coverage_mean: float
    queue_fraction_mean: float


# ============================================================
# 2) Utilities
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def choose_winner(
    willing: List[int],
    p_min: float,
    rng: np.random.Generator,
) -> int:
    """
    Highest-ranked willing animal (smallest index) gets mass p_min.
    Remaining mass is split equally across the others.
    """
    if len(willing) == 1:
        return willing[0]

    willing_sorted = sorted(willing)
    others = willing_sorted[1:]

    probs = np.empty(len(willing_sorted), dtype=float)
    probs[0] = p_min
    probs[1:] = (1.0 - p_min) / len(others)

    idx = rng.choice(len(willing_sorted), p=probs)
    return willing_sorted[idx]


def build_scope_neighbors_from_prev_order(
    J: int,
    prev_first_entry_order: np.ndarray,
    scope: Optional[int],
) -> List[List[int]]:
    """
    prev_first_entry_order[animal] = 1..J gives the first-entry order
    in the previous session.

    scope = None -> global
    scope = s    -> local radius s in previous-session first-entry order
    """
    if scope is None:
        return [[k for k in range(J) if k != j] for j in range(J)]

    pos_to_animal = np.empty(J, dtype=int)
    for animal in range(J):
        pos_to_animal[prev_first_entry_order[animal] - 1] = animal

    neighbors = [[] for _ in range(J)]
    for animal in range(J):
        pos = prev_first_entry_order[animal] - 1
        lo = max(0, pos - scope)
        hi = min(J - 1, pos + scope)
        for q in range(lo, hi + 1):
            other = pos_to_animal[q]
            if other != animal:
                neighbors[animal].append(other)
    return neighbors


def initialize_session0(
    J: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Session 0:
      - random first-entry order
      - each animal receives J units once

    Returns:
      prev_first_entry_order : 1..J by animal
      satiation_start_s1     : satiation counter at the start of session 1
      elapsed_start_s1       : elapsed periods since last entry at the start of session 1

    If animal j entered at period p0(j) in session 0, then at the start of session 1:
      satiation = p0(j) - 1
      elapsed   = J - p0(j) + 1
    """
    order = rng.permutation(J) + 1
    satiation = order - 1
    elapsed = J - order + 1
    return order.astype(int), satiation.astype(int), elapsed.astype(int)


def compute_rank_groups(J: int) -> Dict[str, np.ndarray]:
    """
    Rank groups used in the paper.

    If J >= 15:
      top 5, middle, bottom 5
    Else:
      split as evenly as possible into top/middle/bottom preserving order.
    """
    ranks = np.arange(1, J + 1)

    if J >= 15:
        top = ranks[:5]
        bottom = ranks[-5:]
        middle = ranks[5:-5]
    else:
        a = J // 3
        b = J - 2 * a
        top = ranks[:a]
        middle = ranks[a:a + b]
        bottom = ranks[a + b:]

    return {
        "top": top,
        "middle": middle,
        "bottom": bottom,
    }


# ============================================================
# 3) Core Simulation II engine
# ============================================================

def simulate_policy_run(
    J: int,
    gamma_min: float,
    p_min: float,
    scope: Optional[int],
    policy: str,
    rng: np.random.Generator,
    K: int = 6,
    D_total: int = 400,
    burn_in_days: int = 100,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Simulates long-run performance under:
      - policy = "A": no re-entry within session
      - policy = "free": re-entry allowed within session

    Returns:
      global_indices   : dict with U_feed, V_P4, V_P1, V_P5
      coverage_by_rank : shape (J,)
      queue_by_rank    : shape (J,)
    """
    assert policy in {"A", "free"}

    prev_first_entry_order, satiation, elapsed_since_last_entry = initialize_session0(J, rng)

    measurement_days = D_total - burn_in_days
    measurement_sessions = measurement_days * K
    total_feed_dispensed = 0.0

    ration_accum = np.zeros(J, dtype=float)
    queue_periods = np.zeros(J, dtype=float)

    total_periods_measured = measurement_sessions * J
    target_daily_requirement = K * J

    for day in range(D_total):
        for _session_in_day in range(K):
            neighbors = build_scope_neighbors_from_prev_order(J, prev_first_entry_order, scope)

            served_this_session = np.zeros(J, dtype=bool)
            first_entry_time = np.full(J, fill_value=-1, dtype=int)

            for t in range(1, J + 1):
                hungry = []
                for j in range(J):
                    if satiation[j] == 0:
                        if policy == "A":
                            if not served_this_session[j]:
                                hungry.append(j)
                        else:
                            hungry.append(j)

                willing = set(hungry)
                hungry_set = set(hungry)

                for j in range(J):
                    if j in hungry_set:
                        continue
                    if policy == "A" and served_this_session[j]:
                        continue

                    triggered = False
                    for k in neighbors[j]:
                        if (k in hungry_set) and (k > j):
                            if rng.random() < gamma_min:
                                triggered = True
                                break
                    if triggered:
                        willing.add(j)

                willing = sorted(willing)

                if len(willing) > 0:
                    winner = choose_winner(willing, p_min, rng)
                    ration = min(J, int(elapsed_since_last_entry[winner]))

                    if first_entry_time[winner] == -1:
                        first_entry_time[winner] = t

                    if policy == "A":
                        served_this_session[winner] = True

                    if day >= burn_in_days:
                        ration_accum[winner] += ration
                        total_feed_dispensed += ration
                        for j in willing:
                            if j != winner:
                                queue_periods[j] += 1.0

                    satiation[winner] = ration
                    elapsed_since_last_entry[winner] = 0

                for j in range(J):
                    if satiation[j] > 0:
                        satiation[j] -= 1

                elapsed_since_last_entry += 1

            entered = [(animal, first_entry_time[animal]) for animal in range(J) if first_entry_time[animal] != -1]
            entered.sort(key=lambda x: x[1])

            not_entered = [animal for animal in range(J) if first_entry_time[animal] == -1]
            not_entered.sort(key=lambda animal: prev_first_entry_order[animal])

            combined = [a for a, _ in entered] + not_entered
            next_order = np.empty(J, dtype=int)
            for pos, animal in enumerate(combined, start=1):
                next_order[animal] = pos
            prev_first_entry_order = next_order

    coverage_by_rank = ration_accum / (measurement_days * target_daily_requirement)
    queue_by_rank = queue_periods / total_periods_measured

    mean_coverage = float(np.mean(coverage_by_rank))
    V_P1 = 1.0 - mean_coverage

    capacity_per_day = K * (J ** 2)
    mean_daily_dispensed = total_feed_dispensed / measurement_days
    U_feed = mean_daily_dispensed / capacity_per_day
    V_P4 = 1.0 - U_feed

    V_P5 = float(np.mean(queue_by_rank))

    global_indices = {
        "U_feed": float(U_feed),
        "V_P4": float(V_P4),
        "V_P1": float(V_P1),
        "V_P5": float(V_P5),
    }

    return global_indices, coverage_by_rank, queue_by_rank


# ============================================================
# 4) One-cell runner
# ============================================================

def run_policy_comparison_cell(
    J: int,
    gamma_min: float,
    p_min: float,
    scope: Optional[int],
    scope_label: str,
    policy: str,
    n_rep: int,
    base_seed: int = 12345,
    K: int = 6,
    D_total: int = 400,
    burn_in_days: int = 100,
) -> Tuple[PolicyCellSummary, List[RankProfileRecord], List[RankGroupSummary]]:
    global_records = []
    coverage_stack = []
    queue_stack = []

    for rep in range(n_rep):
        rng = np.random.default_rng(base_seed + 100000 * rep + 37 * J)

        global_indices, coverage_by_rank, queue_by_rank = simulate_policy_run(
            J=J,
            gamma_min=gamma_min,
            p_min=p_min,
            scope=scope,
            policy=policy,
            rng=rng,
            K=K,
            D_total=D_total,
            burn_in_days=burn_in_days,
        )

        global_records.append(global_indices)
        coverage_stack.append(coverage_by_rank)
        queue_stack.append(queue_by_rank)

    coverage_arr = np.vstack(coverage_stack)
    queue_arr = np.vstack(queue_stack)

    cell_summary = PolicyCellSummary(
        policy=policy,
        scope_label=scope_label,
        gamma_min=gamma_min,
        p_min=p_min,
        J=J,
        n_rep=n_rep,
        U_feed_mean=float(np.mean([g["U_feed"] for g in global_records])),
        V_P4_mean=float(np.mean([g["V_P4"] for g in global_records])),
        V_P1_mean=float(np.mean([g["V_P1"] for g in global_records])),
        V_P5_mean=float(np.mean([g["V_P5"] for g in global_records])),
    )

    rank_profiles: List[RankProfileRecord] = []
    for rank in range(1, J + 1):
        rank_profiles.append(
            RankProfileRecord(
                policy=policy,
                scope_label=scope_label,
                gamma_min=gamma_min,
                p_min=p_min,
                J=J,
                rank=rank,
                relative_rank=(rank - 1) / (J - 1) if J > 1 else 0.0,
                coverage_mean=float(np.mean(coverage_arr[:, rank - 1])),
                queue_fraction_mean=float(np.mean(queue_arr[:, rank - 1])),
            )
        )

    group_map = compute_rank_groups(J)
    rank_group_summaries: List[RankGroupSummary] = []
    for group_name, ranks in group_map.items():
        idx = ranks - 1
        rank_group_summaries.append(
            RankGroupSummary(
                policy=policy,
                scope_label=scope_label,
                gamma_min=gamma_min,
                p_min=p_min,
                J=J,
                group_name=group_name,
                coverage_mean=float(np.mean(coverage_arr[:, idx])),
                queue_fraction_mean=float(np.mean(queue_arr[:, idx])),
            )
        )

    return cell_summary, rank_profiles, rank_group_summaries


# ============================================================
# 5) Design runner
# ============================================================

def run_policy_comparison_design(
    J_values: Sequence[int],
    param_pairs: Sequence[Tuple[float, float]],
    scopes: Dict[str, Optional[int]],
    policies: Sequence[str],
    n_rep: int,
    base_seed: int = 12345,
    K: int = 6,
    D_total: int = 400,
    burn_in_days: int = 100,
    max_workers: Optional[int] = 2,
) -> Tuple[List[PolicyCellSummary], List[RankProfileRecord], List[RankGroupSummary]]:
    jobs = []
    counter = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for J, (gamma_min, p_min), (scope_label, scope), policy in itertools.product(
            J_values, param_pairs, scopes.items(), policies
        ):
            seed = base_seed + 10000 * counter
            counter += 1

            fut = ex.submit(
                run_policy_comparison_cell,
                J,
                gamma_min,
                p_min,
                scope,
                scope_label,
                policy,
                n_rep,
                seed,
                K,
                D_total,
                burn_in_days,
            )
            jobs.append(fut)

        cell_summaries: List[PolicyCellSummary] = []
        rank_profiles: List[RankProfileRecord] = []
        rank_group_summaries: List[RankGroupSummary] = []

        total_jobs = len(jobs)
        for idx, fut in enumerate(as_completed(jobs), start=1):
            summary, profiles, groups = fut.result()

            cell_summaries.append(summary)
            rank_profiles.extend(profiles)
            rank_group_summaries.extend(groups)

            print(
                f"[{idx}/{total_jobs}] done: policy={summary.policy}, "
                f"scope={summary.scope_label}, J={summary.J}, "
                f"gamma={summary.gamma_min}, p={summary.p_min}, "
                f"U_feed={summary.U_feed_mean:.4f}, "
                f"V_P1={summary.V_P1_mean:.4f}, "
                f"V_P5={summary.V_P5_mean:.4f}"
            )

    cell_summaries.sort(key=lambda x: (x.scope_label, x.gamma_min, x.p_min, x.policy, x.J))
    rank_profiles.sort(key=lambda x: (x.scope_label, x.gamma_min, x.p_min, x.policy, x.J, x.rank))
    rank_group_summaries.sort(key=lambda x: (x.scope_label, x.gamma_min, x.p_min, x.policy, x.J, x.group_name))

    return cell_summaries, rank_profiles, rank_group_summaries


# ============================================================
# 6) Export
# ============================================================

def export_policy_cell_summaries(
    cell_summaries: Sequence[PolicyCellSummary],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / "policy_comparison_global_indices.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy", "scope", "gamma_min", "p_min", "J", "n_rep",
            "U_feed_mean", "V_P4_mean", "V_P1_mean", "V_P5_mean"
        ])
        for s in sorted(cell_summaries, key=lambda x: (x.scope_label, x.gamma_min, x.p_min, x.policy, x.J)):
            writer.writerow([
                s.policy, s.scope_label, s.gamma_min, s.p_min, s.J, s.n_rep,
                s.U_feed_mean, s.V_P4_mean, s.V_P1_mean, s.V_P5_mean
            ])


def export_rank_profiles(
    rank_profiles: Sequence[RankProfileRecord],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / "policy_comparison_rank_profiles.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy", "scope", "gamma_min", "p_min", "J",
            "rank", "relative_rank", "coverage_mean", "queue_fraction_mean"
        ])
        for r in sorted(rank_profiles, key=lambda x: (x.scope_label, x.gamma_min, x.p_min, x.policy, x.J, x.rank)):
            writer.writerow([
                r.policy, r.scope_label, r.gamma_min, r.p_min, r.J,
                r.rank, r.relative_rank, r.coverage_mean, r.queue_fraction_mean
            ])


def export_rank_group_summaries(
    rank_group_summaries: Sequence[RankGroupSummary],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / "policy_comparison_rank_groups.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy", "scope", "gamma_min", "p_min", "J",
            "group_name", "coverage_mean", "queue_fraction_mean"
        ])
        for r in sorted(rank_group_summaries, key=lambda x: (x.scope_label, x.gamma_min, x.p_min, x.policy, x.J, x.group_name)):
            writer.writerow([
                r.policy, r.scope_label, r.gamma_min, r.p_min, r.J,
                r.group_name, r.coverage_mean, r.queue_fraction_mean
            ])


def export_all_policy_outputs(
    cell_summaries: Sequence[PolicyCellSummary],
    rank_profiles: Sequence[RankProfileRecord],
    rank_group_summaries: Sequence[RankGroupSummary],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    export_policy_cell_summaries(cell_summaries, outdir)
    export_rank_profiles(rank_profiles, outdir)
    export_rank_group_summaries(rank_group_summaries, outdir)


# ============================================================
# 7) Main entry point
# ============================================================

if __name__ == "__main__":
    cell_summaries, rank_profiles, rank_group_summaries = run_policy_comparison_design(
        J_values=[10, 50, 100, 150],
        param_pairs=[
            (0.25, 0.55),  # harder regime
            (0.75, 0.85),  # favorable regime
        ],
        scopes={
            "local_s1": 1,
            "global": None,
        },
        policies=["A", "free"],
        n_rep=20,
        base_seed=20260412,
        K=6,
        D_total=400,
        burn_in_days=100,
        max_workers=2,
    )

    export_all_policy_outputs(
        cell_summaries=cell_summaries,
        rank_profiles=rank_profiles,
        rank_group_summaries=rank_group_summaries,
        outdir="simulation_outputs_policy_comparison",
    )

    print(
        "\nSimulation II finished. Outputs written to "
        "./simulation_outputs_policy_comparison/"
    )