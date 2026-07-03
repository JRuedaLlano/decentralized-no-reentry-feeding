from __future__ import annotations

# ============================================================
# Simulation II runner (fast / multicore)
#   - faster event logic while preserving model semantics
#   - automatic use of multiple CPU cores by default
#   - exports the same output structure expected by postprocessing
# ============================================================

import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# 1) Data containers
# ============================================================

@dataclass(slots=True)
class PolicyCellSummary:
    policy: str
    scope_label: str
    gamma_min: float
    p_min: float
    J: int
    n_rep: int
    U_feed_mean: float
    U_feed_sd: float
    U_feed_ci_low: float
    U_feed_ci_high: float
    V_P4_mean: float
    V_P4_sd: float
    V_P4_ci_low: float
    V_P4_ci_high: float
    V_P1_mean: float
    V_P1_sd: float
    V_P1_ci_low: float
    V_P1_ci_high: float
    V_P5_mean: float
    V_P5_sd: float
    V_P5_ci_low: float
    V_P5_ci_high: float


@dataclass(slots=True)
class RankProfileRecord:
    policy: str
    scope_label: str
    gamma_min: float
    p_min: float
    J: int
    rank: int
    relative_rank: float
    coverage_mean: float
    coverage_sd: float
    coverage_ci_low: float
    coverage_ci_high: float
    queue_fraction_mean: float
    queue_fraction_sd: float
    queue_fraction_ci_low: float
    queue_fraction_ci_high: float


@dataclass(slots=True)
class RankGroupSummary:
    policy: str
    scope_label: str
    gamma_min: float
    p_min: float
    J: int
    group_name: str
    coverage_mean: float
    coverage_sd: float
    coverage_ci_low: float
    coverage_ci_high: float
    queue_fraction_mean: float
    queue_fraction_sd: float
    queue_fraction_ci_low: float
    queue_fraction_ci_high: float


# ============================================================
# 2) Utilities
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _quantile(values: np.ndarray, q: float, axis: int):
    """
    Compatibility wrapper for numpy quantile across versions.
    """
    try:
        return np.quantile(values, q, axis=axis, method="linear")
    except TypeError:
        return np.quantile(values, q, axis=axis, interpolation="linear")


def bootstrap_resample_counts(
    n_rep: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw bootstrap resample counts for n_boot bootstrap samples of size n_rep.
    Each row contains multinomial counts indicating how often each original
    replication appears in the corresponding bootstrap resample.
    """
    if n_rep <= 0:
        raise ValueError("n_rep must be positive.")
    probs = np.full(n_rep, 1.0 / n_rep, dtype=float)
    return rng.multinomial(n_rep, probs, size=n_boot)


def summarize_mean_sd_bootstrap_ci(
    values: np.ndarray,
    counts: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Summarize replication-level outcomes using:
      - sample mean
      - sample standard deviation
      - percentile bootstrap CI for the mean

    Parameters
    ----------
    values
        Shape (n_rep,) or (n_rep, n_features).
    counts
        Bootstrap resample counts with shape (n_boot, n_rep).
    alpha
        Two-sided CI level parameter, e.g. alpha=0.05 for 95% CI.
    """
    arr = np.asarray(values, dtype=float)
    one_dimensional = (arr.ndim == 1)
    if one_dimensional:
        arr = arr[:, None]

    n_rep, n_features = arr.shape
    if counts.shape[1] != n_rep:
        raise ValueError(
            f"counts has incompatible shape {counts.shape}; expected second dimension {n_rep}."
        )

    means = np.mean(arr, axis=0)

    if n_rep <= 1:
        sds = np.full(n_features, np.nan, dtype=float)
        ci_low = np.full(n_features, np.nan, dtype=float)
        ci_high = np.full(n_features, np.nan, dtype=float)
    else:
        sds = np.std(arr, axis=0, ddof=1)
        boot_means = (counts @ arr) / n_rep
        ci_low = _quantile(boot_means, alpha / 2.0, axis=0)
        ci_high = _quantile(boot_means, 1.0 - alpha / 2.0, axis=0)

    if one_dimensional:
        return (
            float(means[0]),
            float(sds[0]),
            float(ci_low[0]),
            float(ci_high[0]),
        )

    return means, sds, ci_low, ci_high


def choose_winner_fast(
    willing_idx: np.ndarray,
    p_min: float,
    rng: np.random.Generator,
) -> int:
    """
    Highest-ranked willing animal (smallest index) gets mass p_min.
    Remaining mass is split equally across the others.

    Equivalent to the previous implementation, but avoids allocating
    a probability vector and calling np.random.choice.
    """
    n = willing_idx.size
    if n == 1:
        return int(willing_idx[0])

    if rng.random() < p_min:
        return int(willing_idx[0])

    other_offset = int(rng.integers(1, n))
    return int(willing_idx[other_offset])


def prepare_scope_state(
    J: int,
    prev_first_entry_order: np.ndarray,
    scope: Optional[int],
):
    """
    Fast scope helpers.

    Returns one of:
      ("global", None)
      ("local_s1", (left_neighbor, right_neighbor))
      ("generic", neighbors)

    where neighbors is a list[list[int]] only for fallback generic scope.
    """
    if scope is None:
        return "global", None

    pos_to_animal = np.empty(J, dtype=np.int32)
    for animal in range(J):
        pos_to_animal[prev_first_entry_order[animal] - 1] = animal

    if scope == 1:
        left_neighbor = np.full(J, -1, dtype=np.int32)
        right_neighbor = np.full(J, -1, dtype=np.int32)

        for pos in range(J):
            animal = pos_to_animal[pos]
            if pos > 0:
                left_neighbor[animal] = pos_to_animal[pos - 1]
            if pos < J - 1:
                right_neighbor[animal] = pos_to_animal[pos + 1]

        return "local_s1", (left_neighbor, right_neighbor)

    neighbors = [[] for _ in range(J)]
    for animal in range(J):
        pos = prev_first_entry_order[animal] - 1
        lo = max(0, pos - scope)
        hi = min(J - 1, pos + scope)
        for q in range(lo, hi + 1):
            other = pos_to_animal[q]
            if other != animal:
                neighbors[animal].append(other)

    return "generic", neighbors


def initialize_session0(
    J: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Session 0:
      - random first-entry order
      - each animal receives J units once
    """
    order = rng.permutation(J) + 1
    satiation = order - 1
    elapsed = J - order + 1
    return order.astype(np.int32), satiation.astype(np.int32), elapsed.astype(np.int32)


def compute_rank_groups(J: int) -> Dict[str, np.ndarray]:
    ranks = np.arange(1, J + 1, dtype=np.int32)

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

    Main speed-ups relative to the original version:
      - equivalent aggregated trigger probabilities instead of looping over
        each possible hungry lower-ranked neighbor pair
      - fast contest resolution
      - vectorized state updates where helpful
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
    one_minus_gamma = 1.0 - gamma_min

    for day in range(D_total):
        for _session_in_day in range(K):
            scope_mode, scope_state = prepare_scope_state(J, prev_first_entry_order, scope)

            served_this_session = np.zeros(J, dtype=bool)
            first_entry_time = np.full(J, fill_value=-1, dtype=np.int32)

            for t in range(1, J + 1):
                hungry_mask = (satiation == 0)
                if policy == "A":
                    hungry_mask &= ~served_this_session

                willing_mask = hungry_mask.copy()

                eligible_reactive = ~hungry_mask
                if policy == "A":
                    eligible_reactive &= ~served_this_session

                if np.any(eligible_reactive) and np.any(hungry_mask):
                    if scope_mode == "global":
                        suffix_counts = np.cumsum(hungry_mask[::-1], dtype=np.int32)[::-1]
                        lower_hungry_counts = suffix_counts - hungry_mask.astype(np.int32)

                        idx = np.flatnonzero(eligible_reactive)
                        counts = lower_hungry_counts[idx]
                        active = counts > 0
                        if np.any(active):
                            probs = 1.0 - np.power(one_minus_gamma, counts[active])
                            draws = rng.random(active.sum())
                            triggered_idx = idx[active][draws < probs]
                            willing_mask[triggered_idx] = True

                    elif scope_mode == "local_s1":
                        left_neighbor, right_neighbor = scope_state
                        idx = np.flatnonzero(eligible_reactive)
                        for j in idx:
                            count = 0

                            left = int(left_neighbor[j])
                            if left >= 0 and left > j and hungry_mask[left]:
                                count += 1

                            right = int(right_neighbor[j])
                            if right >= 0 and right > j and hungry_mask[right]:
                                count += 1

                            if count > 0:
                                trigger_prob = 1.0 - (one_minus_gamma ** count)
                                if rng.random() < trigger_prob:
                                    willing_mask[j] = True

                    else:
                        neighbors = scope_state
                        idx = np.flatnonzero(eligible_reactive)
                        for j in idx:
                            count = 0
                            for k in neighbors[j]:
                                if k > j and hungry_mask[k]:
                                    count += 1
                            if count > 0:
                                trigger_prob = 1.0 - (one_minus_gamma ** count)
                                if rng.random() < trigger_prob:
                                    willing_mask[j] = True

                willing_idx = np.flatnonzero(willing_mask)

                if willing_idx.size > 0:
                    winner = choose_winner_fast(willing_idx, p_min, rng)
                    ration = min(J, int(elapsed_since_last_entry[winner]))

                    if first_entry_time[winner] == -1:
                        first_entry_time[winner] = t

                    if policy == "A":
                        served_this_session[winner] = True

                    if day >= burn_in_days:
                        ration_accum[winner] += ration
                        total_feed_dispensed += ration

                        if willing_idx.size > 1:
                            non_winners = willing_idx[willing_idx != winner]
                            queue_periods[non_winners] += 1.0

                    satiation[winner] = ration
                    elapsed_since_last_entry[winner] = 0

                positive_satiety = satiation > 0
                satiation[positive_satiety] -= 1
                elapsed_since_last_entry += 1

            entered = np.flatnonzero(first_entry_time != -1)
            not_entered = np.flatnonzero(first_entry_time == -1)

            if entered.size > 0:
                entered = entered[np.argsort(first_entry_time[entered], kind="stable")]
            if not_entered.size > 0:
                not_entered = not_entered[np.argsort(prev_first_entry_order[not_entered], kind="stable")]

            combined = np.concatenate((entered, not_entered))
            next_order = np.empty(J, dtype=np.int32)
            next_order[combined] = np.arange(1, J + 1, dtype=np.int32)
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
    n_boot: int = 2000,
    ci_alpha: float = 0.05,
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

    coverage_arr = np.vstack(coverage_stack)   # shape (n_rep, J)
    queue_arr = np.vstack(queue_stack)         # shape (n_rep, J)

    # One reproducible bootstrap design per cell, reused across all summaries
    bootstrap_rng = np.random.default_rng(base_seed + 9_973_733)
    counts = bootstrap_resample_counts(
        n_rep=n_rep,
        n_boot=n_boot,
        rng=bootstrap_rng,
    )

    # Global outcomes
    global_matrix = np.column_stack([
        np.array([g["U_feed"] for g in global_records], dtype=float),
        np.array([g["V_P4"] for g in global_records], dtype=float),
        np.array([g["V_P1"] for g in global_records], dtype=float),
        np.array([g["V_P5"] for g in global_records], dtype=float),
    ])

    global_means, global_sds, global_ci_low, global_ci_high = summarize_mean_sd_bootstrap_ci(
        global_matrix,
        counts=counts,
        alpha=ci_alpha,
    )

    cell_summary = PolicyCellSummary(
        policy=policy,
        scope_label=scope_label,
        gamma_min=gamma_min,
        p_min=p_min,
        J=J,
        n_rep=n_rep,
        U_feed_mean=float(global_means[0]),
        U_feed_sd=float(global_sds[0]),
        U_feed_ci_low=float(global_ci_low[0]),
        U_feed_ci_high=float(global_ci_high[0]),
        V_P4_mean=float(global_means[1]),
        V_P4_sd=float(global_sds[1]),
        V_P4_ci_low=float(global_ci_low[1]),
        V_P4_ci_high=float(global_ci_high[1]),
        V_P1_mean=float(global_means[2]),
        V_P1_sd=float(global_sds[2]),
        V_P1_ci_low=float(global_ci_low[2]),
        V_P1_ci_high=float(global_ci_high[2]),
        V_P5_mean=float(global_means[3]),
        V_P5_sd=float(global_sds[3]),
        V_P5_ci_low=float(global_ci_low[3]),
        V_P5_ci_high=float(global_ci_high[3]),
    )

    # Rank-profile summaries
    coverage_means, coverage_sds, coverage_ci_low, coverage_ci_high = summarize_mean_sd_bootstrap_ci(
        coverage_arr,
        counts=counts,
        alpha=ci_alpha,
    )
    queue_means, queue_sds, queue_ci_low, queue_ci_high = summarize_mean_sd_bootstrap_ci(
        queue_arr,
        counts=counts,
        alpha=ci_alpha,
    )

    rank_profiles: List[RankProfileRecord] = []
    for rank in range(1, J + 1):
        idx = rank - 1
        rank_profiles.append(
            RankProfileRecord(
                policy=policy,
                scope_label=scope_label,
                gamma_min=gamma_min,
                p_min=p_min,
                J=J,
                rank=rank,
                relative_rank=(rank - 1) / (J - 1) if J > 1 else 0.0,
                coverage_mean=float(coverage_means[idx]),
                coverage_sd=float(coverage_sds[idx]),
                coverage_ci_low=float(coverage_ci_low[idx]),
                coverage_ci_high=float(coverage_ci_high[idx]),
                queue_fraction_mean=float(queue_means[idx]),
                queue_fraction_sd=float(queue_sds[idx]),
                queue_fraction_ci_low=float(queue_ci_low[idx]),
                queue_fraction_ci_high=float(queue_ci_high[idx]),
            )
        )

    # Rank-group summaries
    group_map = compute_rank_groups(J)
    group_names = list(group_map.keys())

    coverage_group_matrix = np.column_stack([
        np.mean(coverage_arr[:, group_map[group_name] - 1], axis=1)
        for group_name in group_names
    ])
    queue_group_matrix = np.column_stack([
        np.mean(queue_arr[:, group_map[group_name] - 1], axis=1)
        for group_name in group_names
    ])

    coverage_group_means, coverage_group_sds, coverage_group_ci_low, coverage_group_ci_high = summarize_mean_sd_bootstrap_ci(
        coverage_group_matrix,
        counts=counts,
        alpha=ci_alpha,
    )
    queue_group_means, queue_group_sds, queue_group_ci_low, queue_group_ci_high = summarize_mean_sd_bootstrap_ci(
        queue_group_matrix,
        counts=counts,
        alpha=ci_alpha,
    )

    rank_group_summaries: List[RankGroupSummary] = []
    for group_idx, group_name in enumerate(group_names):
        rank_group_summaries.append(
            RankGroupSummary(
                policy=policy,
                scope_label=scope_label,
                gamma_min=gamma_min,
                p_min=p_min,
                J=J,
                group_name=group_name,
                coverage_mean=float(coverage_group_means[group_idx]),
                coverage_sd=float(coverage_group_sds[group_idx]),
                coverage_ci_low=float(coverage_group_ci_low[group_idx]),
                coverage_ci_high=float(coverage_group_ci_high[group_idx]),
                queue_fraction_mean=float(queue_group_means[group_idx]),
                queue_fraction_sd=float(queue_group_sds[group_idx]),
                queue_fraction_ci_low=float(queue_group_ci_low[group_idx]),
                queue_fraction_ci_high=float(queue_group_ci_high[group_idx]),
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
    max_workers: Optional[int] = None,
    n_boot: int = 2000,
    ci_alpha: float = 0.05,
) -> Tuple[List[PolicyCellSummary], List[RankProfileRecord], List[RankGroupSummary]]:
    jobs = []
    counter = 0

    for J, (gamma_min, p_min), (scope_label, scope), policy in itertools.product(
        J_values, param_pairs, scopes.items(), policies
    ):
        seed = base_seed + 10000 * counter
        counter += 1
        jobs.append((
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
            n_boot,
            ci_alpha,
        ))

    total_jobs = len(jobs)
    if max_workers is None:
        detected = os.cpu_count() or 1
        max_workers = min(total_jobs, detected)
    else:
        max_workers = max(1, min(total_jobs, int(max_workers)))

    print(f"Running {total_jobs} simulation cells with max_workers={max_workers}")

    cell_summaries: List[PolicyCellSummary] = []
    rank_profiles: List[RankProfileRecord] = []
    rank_group_summaries: List[RankGroupSummary] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(run_policy_comparison_cell, *job)
            for job in jobs
        ]

        for idx, fut in enumerate(as_completed(futures), start=1):
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
            "U_feed_mean", "U_feed_sd", "U_feed_ci_low", "U_feed_ci_high",
            "V_P4_mean", "V_P4_sd", "V_P4_ci_low", "V_P4_ci_high",
            "V_P1_mean", "V_P1_sd", "V_P1_ci_low", "V_P1_ci_high",
            "V_P5_mean", "V_P5_sd", "V_P5_ci_low", "V_P5_ci_high",
        ])
        for s in sorted(cell_summaries, key=lambda x: (x.scope_label, x.gamma_min, x.p_min, x.policy, x.J)):
            writer.writerow([
                s.policy, s.scope_label, s.gamma_min, s.p_min, s.J, s.n_rep,
                s.U_feed_mean, s.U_feed_sd, s.U_feed_ci_low, s.U_feed_ci_high,
                s.V_P4_mean, s.V_P4_sd, s.V_P4_ci_low, s.V_P4_ci_high,
                s.V_P1_mean, s.V_P1_sd, s.V_P1_ci_low, s.V_P1_ci_high,
                s.V_P5_mean, s.V_P5_sd, s.V_P5_ci_low, s.V_P5_ci_high,
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
            "rank", "relative_rank",
            "coverage_mean", "coverage_sd", "coverage_ci_low", "coverage_ci_high",
            "queue_fraction_mean", "queue_fraction_sd", "queue_fraction_ci_low", "queue_fraction_ci_high",
        ])
        for r in sorted(rank_profiles, key=lambda x: (x.scope_label, x.gamma_min, x.p_min, x.policy, x.J, x.rank)):
            writer.writerow([
                r.policy, r.scope_label, r.gamma_min, r.p_min, r.J,
                r.rank, r.relative_rank,
                r.coverage_mean, r.coverage_sd, r.coverage_ci_low, r.coverage_ci_high,
                r.queue_fraction_mean, r.queue_fraction_sd, r.queue_fraction_ci_low, r.queue_fraction_ci_high,
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
            "group_name",
            "coverage_mean", "coverage_sd", "coverage_ci_low", "coverage_ci_high",
            "queue_fraction_mean", "queue_fraction_sd", "queue_fraction_ci_low", "queue_fraction_ci_high",
        ])
        for r in sorted(rank_group_summaries, key=lambda x: (x.scope_label, x.gamma_min, x.p_min, x.policy, x.J, x.group_name)):
            writer.writerow([
                r.policy, r.scope_label, r.gamma_min, r.p_min, r.J,
                r.group_name,
                r.coverage_mean, r.coverage_sd, r.coverage_ci_low, r.coverage_ci_high,
                r.queue_fraction_mean, r.queue_fraction_sd, r.queue_fraction_ci_low, r.queue_fraction_ci_high,
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
    start = time.perf_counter()

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
        n_rep=100,
        base_seed=20260412,
        K=6,
        D_total=400,
        burn_in_days=100,
        max_workers=6,
        n_boot=2000,
        ci_alpha=0.05,
    )

    export_all_policy_outputs(
        cell_summaries=cell_summaries,
        rank_profiles=rank_profiles,
        rank_group_summaries=rank_group_summaries,
        outdir="simulation_outputs_policy_comparison_v2_fast",
    )

    elapsed = time.perf_counter() - start
    print(
        "\nSimulation II finished. Outputs written to "
        "./simulation_outputs_policy_comparison_v2_fast/"
    )
    print(f"Elapsed time: {elapsed/60:.2f} minutes")
