from __future__ import annotations

# ============================================================
# Simulation I runner:
#   - runs convergence simulations
#   - exports raw replication times
#   - exports cell-level summaries
# No plotting, no model fitting, no LaTeX export.
# ============================================================

import csv
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# 1) Data container
# ============================================================

@dataclass
class ConvergenceCellSummary:
    J: int
    gamma_min: float
    p_min: float
    scope_label: str
    n_rep: int
    times: np.ndarray
    censored: int
    q50: float
    q90: float
    q95: float
    mean: float
    q_target_estimate: float
    ci_low: float
    ci_high: float
    rel_half_width: float
    stopped_by_precision: bool


# ============================================================
# 2) Utilities
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def summary_to_row(summary: ConvergenceCellSummary) -> dict:
    return {
        "scope": summary.scope_label,
        "J": summary.J,
        "gamma_min": summary.gamma_min,
        "p_min": summary.p_min,
        "n_rep": summary.n_rep,
        "censored": summary.censored,
        "q50": summary.q50,
        "q90": summary.q90,
        "q95": summary.q95,
        "mean": summary.mean,
        "q_target_estimate": summary.q_target_estimate,
        "ci_low": summary.ci_low,
        "ci_high": summary.ci_high,
        "rel_half_width": summary.rel_half_width,
        "stopped_by_precision": summary.stopped_by_precision,
    }


def bootstrap_quantile_ci(
    data: np.ndarray,
    q: float = 0.95,
    n_boot: int = 300,
    alpha: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for an empirical quantile.
    Returns (q_hat, ci_low, ci_high).
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data)
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    q_hat = float(np.quantile(data, q))
    n = len(data)
    boot_stats = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[b] = np.quantile(sample, q)

    lo = float(np.quantile(boot_stats, alpha / 2))
    hi = float(np.quantile(boot_stats, 1 - alpha / 2))
    return q_hat, lo, hi


# ============================================================
# 3) Core model
# ============================================================

def compute_first_hunger_times(J: int, p_prev: np.ndarray, r_prev: np.ndarray) -> np.ndarray:
    """
    Session-start first-hunger times:
        u_n(j) = max(1, r_{n-1}(j) - (J - p_{n-1}(j)))
    Arrays are indexed by animal-1, while positions are 1-based.
    """
    u = r_prev - (J - p_prev)
    return np.maximum(u, 1).astype(int)


def build_scope_neighbors_from_prev_order(
    J: int,
    p_prev: np.ndarray,
    scope: Optional[int],
) -> List[List[int]]:
    """
    scope = None  -> global scope
    scope = s     -> local scope with radius s in previous-session positions

    Returns, for each animal j (0-based), the list of animals in scope of j,
    excluding j itself.
    """
    if scope is None:
        return [[k for k in range(J) if k != j] for j in range(J)]

    pos_to_animal = np.empty(J, dtype=int)
    for animal in range(J):
        pos_to_animal[p_prev[animal] - 1] = animal

    neighbors = [[] for _ in range(J)]
    for animal in range(J):
        pos = p_prev[animal] - 1
        lo = max(0, pos - scope)
        hi = min(J - 1, pos + scope)
        for q in range(lo, hi + 1):
            other = pos_to_animal[q]
            if other != animal:
                neighbors[animal].append(other)
    return neighbors


def choose_winner(
    willing: List[int],
    p_min: float,
    rng: np.random.Generator,
) -> int:
    """
    Highest-ranked willing animal (smallest index) gets probability p_min.
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


def simulate_one_session(
    J: int,
    p_prev: np.ndarray,
    r_prev: np.ndarray,
    gamma_min: float,
    p_min: float,
    scope: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Session update under Policy A:
      - hunger-based willingness persists within session
      - social reactivity is ephemeral and recomputed each period
      - pairwise-independent triggering with common gamma_min
      - only hungry lower-ranked animals trigger reactivity
      - once served, an animal cannot re-enter within the session
    Returns:
      u_cur, p_cur, r_cur
    """
    u_cur = compute_first_hunger_times(J, p_prev, r_prev)
    neighbors = build_scope_neighbors_from_prev_order(J, p_prev, scope)

    served = np.zeros(J, dtype=bool)
    p_cur = np.zeros(J, dtype=int)
    r_cur = np.zeros(J, dtype=int)

    for t in range(1, J + 1):
        hungry_willing = [j for j in range(J) if (not served[j]) and (u_cur[j] <= t)]

        if not hungry_willing:
            raise RuntimeError(
                f"No hungry unserved animal at period {t}. "
                "This should not happen under the model assumptions."
            )

        willing = set(hungry_willing)
        hungry_set = set(hungry_willing)

        for j in range(J):
            if served[j] or j in hungry_set:
                continue

            triggered = False
            for k in neighbors[j]:
                if (k in hungry_set) and (k > j):
                    if rng.random() < gamma_min:
                        triggered = True
                        break

            if triggered:
                willing.add(j)

        winner = choose_winner(list(willing), p_min, rng)

        served[winner] = True
        p_cur[winner] = t

        elapsed = J - p_prev[winner] + t
        r_cur[winner] = min(J, elapsed)

    return u_cur, p_cur, r_cur


def is_absorbing(J: int, p_cur: np.ndarray, r_cur: np.ndarray) -> bool:
    """
    Absorbing configuration:
      - identity entry order
      - all animals receive J units
    """
    return np.all(p_cur == np.arange(1, J + 1)) and np.all(r_cur == J)


def simulate_until_convergence(
    J: int,
    gamma_min: float,
    p_min: float,
    scope: Optional[int],
    rng: np.random.Generator,
    max_sessions: int = 10000,
) -> int:
    """
    Initialization:
      - session 0 order is random
      - all animals receive J units in session 0

    Returns:
      number of sessions until first absorbing session,
      or max_sessions + 1 if censored.
    """
    p_prev = rng.permutation(J) + 1
    r_prev = np.full(J, J, dtype=int)

    for session in range(1, max_sessions + 1):
        _, p_cur, r_cur = simulate_one_session(
            J=J,
            p_prev=p_prev,
            r_prev=r_prev,
            gamma_min=gamma_min,
            p_min=p_min,
            scope=scope,
            rng=rng,
        )

        if is_absorbing(J, p_cur, r_cur):
            return session

        p_prev, r_prev = p_cur, r_cur

    return max_sessions + 1


# ============================================================
# 4) Adaptive replication for one design cell
# ============================================================

def run_convergence_cell(
    J: int,
    gamma_min: float,
    p_min: float,
    scope: Optional[int],
    scope_label: str,
    seed: int,
    max_sessions: int = 10000,
    q_target: float = 0.95,
    n_init: int = 100,
    chunk_size: int = 100,
    n_max: int = 1500,
    n_boot: int = 300,
    alpha: float = 0.10,
    rel_tol: float = 0.08,
    min_chunks_before_stop: int = 2,
    min_runs_before_bootstrap: int = 300,
    bootstrap_check_every: int = 200,
) -> ConvergenceCellSummary:
    """
    Adaptive replication for one Simulation I cell.

    Stopping rule:
        relative half-width <= rel_tol
    where
        relative half-width = ((ci_high - ci_low)/2) / q_hat
    """
    rng = np.random.default_rng(seed)
    times_list: List[int] = []

    total_runs = 0
    chunks_done = 0
    stopped_by_precision = False

    q_hat = np.nan
    ci_low = np.nan
    ci_high = np.nan
    rel_half_width = np.nan

    batch = n_init

    while total_runs < n_max:
        batch = min(batch, n_max - total_runs)

        for _ in range(batch):
            t = simulate_until_convergence(
                J=J,
                gamma_min=gamma_min,
                p_min=p_min,
                scope=scope,
                rng=rng,
                max_sessions=max_sessions,
            )
            times_list.append(t)

        total_runs += batch
        chunks_done += 1

        times = np.array(times_list, dtype=int)
        finite_times = times[times <= max_sessions]

        should_check_precision = (
            chunks_done >= min_chunks_before_stop
            and total_runs >= min_runs_before_bootstrap
            and (
                total_runs == min_runs_before_bootstrap
                or (total_runs - min_runs_before_bootstrap) % bootstrap_check_every == 0
                or total_runs == n_max
            )
        )

        if should_check_precision:
            if len(finite_times) > 0:
                q_hat, ci_low, ci_high = bootstrap_quantile_ci(
                    finite_times,
                    q=q_target,
                    n_boot=n_boot,
                    alpha=alpha,
                    rng=rng,
                )

                if np.isfinite(q_hat) and q_hat > 0:
                    rel_half_width = ((ci_high - ci_low) / 2.0) / q_hat
                else:
                    rel_half_width = np.inf
            else:
                q_hat = ci_low = ci_high = np.nan
                rel_half_width = np.inf

            if np.isfinite(rel_half_width) and rel_half_width <= rel_tol:
                stopped_by_precision = True
                break

        batch = chunk_size

    times = np.array(times_list, dtype=int)
    censored = int(np.sum(times > max_sessions))
    finite_times = times[times <= max_sessions]

    if len(finite_times) == 0:
        q50 = q90 = q95 = mean = np.nan
        if not np.isfinite(q_hat):
            q_hat = ci_low = ci_high = rel_half_width = np.nan
    else:
        q50 = float(np.quantile(finite_times, 0.50))
        q90 = float(np.quantile(finite_times, 0.90))
        q95 = float(np.quantile(finite_times, 0.95))
        mean = float(np.mean(finite_times))

        if not np.isfinite(q_hat):
            q_hat, ci_low, ci_high = bootstrap_quantile_ci(
                finite_times,
                q=q_target,
                n_boot=n_boot,
                alpha=alpha,
                rng=rng,
            )
            if np.isfinite(q_hat) and q_hat > 0:
                rel_half_width = ((ci_high - ci_low) / 2.0) / q_hat
            else:
                rel_half_width = np.inf

    return ConvergenceCellSummary(
        J=J,
        gamma_min=gamma_min,
        p_min=p_min,
        scope_label=scope_label,
        n_rep=total_runs,
        times=times,
        censored=censored,
        q50=q50,
        q90=q90,
        q95=q95,
        mean=mean,
        q_target_estimate=q_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        rel_half_width=float(rel_half_width),
        stopped_by_precision=stopped_by_precision,
    )


def run_convergence_design(
    J_values: Sequence[int],
    gamma_values: Sequence[float],
    p_values: Sequence[float],
    scopes: Dict[str, Optional[int]],
    max_sessions: int = 10000,
    max_workers: Optional[int] = 2,
    base_seed: int = 12345,
    q_target: float = 0.95,
    n_init: int = 100,
    chunk_size: int = 100,
    n_max: int = 1500,
    n_boot: int = 300,
    alpha: float = 0.10,
    rel_tol: float = 0.08,
    min_chunks_before_stop: int = 2,
    min_runs_before_bootstrap: int = 300,
    bootstrap_check_every: int = 200,
) -> List[ConvergenceCellSummary]:
    jobs = []
    counter = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for J, gamma_min, p_min, (scope_label, scope) in itertools.product(
            J_values, gamma_values, p_values, scopes.items()
        ):
            seed = base_seed + 100000 * counter
            counter += 1

            fut = ex.submit(
                run_convergence_cell,
                J=J,
                gamma_min=gamma_min,
                p_min=p_min,
                scope=scope,
                scope_label=scope_label,
                seed=seed,
                max_sessions=max_sessions,
                q_target=q_target,
                n_init=n_init,
                chunk_size=chunk_size,
                n_max=n_max,
                n_boot=n_boot,
                alpha=alpha,
                rel_tol=rel_tol,
                min_chunks_before_stop=min_chunks_before_stop,
                min_runs_before_bootstrap=min_runs_before_bootstrap,
                bootstrap_check_every=bootstrap_check_every,
            )
            jobs.append(fut)

        results: List[ConvergenceCellSummary] = []
        total_jobs = len(jobs)

        for idx, fut in enumerate(as_completed(jobs), start=1):
            res = fut.result()
            results.append(res)
            print(
                f"[{idx}/{total_jobs}] done: scope={res.scope_label}, "
                f"J={res.J}, gamma={res.gamma_min}, p={res.p_min}, "
                f"reps={res.n_rep}, q95={res.q95:.2f}"
            )

    results.sort(key=lambda x: (x.scope_label, x.gamma_min, x.p_min, x.J))
    return results


# ============================================================
# 5) Export
# ============================================================

def export_convergence_cell_summaries(
    results: Sequence[ConvergenceCellSummary],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / "convergence_cell_summaries.csv"

    rows = [summary_to_row(r) for r in results]
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_convergence_raw_times(
    results: Sequence[ConvergenceCellSummary],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / "convergence_raw_times.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "J", "gamma_min", "p_min", "replicate", "time"])
        for r in results:
            for idx, t in enumerate(r.times, start=1):
                writer.writerow([r.scope_label, r.J, r.gamma_min, r.p_min, idx, int(t)])


def export_all_convergence_outputs(
    results: Sequence[ConvergenceCellSummary],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    export_convergence_cell_summaries(results, outdir)
    export_convergence_raw_times(results, outdir)


# ============================================================
# 6) Main entry point
# ============================================================

if __name__ == "__main__":
    results = run_convergence_design(
        J_values=[10, 30, 50, 70, 90, 110, 130, 150],
        gamma_values=[0.25, 0.75],
        p_values=[0.55, 0.85],
        scopes={
            "local_s1": 1,
            "global": None,
        },
        max_sessions=10000,
        max_workers=4,
        base_seed=20260412,
        q_target=0.95,
        n_init=100,
        chunk_size=100,
        n_max=400,
        n_boot=50,
        alpha=0.10,
        rel_tol=0.10,
        min_chunks_before_stop=3,
        min_runs_before_bootstrap=300,
        bootstrap_check_every=200,
    )

    export_all_convergence_outputs(
        results=results,
        outdir="simulation_outputs_convergence",
    )

    print(
        "\nSimulation I finished. Outputs written to "
        "./simulation_outputs_convergence/"
    )