from __future__ import annotations

# ============================================================
# Simulation I postprocessing:
#   - reads raw convergence outputs
#   - recomputes/checks cell summaries
#   - fits descriptive growth models
#   - exports processed CSVs, LaTeX tables, and paper figures
# ============================================================

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


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


def bootstrap_quantile_ci(
    data: np.ndarray,
    q: float = 0.95,
    n_boot: int = 1000,
    alpha: float = 0.10,
    seed: int = 12345,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
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
# 2) Data containers
# ============================================================

@dataclass
class ConvergenceCellStats:
    scope: str
    gamma_min: float
    p_min: float
    J: int
    n_rep: int
    censored: int
    mean: float
    q50: float
    q90: float
    q95: float
    ci_low: float
    ci_high: float
    rel_half_width: float


@dataclass
class FitResult:
    model: str
    params: Tuple[float, ...]
    fitted: np.ndarray
    cv_mse: float


# ============================================================
# 3) Read raw times and aggregate by cell
# ============================================================

def load_raw_times(
    raw_times_path: str | Path,
) -> dict[tuple[str, float, float, int], np.ndarray]:
    rows = read_csv_rows(raw_times_path)

    grouped: dict[tuple[str, float, float, int], list[int]] = {}
    for row in rows:
        key = (
            row["scope"],
            to_float(row["gamma_min"]),
            to_float(row["p_min"]),
            to_int(row["J"]),
        )
        grouped.setdefault(key, []).append(to_int(row["time"]))

    return {k: np.array(v, dtype=int) for k, v in grouped.items()}


def compute_cell_stats(
    grouped_times: dict[tuple[str, float, float, int], np.ndarray],
    max_sessions: int = 10000,
    q_target: float = 0.95,
    n_boot: int = 1000,
    alpha: float = 0.10,
    base_seed: int = 12345,
) -> List[ConvergenceCellStats]:
    results: List[ConvergenceCellStats] = []

    for idx, (key, times) in enumerate(sorted(grouped_times.items())):
        scope, gamma_min, p_min, J = key

        censored = int(np.sum(times > max_sessions))
        finite_times = times[times <= max_sessions]

        if len(finite_times) == 0:
            mean = np.nan
            q50 = np.nan
            q90 = np.nan
            q95 = np.nan
            ci_low = np.nan
            ci_high = np.nan
            rel_half_width = np.nan
        else:
            mean = float(np.mean(finite_times))
            q50 = float(np.quantile(finite_times, 0.50))
            q90 = float(np.quantile(finite_times, 0.90))
            q95 = float(np.quantile(finite_times, 0.95))

            q_hat, ci_low, ci_high = bootstrap_quantile_ci(
                finite_times,
                q=q_target,
                n_boot=n_boot,
                alpha=alpha,
                seed=base_seed + idx,
            )

            if np.isfinite(q_hat) and q_hat > 0:
                rel_half_width = ((ci_high - ci_low) / 2.0) / q_hat
            else:
                rel_half_width = np.nan

        results.append(
            ConvergenceCellStats(
                scope=scope,
                gamma_min=gamma_min,
                p_min=p_min,
                J=J,
                n_rep=len(times),
                censored=censored,
                mean=mean,
                q50=q50,
                q90=q90,
                q95=q95,
                ci_low=ci_low,
                ci_high=ci_high,
                rel_half_width=rel_half_width,
            )
        )

    return results


# ============================================================
# 4) Cross-check against runner output
# ============================================================

def crosscheck_with_cell_summaries(
    stats: Sequence[ConvergenceCellStats],
    cell_summaries_path: str | Path,
    tol: float = 1e-6,
) -> None:
    rows = read_csv_rows(cell_summaries_path)

    summary_map = {}
    for row in rows:
        key = (
            row["scope"],
            to_float(row["gamma_min"]),
            to_float(row["p_min"]),
            to_int(row["J"]),
        )
        summary_map[key] = {
            "q50": to_float(row["q50"]),
            "q90": to_float(row["q90"]),
            "q95": to_float(row["q95"]),
            "mean": to_float(row["mean"]),
            "n_rep": to_int(row["n_rep"]),
            "censored": to_int(row["censored"]),
        }

    mismatches = []
    for s in stats:
        key = (s.scope, s.gamma_min, s.p_min, s.J)
        if key not in summary_map:
            mismatches.append((key, "missing in convergence_cell_summaries.csv"))
            continue

        ref = summary_map[key]
        if (
            abs(s.q50 - ref["q50"]) > tol
            or abs(s.q90 - ref["q90"]) > tol
            or abs(s.q95 - ref["q95"]) > tol
            or abs(s.mean - ref["mean"]) > tol
            or s.n_rep != ref["n_rep"]
            or s.censored != ref["censored"]
        ):
            mismatches.append((key, ref, s))

    if mismatches:
        print("\nCross-check mismatches found:")
        for x in mismatches[:10]:
            print(x)
    else:
        print("\nCross-check passed: recomputed summaries match convergence_cell_summaries.csv.")


# ============================================================
# 5) Growth-model fitting
# ============================================================

def model_logj(J, a, b):
    return a + b * np.log(J)


def model_linear(J, a, b):
    return a + b * J


def model_jlogj(J, a, b):
    return a + b * J * np.log(J)


def model_power(J, a, b, alpha):
    return a + b * (J ** alpha)


def fit_logj_model(J: np.ndarray, y: np.ndarray) -> FitResult:
    z = np.log(J)
    X = np.column_stack([np.ones_like(J), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta

    errs = []
    for k in range(len(J)):
        mask = np.ones(len(J), dtype=bool)
        mask[k] = False
        ztr = np.log(J[mask])
        Xtr = np.column_stack([np.ones(np.sum(mask)), ztr])
        beta_k, *_ = np.linalg.lstsq(Xtr, y[mask], rcond=None)
        pred = beta_k[0] + beta_k[1] * np.log(J[k])
        errs.append((y[k] - pred) ** 2)

    return FitResult("logJ", tuple(beta), fitted, float(np.mean(errs)))


def fit_linear_model(J: np.ndarray, y: np.ndarray) -> FitResult:
    X = np.column_stack([np.ones_like(J), J])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta

    errs = []
    for k in range(len(J)):
        mask = np.ones(len(J), dtype=bool)
        mask[k] = False
        Xtr = np.column_stack([np.ones(np.sum(mask)), J[mask]])
        beta_k, *_ = np.linalg.lstsq(Xtr, y[mask], rcond=None)
        pred = beta_k[0] + beta_k[1] * J[k]
        errs.append((y[k] - pred) ** 2)

    return FitResult("linear", tuple(beta), fitted, float(np.mean(errs)))


def fit_jlogj_model(J: np.ndarray, y: np.ndarray) -> FitResult:
    z = J * np.log(J)
    X = np.column_stack([np.ones_like(J), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta

    errs = []
    for k in range(len(J)):
        mask = np.ones(len(J), dtype=bool)
        mask[k] = False
        ztr = J[mask] * np.log(J[mask])
        Xtr = np.column_stack([np.ones(np.sum(mask)), ztr])
        beta_k, *_ = np.linalg.lstsq(Xtr, y[mask], rcond=None)
        pred = beta_k[0] + beta_k[1] * J[k] * np.log(J[k])
        errs.append((y[k] - pred) ** 2)

    return FitResult("JlogJ", tuple(beta), fitted, float(np.mean(errs)))


def fit_power_model(J: np.ndarray, y: np.ndarray) -> FitResult:
    if len(J) < 4:
        return FitResult("power", tuple(), np.full_like(y, np.nan, dtype=float), np.inf)

    if SCIPY_AVAILABLE:
        mask = y > 0
        alpha0 = 1.0
        if np.sum(mask) >= 2:
            coeffs = np.polyfit(np.log(J[mask]), np.log(y[mask]), 1)
            alpha0 = max(0.1, coeffs[0])

        p0 = (float(np.min(y)), float(np.max(y) - np.min(y)), alpha0)

        try:
            params, _ = curve_fit(model_power, J, y, p0=p0, maxfev=20000)
            fitted = model_power(J, *params)
        except Exception:
            return FitResult("power", tuple(), np.full_like(y, np.nan, dtype=float), np.inf)

        errs = []
        for k in range(len(J)):
            mask_k = np.ones(len(J), dtype=bool)
            mask_k[k] = False
            if np.sum(mask_k) < 3:
                return FitResult("power", tuple(params), fitted, np.inf)
            try:
                params_k, _ = curve_fit(
                    model_power,
                    J[mask_k],
                    y[mask_k],
                    p0=params,
                    maxfev=20000,
                )
                pred = model_power(np.array([J[k]]), *params_k)[0]
                errs.append((y[k] - pred) ** 2)
            except Exception:
                errs.append(np.inf)

        return FitResult("power", tuple(params), fitted, float(np.mean(errs)))

    best = None
    alphas = np.linspace(0.5, 2.5, 81)
    for alpha in alphas:
        z = J ** alpha
        X = np.column_stack([np.ones_like(J), z])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ beta
        sse = float(np.sum((y - fitted) ** 2))
        if best is None or sse < best[0]:
            best = (sse, beta[0], beta[1], alpha, fitted)

    _, a, b, alpha, fitted = best
    return FitResult("power", (a, b, alpha), fitted, np.inf)


def fit_regime(
    stats: Sequence[ConvergenceCellStats],
    scope: str,
    gamma_min: float,
    p_min: float,
) -> List[FitResult]:
    subset = [s for s in stats if s.scope == scope and s.gamma_min == gamma_min and s.p_min == p_min]
    subset = sorted(subset, key=lambda s: s.J)

    J = np.array([s.J for s in subset], dtype=float)
    y = np.array([s.q95 for s in subset], dtype=float)

    mask = np.isfinite(y)
    J = J[mask]
    y = y[mask]

    fits = [
        fit_logj_model(J, y),
        fit_linear_model(J, y),
        fit_jlogj_model(J, y),
    ]
    if len(J) >= 4:
        fits.append(fit_power_model(J, y))

    fits.sort(key=lambda f: f.cv_mse)
    return fits


# ============================================================
# 6) CSV exports
# ============================================================

def export_processed_summary_csv(
    stats: Sequence[ConvergenceCellStats],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / "convergence_processed_summary.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scope", "gamma_min", "p_min", "J", "n_rep", "censored",
            "mean", "q50", "q90", "q95", "ci_low", "ci_high", "rel_half_width"
        ])
        for s in sorted(stats, key=lambda x: (x.scope, x.gamma_min, x.p_min, x.J)):
            writer.writerow([
                s.scope, s.gamma_min, s.p_min, s.J, s.n_rep, s.censored,
                s.mean, s.q50, s.q90, s.q95, s.ci_low, s.ci_high, s.rel_half_width
            ])


def export_fit_summary_csv(
    stats: Sequence[ConvergenceCellStats],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / "convergence_fit_summary.csv"

    keys = sorted({
        (s.scope, s.gamma_min, s.p_min)
        for s in stats
    })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scope", "gamma_min", "p_min", "rank", "model", "cv_mse", "params"
        ])
        for scope, gamma_min, p_min in keys:
            fits = fit_regime(stats, scope, gamma_min, p_min)
            for rank, fit in enumerate(fits, start=1):
                writer.writerow([
                    scope, gamma_min, p_min, rank, fit.model, fit.cv_mse, repr(fit.params)
                ])


# ============================================================
# 7) LaTeX table exports
# ============================================================

def export_latex_selected_q95_table(
    stats: Sequence[ConvergenceCellStats],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / "table_selected_q95.tex"

    selected_J = [10, 90, 150]
    regimes = sorted({
        (s.scope, s.gamma_min, s.p_min)
        for s in stats
    })

    stat_map = {(s.scope, s.gamma_min, s.p_min, s.J): s for s in stats}

    lines = []
    lines.append("\\begin{tabular}{lllrrr}")
    lines.append("\\toprule")
    lines.append("Scope & $\\gamma_{\\min}$ & $p_{\\min}$ & $q_{0.95}(10)$ & $q_{0.95}(90)$ & $q_{0.95}(150)$ \\\\")
    lines.append("\\midrule")

    for scope, gamma_min, p_min in regimes:
        vals = []
        for J in selected_J:
            s = stat_map[(scope, gamma_min, p_min, J)]
            vals.append(f"{s.q95:.2f}")
        lines.append(
            f"{format_scope_short(scope)} & {gamma_min:.2f} & {p_min:.2f} & "
            f"{vals[0]} & {vals[1]} & {vals[2]} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_latex_fit_comparison_table_by_scope(
    stats: Sequence[ConvergenceCellStats],
    outdir: str | Path,
    scope_filter: str,
    filename: str,
) -> None:
    outdir = ensure_dir(outdir)
    path = outdir / filename

    regimes = sorted({
        (s.scope, s.gamma_min, s.p_min)
        for s in stats
        if s.scope == scope_filter
    })

    lines = []
    lines.append("\\begin{tabular}{lllll}")
    lines.append("\\toprule")
    lines.append("$\\gamma_{\\min}$ & $p_{\\min}$ & Best & Second & $\\hat{\\alpha}$ \\\\")
    lines.append("\\midrule")

    for scope, gamma_min, p_min in regimes:
        fits = fit_regime(stats, scope, gamma_min, p_min)
        best = fits[0]
        second = fits[1] if len(fits) > 1 else None

        alpha_text = "--"
        if best.model == "power" and len(best.params) == 3:
            alpha_text = f"{best.params[2]:.3f}"
        elif second is not None and second.model == "power" and len(second.params) == 3:
            alpha_text = f"({second.params[2]:.3f})"

        second_name = second.model if second is not None else "--"

        lines.append(
            f"{gamma_min:.2f} & {p_min:.2f} & "
            f"{best.model} ({best.cv_mse:.1f}) & "
            f"{second_name} ({second.cv_mse:.1f}) & "
            f"{alpha_text} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 8) Publication figures
# ============================================================

def get_series(
    stats: Sequence[ConvergenceCellStats],
    scope: str,
    gamma_min: float,
    p_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    subset = [s for s in stats if s.scope == scope and s.gamma_min == gamma_min and s.p_min == p_min]
    subset = sorted(subset, key=lambda s: s.J)
    J = np.array([s.J for s in subset], dtype=float)
    y = np.array([s.q95 for s in subset], dtype=float)
    return J, y


def plot_scope_comparison_by_reactivity(
    stats: Sequence[ConvergenceCellStats],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)

    for gamma_min in [0.25, 0.75]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

        for ax, p_min in zip(axes, [0.55, 0.85]):
            J_loc, y_loc = get_series(stats, "local_s1", gamma_min, p_min)
            J_glb, y_glb = get_series(stats, "global", gamma_min, p_min)

            ax.plot(J_loc, y_loc, marker="o", label=r"Local scope ($s=1$)")
            ax.plot(J_glb, y_glb, marker="s", label=r"Global scope")
            ax.set_title(rf"$\gamma_{{\min}}={gamma_min:.2f},\ p_{{\min}}={p_min:.2f}$")
            ax.set_xlabel("J")
            ax.set_ylabel(r"$q_{0.95}$")
            ax.grid(True, alpha=0.3)

        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"scope_comparison_gamma_{gamma_min:.2f}.png", dpi=300)
        fig.savefig(outdir / f"scope_comparison_gamma_{gamma_min:.2f}.pdf")
        plt.close(fig)


def plot_parameter_and_ratio_combined(
    stats: Sequence[ConvergenceCellStats],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1, 1])

    ax_local = fig.add_subplot(gs[0, 0])
    ax_global = fig.add_subplot(gs[1, 0])
    ax_ratio = fig.add_subplot(gs[:, 1])

    regimes = [
        (0.25, 0.55, "o"),
        (0.25, 0.85, "s"),
        (0.75, 0.55, "^"),
        (0.75, 0.85, "D"),
    ]

    for gamma_min, p_min, marker in regimes:
        J, y = get_series(stats, "local_s1", gamma_min, p_min)
        ax_local.plot(
            J, y,
            marker=marker,
            label=rf"$\gamma_{{\min}}={gamma_min:.2f},\ p_{{\min}}={p_min:.2f}$"
        )

    ax_local.set_title(r"Local scope ($s=1$)")
    ax_local.set_xlabel("J")
    ax_local.set_ylabel(r"$q_{0.95}$")
    ax_local.grid(True, alpha=0.3)
    ax_local.legend(frameon=False, fontsize=8)

    for gamma_min, p_min, marker in regimes:
        J, y = get_series(stats, "global", gamma_min, p_min)
        ax_global.plot(
            J, y,
            marker=marker,
            label=rf"$\gamma_{{\min}}={gamma_min:.2f},\ p_{{\min}}={p_min:.2f}$"
        )

    ax_global.set_title(r"Global scope")
    ax_global.set_xlabel("J")
    ax_global.set_ylabel(r"$q_{0.95}$")
    ax_global.grid(True, alpha=0.3)

    for gamma_min, p_min, marker in regimes:
        J_loc, y_loc = get_series(stats, "local_s1", gamma_min, p_min)
        J_glb, y_glb = get_series(stats, "global", gamma_min, p_min)

        if not np.array_equal(J_loc, J_glb):
            raise ValueError("Local and global J grids do not match.")

        ratio = y_loc / y_glb
        ax_ratio.plot(
            J_loc, ratio,
            marker=marker,
            label=rf"$\gamma_{{\min}}={gamma_min:.2f},\ p_{{\min}}={p_min:.2f}$"
        )

    ax_ratio.set_title(r"Relative scope advantage")
    ax_ratio.set_xlabel("J")
    ax_ratio.set_ylabel(r"$q_{0.95}^{\mathrm{local}} / q_{0.95}^{\mathrm{global}}$")
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "parameter_and_ratio_combined.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outdir / "parameter_and_ratio_combined.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ============================================================
# 9) Master export
# ============================================================

def export_all_postprocessed_outputs(
    stats: Sequence[ConvergenceCellStats],
    outdir: str | Path,
) -> None:
    outdir = ensure_dir(outdir)

    export_processed_summary_csv(stats, outdir)
    export_fit_summary_csv(stats, outdir)

    table_dir = ensure_dir(outdir / "tables")
    export_latex_selected_q95_table(stats, table_dir)
    export_latex_fit_comparison_table_by_scope(
        stats,
        table_dir,
        scope_filter="local_s1",
        filename="table_fit_comparison_local.tex",
    )
    export_latex_fit_comparison_table_by_scope(
        stats,
        table_dir,
        scope_filter="global",
        filename="table_fit_comparison_global.tex",
    )

    fig_dir = ensure_dir(outdir / "figures")
    plot_scope_comparison_by_reactivity(stats, fig_dir)
    plot_parameter_and_ratio_combined(stats, fig_dir)


# ============================================================
# 10) Main entry point
# ============================================================

if __name__ == "__main__":
    RAW_TIMES = "simulation_outputs_convergence/convergence_raw_times.csv"
    CELL_SUMMARIES = "simulation_outputs_convergence/convergence_cell_summaries.csv"
    OUTDIR = "simulation_outputs_convergence_postprocessed"

    grouped = load_raw_times(RAW_TIMES)

    stats = compute_cell_stats(
        grouped,
        max_sessions=10000,
        q_target=0.95,
        n_boot=1000,
        alpha=0.10,
        base_seed=20260412,
    )

    crosscheck_with_cell_summaries(stats, CELL_SUMMARIES)
    export_all_postprocessed_outputs(stats, OUTDIR)

    print(
        f"\nSimulation I postprocessing finished. Outputs written to ./{OUTDIR}/"
    )