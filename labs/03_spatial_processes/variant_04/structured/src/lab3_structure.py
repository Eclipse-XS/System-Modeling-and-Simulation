import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class HeatProblem:
    a: float
    L: float
    T: float
    N: int
    alpha: float
    beta: float
    phi: float = 0.0
    h_user: float = 0.002
    series_terms: int = 30


@dataclass
class HeatSolution:
    y: np.ndarray
    t_grid: np.ndarray
    U_num: np.ndarray
    U_an: np.ndarray
    mid_t: np.ndarray
    mid_u: np.ndarray
    delta: float
    dt_max: float
    h_used: float
    steps_total: int
    mae: float
    mse: float


def build_snapshot_times(T: float) -> np.ndarray:
    parts = [np.array([0.0], dtype=float)]

    if T > 0.0:
        t1_end = min(T, 10.0)
        t1 = np.linspace(0.0, t1_end, 81)
        parts.append(t1[1:])

    if T > 10.0:
        t2_end = min(T, 120.0)
        t2 = np.linspace(10.0, t2_end, 81)
        parts.append(t2[1:])

    if T > 120.0:
        t3 = np.linspace(120.0, T, 31)
        parts.append(t3[1:])

    return np.unique(np.concatenate(parts))


def build_spatial_grid(L: float, N: int) -> tuple[np.ndarray, float]:
    y = np.linspace(0.0, L, N + 2)
    delta = L / (N + 1)
    return y, delta


def compute_time_step(a: float, delta: float, T: float, h_user: float) -> tuple[float, float, int]:
    dt_max = (delta * delta) / (2.0 * a)
    h = min(h_user, 0.9 * dt_max)
    steps_total = int(np.ceil(T / h))
    h = T / steps_total
    return dt_max, h, steps_total


def assemble_profile(u_inner: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    prof = np.empty(u_inner.size + 2, dtype=float)
    prof[0] = alpha
    prof[-1] = beta
    prof[1:-1] = u_inner
    return prof


def heat_rhs(
    u_inner: np.ndarray,
    out: np.ndarray,
    mu: float,
    alpha: float,
    beta: float
) -> None:
    out[1:-1] = mu * (u_inner[2:] - 2.0 * u_inner[1:-1] + u_inner[:-2])
    out[0] = mu * (u_inner[1] - 2.0 * u_inner[0] + alpha)
    out[-1] = mu * (beta - 2.0 * u_inner[-1] + u_inner[-2])


def rk4_step(
    u_inner: np.ndarray,
    h: float,
    mu: float,
    alpha: float,
    beta: float,
    k1: np.ndarray,
    k2: np.ndarray,
    k3: np.ndarray,
    k4: np.ndarray,
    tmp: np.ndarray
) -> None:
    heat_rhs(u_inner, k1, mu, alpha, beta)

    tmp[:] = u_inner + 0.5 * h * k1
    heat_rhs(tmp, k2, mu, alpha, beta)

    tmp[:] = u_inner + 0.5 * h * k2
    heat_rhs(tmp, k3, mu, alpha, beta)

    tmp[:] = u_inner + h * k3
    heat_rhs(tmp, k4, mu, alpha, beta)

    u_inner[:] = u_inner + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def initialize_rk4_buffers(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1 = np.empty(N, dtype=float)
    k2 = np.empty(N, dtype=float)
    k3 = np.empty(N, dtype=float)
    k4 = np.empty(N, dtype=float)
    tmp = np.empty(N, dtype=float)
    return k1, k2, k3, k4, tmp


def prepare_analytic_basis(
    y: np.ndarray,
    a: float,
    L: float,
    alpha: float,
    beta: float,
    terms: int,
    h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.arange(1, terms + 1, dtype=float)
    base = alpha + (beta - alpha) * (y / L)
    coeff = (2.0 / (np.pi * n)) * (((-1.0) ** n) * beta - alpha)
    sin_part = np.sin((np.pi * n[:, None] * y[None, :]) / L)
    weighted_modes = coeff[:, None] * sin_part
    q = np.exp(-a * ((np.pi * n / L) ** 2) * h)
    return base, weighted_modes, q


def analytic_profile_from_decay(
    base: np.ndarray,
    weighted_modes: np.ndarray,
    decay: np.ndarray
) -> np.ndarray:
    return base + decay @ weighted_modes


def simulate_numerical_solution(
    problem: HeatProblem,
    y: np.ndarray,
    delta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int, float, float]:
    mu = problem.a / (delta * delta)
    dt_max, h, steps_total = compute_time_step(problem.a, delta, problem.T, problem.h_user)

    t_requested = build_snapshot_times(problem.T)
    snap_steps = np.rint(t_requested / h).astype(int)
    snap_steps = np.clip(snap_steps, 0, steps_total)
    t_grid = snap_steps * h

    U_num = np.empty((t_grid.size, problem.N + 2), dtype=float)
    U_an = np.empty((t_grid.size, problem.N + 2), dtype=float)

    u = np.full(problem.N, float(problem.phi), dtype=float)

    prof_num = assemble_profile(u, problem.alpha, problem.beta)

    base, weighted_modes, q = prepare_analytic_basis(
        y=y,
        a=problem.a,
        L=problem.L,
        alpha=problem.alpha,
        beta=problem.beta,
        terms=problem.series_terms,
        h=h
    )

    decay = np.ones(problem.series_terms, dtype=float)
    prof_an = analytic_profile_from_decay(base, weighted_modes, decay)

    snap_ptr = 0
    while snap_ptr < t_grid.size and snap_steps[snap_ptr] == 0:
        U_num[snap_ptr, :] = prof_num
        U_an[snap_ptr, :] = prof_an
        snap_ptr += 1

    mid_idx = (problem.N + 2) // 2
    mid_t = np.empty(steps_total + 1, dtype=float)
    mid_u = np.empty(steps_total + 1, dtype=float)
    mid_t[0] = 0.0
    mid_u[0] = prof_num[mid_idx]

    diff0 = prof_num[1:-1] - prof_an[1:-1]
    mae = float(np.max(np.abs(diff0)))
    mse_sum = float(np.sum(diff0 * diff0))
    mse_count = problem.N

    k1, k2, k3, k4, tmp = initialize_rk4_buffers(problem.N)

    print_every = max(1, steps_total // 50)

    for step in range(1, steps_total + 1):
        rk4_step(
            u_inner=u,
            h=h,
            mu=mu,
            alpha=problem.alpha,
            beta=problem.beta,
            k1=k1,
            k2=k2,
            k3=k3,
            k4=k4,
            tmp=tmp
        )

        decay *= q

        prof_num = assemble_profile(u, problem.alpha, problem.beta)
        prof_an = analytic_profile_from_decay(base, weighted_modes, decay)

        diff = prof_num[1:-1] - prof_an[1:-1]
        mae = max(mae, float(np.max(np.abs(diff))))
        mse_sum += float(np.sum(diff * diff))
        mse_count += problem.N

        t = step * h
        mid_t[step] = t
        mid_u[step] = prof_num[mid_idx]

        while snap_ptr < t_grid.size and snap_steps[snap_ptr] == step:
            U_num[snap_ptr, :] = prof_num
            U_an[snap_ptr, :] = prof_an
            snap_ptr += 1

        if step % print_every == 0 or step == steps_total:
            pct = 100.0 * step / steps_total
            print(f"iter {step}/{steps_total} ({pct:.1f}%), t={t:.2f}s, h={h:.6e}s")

    mse = mse_sum / mse_count

    return U_num, U_an, t_grid, mid_t, mid_u, dt_max, h, steps_total, mae, mse


def solve_heat_equation(problem: HeatProblem) -> HeatSolution:
    y, delta = build_spatial_grid(problem.L, problem.N)

    U_num, U_an, t_grid, mid_t, mid_u, dt_max, h_used, steps_total, mae, mse = simulate_numerical_solution(
        problem=problem,
        y=y,
        delta=delta
    )

    return HeatSolution(
        y=y,
        t_grid=t_grid,
        U_num=U_num,
        U_an=U_an,
        mid_t=mid_t,
        mid_u=mid_u,
        delta=delta,
        dt_max=dt_max,
        h_used=h_used,
        steps_total=steps_total,
        mae=mae,
        mse=mse
    )


def print_solution_summary(solution: HeatSolution) -> None:
    print(f"delta = {solution.delta:.6e} m")
    print(f"dt_max = {solution.dt_max:.6e} s")
    print(f"h_used = {solution.h_used:.6e} s")
    print(f"steps_total = {solution.steps_total}")
    print(f"MAE = {solution.mae:.6e}")
    print(f"MSE = {solution.mse:.6e}")


def plot_midplane_vs_time_hours(solution: HeatSolution) -> None:
    plt.figure()
    plt.plot(solution.mid_t / 3600.0, solution.mid_u)
    plt.xlabel("t, h")
    plt.ylabel("Temperature at mid-plane, °C")
    plt.title("Mid-plane temperature vs time (hours)")
    plt.grid(True)
    plt.show()


def plot_midplane_vs_time_seconds(solution: HeatSolution, t_max: float = 120.0) -> None:
    mask = solution.mid_t <= t_max
    plt.figure()
    plt.plot(solution.mid_t[mask], solution.mid_u[mask])
    plt.xlabel("t, s")
    plt.ylabel("Temperature at mid-plane, °C")
    plt.title(f"Mid-plane temperature vs time (0..{int(t_max)} s)")
    plt.grid(True)
    plt.show()


def plot_temperature_snapshots(solution: HeatSolution, snap_sec: list[float] | np.ndarray) -> None:
    plt.figure()
    for ts in snap_sec:
        idx = int(np.argmin(np.abs(solution.t_grid - ts)))
        plt.plot(solution.y, solution.U_num[idx, :], label=f"t={solution.t_grid[idx]:.2f}s")
    plt.xlabel("y, m")
    plt.ylabel("Temperature, °C")
    plt.title("Temperature distribution in the wall (numerical)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_numerical_vs_analytic_3d(solution: HeatSolution, t_min_sec: float = 1.0) -> None:
    mask = solution.t_grid >= t_min_sec
    Y, TT = np.meshgrid(solution.y, solution.t_grid[mask] / 3600.0)

    U_num_3d = solution.U_num[mask, :]
    U_an_3d = solution.U_an[mask, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        Y,
        TT,
        U_an_3d,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.85,
        color="red"
    )

    ax.plot_wireframe(
        Y,
        TT,
        U_num_3d,
        rstride=6,
        cstride=6,
        linewidth=0.8,
        color="black"
    )

    ax.set_xlabel("y, m")
    ax.set_ylabel("t, h")
    ax.set_zlabel("Temperature, °C")
    ax.set_title("Analytical (surface, red) vs Numerical (wireframe, black)")

    plt.show()


def plot_all_diagnostics(solution: HeatSolution) -> None:
    print_solution_summary(solution)
    plot_midplane_vs_time_hours(solution)
    plot_midplane_vs_time_seconds(solution, t_max=120.0)
    plot_temperature_snapshots(solution, snap_sec=[0.0, 0.2, 1.0, 5.0, 20.0])
    plot_numerical_vs_analytic_3d(solution, t_min_sec=1.0)


def main() -> None:
    problem = HeatProblem(
        a=111.0e-6,
        L=0.07,
        T=1.0 * 3600.0,
        N=100,
        alpha=4.0,
        beta=23.0,
        phi=0.0,
        h_user=10,
        series_terms=30
    )

    solution = solve_heat_equation(problem)
    plot_all_diagnostics(solution)


if __name__ == "__main__":
    main()