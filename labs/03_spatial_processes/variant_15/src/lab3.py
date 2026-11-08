import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class HeatConfig:
    material: str = "Повітря"
    a: float = 19.0e-6
    L: float = 1.5
    T_hours: float = 4.5
    N: int = 100
    alpha: float = 15.0
    beta: float = 34.0
    phi0: float = 0.0
    dt_requested: float = 30.0
    series_terms: int = 30


@dataclass
class HeatResult:
    y: np.ndarray
    t_snap: np.ndarray
    U_num: np.ndarray
    U_an: np.ndarray
    t_full: np.ndarray
    center_u: np.ndarray
    dy: float
    dt: float
    dt_limit: float
    steps: int
    mae: float
    mse: float


def spatial_grid(L, N):
    y = np.linspace(0.0, L, N + 2)
    dy = L / (N + 1)
    return y, dy


def corrected_time_grid(T, requested_dt, dt_limit):
    dt = min(requested_dt, 0.85 * dt_limit)
    steps = int(np.ceil(T / dt))
    dt = T / steps
    t = np.linspace(0.0, T, steps + 1)
    return t, dt, steps


def rhs(u, out, coef, alpha, beta):
    out[0] = coef * (u[1] - 2.0 * u[0] + alpha)
    out[-1] = coef * (beta - 2.0 * u[-1] + u[-2])
    out[1:-1] = coef * (u[2:] - 2.0 * u[1:-1] + u[:-2])


def rk4_step(u, dt, coef, alpha, beta):
    k1 = np.empty_like(u)
    k2 = np.empty_like(u)
    k3 = np.empty_like(u)
    k4 = np.empty_like(u)

    rhs(u, k1, coef, alpha, beta)
    rhs(u + 0.5 * dt * k1, k2, coef, alpha, beta)
    rhs(u + 0.5 * dt * k2, k3, coef, alpha, beta)
    rhs(u + dt * k3, k4, coef, alpha, beta)

    return u + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def add_boundaries(u_inner, alpha, beta):
    u = np.empty(u_inner.size + 2)
    u[0] = alpha
    u[-1] = beta
    u[1:-1] = u_inner
    return u


def snapshot_indices(t, T):
    points = np.array([
        0.00 * T,
        0.05 * T,
        0.10 * T,
        0.25 * T,
        0.50 * T,
        0.75 * T,
        1.00 * T
    ])

    ids = [int(np.argmin(np.abs(t - point))) for point in points]
    return np.unique(ids)


def analytical_solution(y, t, cfg):
    U = np.zeros((len(t), len(y)))
    base = cfg.alpha + (cfg.beta - cfg.alpha) * y / cfg.L
    n_values = np.arange(1, cfg.series_terms + 1)

    for i, time_value in enumerate(t):
        row = base.copy()

        for n in n_values:
            coefficient = (2.0 / (np.pi * n)) * (((-1.0) ** n) * cfg.beta - cfg.alpha)
            exponent = np.exp(-cfg.a * (np.pi * n / cfg.L) ** 2 * time_value)
            harmonic = np.sin(np.pi * n * y / cfg.L)
            row += coefficient * exponent * harmonic

        row[0] = cfg.alpha
        row[-1] = cfg.beta
        U[i] = row

    return U


def solve_heat_problem(cfg):
    T = cfg.T_hours * 3600.0

    y, dy = spatial_grid(cfg.L, cfg.N)

    dt_limit = dy * dy / (2.0 * cfg.a)
    t_full, dt, steps = corrected_time_grid(T, cfg.dt_requested, dt_limit)

    ids = snapshot_indices(t_full, T)
    t_snap = t_full[ids]

    coef = cfg.a / (dy * dy)
    u_inner = np.full(cfg.N, cfg.phi0)

    U_num = []
    center_u = np.empty(steps + 1)
    center_id = (cfg.N + 2) // 2
    center_u[0] = add_boundaries(u_inner, cfg.alpha, cfg.beta)[center_id]

    snap_pointer = 0

    for step in range(steps + 1):
        if snap_pointer < len(ids) and step == ids[snap_pointer]:
            U_num.append(add_boundaries(u_inner, cfg.alpha, cfg.beta))
            snap_pointer += 1

        if step < steps:
            u_inner = rk4_step(u_inner, dt, coef, cfg.alpha, cfg.beta)
            center_u[step + 1] = add_boundaries(u_inner, cfg.alpha, cfg.beta)[center_id]

    U_num = np.array(U_num)
    U_an = analytical_solution(y, t_snap, cfg)

    error = U_num[1:, 1:-1] - U_an[1:, 1:-1]
    mae = float(np.max(np.abs(error)))
    mse = float(np.mean(error ** 2))

    return HeatResult(
        y=y,
        t_snap=t_snap,
        U_num=U_num,
        U_an=U_an,
        t_full=t_full,
        center_u=center_u,
        dy=dy,
        dt=dt,
        dt_limit=dt_limit,
        steps=steps,
        mae=mae,
        mse=mse
    )


def print_summary(cfg, res):
    print("ЛАБОРАТОРНА РОБОТА №3")
    print("Моделювання процесу теплопровідності")
    print()
    print("Варіант 15")
    print(f"Матеріал: {cfg.material}")
    print(f"a = {cfg.a:.8e} м²/с")
    print(f"L = {cfg.L:.3f} м")
    print(f"T = {cfg.T_hours:.2f} год")
    print(f"N = {cfg.N}")
    print(f"α = {cfg.alpha:.2f} °C")
    print(f"β = {cfg.beta:.2f} °C")
    print(f"φ(y) = {cfg.phi0:.2f} °C")
    print()
    print("Сітка та інтегрування")
    print(f"Δy = {res.dy:.8f} м")
    print(f"dt max = {res.dt_limit:.6f} с")
    print(f"dt RK4 = {res.dt:.6f} с")
    print(f"Кількість кроків = {res.steps}")
    print()
    print("Похибки")
    print(f"MAE = {res.mae:.8e}")
    print(f"MSE = {res.mse:.8e}")


def plot_center_temperature(res):
    plt.figure(figsize=(8, 5))
    plt.plot(res.t_full / 3600.0, res.center_u)
    plt.xlabel("t, год")
    plt.ylabel("u(t, L/2), °C")
    plt.title("Температура в центральній точці стінки")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_temperature_profiles(res):
    plt.figure(figsize=(8, 5))

    for i in range(len(res.t_snap)):
        plt.plot(res.y, res.U_num[i], label=f"{res.t_snap[i] / 3600.0:.2f} год")

    plt.xlabel("y, м")
    plt.ylabel("u(t, y), °C")
    plt.title("Розподіл температури по товщині стінки")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_3d_solutions(res):
    mask = res.t_snap > 0.0

    Y, T = np.meshgrid(res.y, res.t_snap[mask] / 3600.0)

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(Y, T, res.U_num[mask], cmap="viridis", edgecolor="none")
    ax1.set_title("Числовий розв’язок")
    ax1.set_xlabel("y, м")
    ax1.set_ylabel("t, год")
    ax1.set_zlabel("u(t, y), °C")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(Y, T, res.U_an[mask], cmap="plasma", edgecolor="none")
    ax2.set_title("Аналітичний розв’язок")
    ax2.set_xlabel("y, м")
    ax2.set_ylabel("t, год")
    ax2.set_zlabel("u(t, y), °C")

    plt.tight_layout()
    plt.show()


def plot_absolute_error_surface(res):
    y = res.y[1:-1]
    t = res.t_snap[1:] / 3600.0

    E = np.abs(res.U_num[1:, 1:-1] - res.U_an[1:, 1:-1])
    Y, T = np.meshgrid(y, t)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(Y, T, E, cmap="viridis", edgecolor="none")

    ax.set_title("Локальна абсолютна похибка")
    ax.set_xlabel("y, м")
    ax.set_ylabel("t, год")
    ax.set_zlabel("|u_num - u_an|, °C")

    plt.tight_layout()
    plt.show()


def plot_error_metrics(res):
    error = res.U_num[1:, 1:-1] - res.U_an[1:, 1:-1]

    t = res.t_snap[1:] / 3600.0
    mae_t = np.max(np.abs(error), axis=1)
    mse_t = np.mean(error ** 2, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(t, mae_t)
    axes[0].set_xlabel("t, год")
    axes[0].set_ylabel("MAE(t), °C")
    axes[0].set_title("Максимальна абсолютна похибка в часі")
    axes[0].grid(True)

    axes[1].plot(t, mse_t)
    axes[1].set_xlabel("t, год")
    axes[1].set_ylabel("MSE(t)")
    axes[1].set_title("Середньоквадратична похибка в часі")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    cfg = HeatConfig()
    res = solve_heat_problem(cfg)

    print_summary(cfg, res)

    plot_center_temperature(res)
    plot_temperature_profiles(res)
    plot_3d_solutions(res)
    plot_absolute_error_surface(res)
    plot_error_metrics(res)


main()