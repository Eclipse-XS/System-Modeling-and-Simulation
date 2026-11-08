import numpy as np
import matplotlib.pyplot as plt

def build_snapshot_times(T):
    t1 = np.linspace(0.0, 10.0, 81)
    t2 = np.linspace(10.0, 120.0, 81)[1:]
    t3 = np.linspace(120.0, T, 31)[1:]
    return np.concatenate([t1, t2, t3])

def analytic_solution_sine_series(t, y, a, L, alpha, beta, terms=30):
    y = np.asarray(y, dtype=float)
    base = alpha + (beta - alpha) * (y / L)
    n = np.arange(1, terms + 1, dtype=float)
    coeff = (2.0 / (np.pi * n)) * (((-1.0) ** n) * beta - alpha)
    sin_part = np.sin((np.pi * n[:, None] * y[None, :]) / L)
    expo = np.exp(-a * ((np.pi * n / L) ** 2) * t)[:, None]
    return base + np.sum(coeff[:, None] * sin_part * expo, axis=0)

def solve_heat_equation_rk4(a, L, T, N, alpha, beta, phi=0.0, h_user=0.002, series_terms=30):
    delta = L / (N + 1)
    mu = a / (delta * delta)

    dt_max = (delta * delta) / (2.0 * a)
    h = min(h_user, 0.9 * dt_max)
    steps_total = int(np.ceil(T / h))
    h = T / steps_total

    y = np.linspace(0.0, L, N + 2)
    u = np.full(N, float(phi), dtype=float)

    k1 = np.empty_like(u)
    k2 = np.empty_like(u)
    k3 = np.empty_like(u)
    k4 = np.empty_like(u)
    tmp = np.empty_like(u)

    def rhs(u_inner, out):
        out[1:-1] = mu * (u_inner[2:] - 2.0 * u_inner[1:-1] + u_inner[:-2])
        out[0] = mu * (u_inner[1] - 2.0 * u_inner[0] + alpha)
        out[-1] = mu * (beta - 2.0 * u_inner[-1] + u_inner[-2])

    def rk4_step(u_inner, h_step):
        rhs(u_inner, k1)
        tmp[:] = u_inner + 0.5 * h_step * k1
        rhs(tmp, k2)
        tmp[:] = u_inner + 0.5 * h_step * k2
        rhs(tmp, k3)
        tmp[:] = u_inner + h_step * k3
        rhs(tmp, k4)
        u_inner[:] = u_inner + (h_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def assemble_profile(u_inner):
        prof = np.empty(N + 2, dtype=float)
        prof[0] = alpha
        prof[-1] = beta
        prof[1:-1] = u_inner
        return prof

    t_grid = build_snapshot_times(T)
    M_time = t_grid.size

    U_num = np.empty((M_time, N + 2), dtype=float)
    U_num[0, :] = assemble_profile(u)

    mid_idx = (N + 2) // 2
    mid_t = np.empty(steps_total + 1, dtype=float)
    mid_u = np.empty(steps_total + 1, dtype=float)
    mid_t[0] = 0.0
    mid_u[0] = U_num[0, mid_idx]

    print_every = max(1, steps_total // 50)
    snap_idx = 1
    t = 0.0
    prof = U_num[0, :].copy()

    for step in range(1, steps_total + 1):
        rk4_step(u, h)
        t = step * h
        prof = assemble_profile(u)

        mid_t[step] = t
        mid_u[step] = prof[mid_idx]

        if step % print_every == 0 or step == steps_total:
            pct = 100.0 * step / steps_total
            print(f"iter {step}/{steps_total} ({pct:.1f}%), t={t:.2f}s, h={h:.3e}s")

        while snap_idx < M_time and t + 1e-12 >= t_grid[snap_idx]:
            U_num[snap_idx, :] = prof
            snap_idx += 1

    U_an = np.empty_like(U_num)
    for i, ti in enumerate(t_grid):
        U_an[i, :] = analytic_solution_sine_series(ti, y, a, L, alpha, beta, terms=series_terms)

    diff = U_num - U_an
    meta = {
        "delta": delta,
        "dt_max": dt_max,
        "h_used": h,
        "steps_total": steps_total,
        "mae": float(np.max(np.abs(diff))),
        "mse": float(np.mean(diff * diff)),
        "t_grid": t_grid,
        "y": y
    }
    return U_num, U_an, mid_t, mid_u, meta

def plot_heat_solution_diagnostics(U_num, U_an, mid_t, mid_u, meta):
    y = meta["y"]
    t_grid = meta["t_grid"]

    print(f"delta = {meta['delta']:.6e} m")
    print(f"dt_max(stability) = {meta['dt_max']:.6e} s")
    print(f"h_used = {meta['h_used']:.6e} s")
    print(f"steps_total = {meta['steps_total']}")
    print(f"MAE = {meta['mae']:.6e}")
    print(f"MSE = {meta['mse']:.6e}")

    plt.figure()
    plt.plot(mid_t / 3600.0, mid_u)
    plt.xlabel("t, h")
    plt.ylabel("Temperature at mid-plane, °C")
    plt.title("Mid-plane temperature vs time (hours)")
    plt.show()

    mask = mid_t <= 120.0
    plt.figure()
    plt.plot(mid_t[mask], mid_u[mask])
    plt.xlabel("t, s")
    plt.ylabel("Temperature at mid-plane, °C")
    plt.title("Mid-plane temperature vs time (0..120 s)")
    plt.show()

    snap_sec = np.array([0.0, 0.2, 1.0, 5.0, 20.0])
    plt.figure()
    for ts in snap_sec:
        idx = int(np.argmin(np.abs(t_grid - ts)))
        plt.plot(y, U_num[idx, :], label=f"t={t_grid[idx]:.1f}s")
    plt.xlabel("y, m")
    plt.ylabel("Temperature, °C")
    plt.title("Temperature distribution in the wall (numerical) - 5 distinct snapshots")
    plt.legend()
    plt.show()

    mask3d = t_grid >= 1.0
    Y, TT = np.meshgrid(y, (t_grid[mask3d] / 3600.0))
    U_num_3d = U_num[mask3d, :]
    U_an_3d = U_an[mask3d, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Y, TT, U_num_3d, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.85)
    ax.plot_wireframe(Y, TT, U_an_3d, rstride=6, cstride=6, linewidth=0.8)
    ax.set_xlabel("y, m")
    ax.set_ylabel("t, h")
    ax.set_zlabel("Temperature, °C")
    ax.set_title("Heat equation: numerical (surface) + analytic (wireframe), t>=1s")
    plt.show()

a = 111.0e-6
L = 0.07
T = 1.0 * 3600.0
N = 100
alpha = 4.0
beta = 23.0
phi = 0.0

U_num, U_an, mid_t, mid_u, meta = solve_heat_equation_rk4(
    a=a,
    L=L,
    T=T,
    N=N,
    alpha=alpha,
    beta=beta,
    phi=phi,
    h_user=0.002,
    series_terms=30
)

plot_heat_solution_diagnostics(U_num, U_an, mid_t, mid_u, meta)