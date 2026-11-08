import numpy as np
import matplotlib.pyplot as plt


def params_from_variant(N: int):
    H = 1000 - N
    beta = 25 - N
    gamma_days = N

    x0 = 900 - N
    y0 = 90 - N
    z0 = H - x0 - y0

    t0 = 0.0
    T = 40.0
    h = 0.1

    if H <= 0:
        raise ValueError("H = 1000 - N має бути > 0")
    if gamma_days <= 0:
        raise ValueError("gamma = N має бути > 0")
    if beta < 0:
        raise ValueError("beta = 25 - N має бути >= 0 для цього варіанту")
    if x0 < 0 or y0 < 0 or z0 < 0:
        raise ValueError("Початкові умови мають бути невід’ємними")
    if abs((x0 + y0 + z0) - H) > 1e-9:
        raise ValueError("x0 + y0 + z0 має дорівнювати H")

    p = np.array([H, beta, 1.0 / gamma_days], dtype=float)
    z_init = np.array([x0, y0, z0], dtype=float)

    return p, z_init, t0, T, h


def make_time_grid(t0: float, T: float, h: float):
    steps = int(round((T - t0) / h))
    return np.linspace(t0, T, steps + 1, dtype=float)


def sir_rhs(t: float, state: np.ndarray, p: np.ndarray):
    x, y, z = state
    H, beta, gamma = p

    inf = (beta / H) * x * y
    dx = -inf
    dy = inf - gamma * y
    dz = gamma * y


    return np.array([dx, dy, dz], dtype=float)


def rk4_solve(rhs, t: np.ndarray, z0: np.ndarray, args=()):
    z = np.zeros((t.size, z0.size), dtype=float)
    z[0] = z0
    for k in range(t.size - 1):
        h = t[k + 1] - t[k]
        zk = z[k]
        tk = t[k]

        k1 = rhs(tk, zk, *args)
        k2 = rhs(tk + 0.5 * h, zk + 0.5 * h * k1, *args)
        k3 = rhs(tk + 0.5 * h, zk + 0.5 * h * k2, *args)
        k4 = rhs(tk + h, zk + h * k3, *args)

        z[k + 1] = zk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return z


def plot_results(t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, t_zoom: float = 5.0):
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("t, дні")
    plt.ylabel("x(t), здорові")
    plt.title("Залежність x(t)")
    plt.grid(True)

    plt.figure()
    plt.plot(t, y)
    plt.xlabel("t, дні")
    plt.ylabel("y(t), хворі")
    plt.title("Залежність y(t)")
    plt.grid(True)

    plt.figure()
    plt.plot(t, z)
    plt.xlabel("t, дні")
    plt.ylabel("z(t), одужалі")
    plt.title("Залежність z(t)")
    plt.grid(True)

    m = t <= t_zoom

    plt.figure()
    plt.plot(t[m], x[m])
    plt.xlabel("t, дні")
    plt.ylabel("x(t), здорові")
    plt.title(f"x(t) за перші {t_zoom:g} днів")
    plt.grid(True)

    plt.figure()
    plt.plot(t[m], y[m])
    plt.xlabel("t, дні")
    plt.ylabel("y(t), хворі")
    plt.title(f"y(t) за перші {t_zoom:g} днів")
    plt.grid(True)

    plt.show()


def main(N: int = 4):
    p, z0, t0, T, h = params_from_variant(N)
    t = make_time_grid(t0, T, h)
    sol = rk4_solve(sir_rhs, t, z0, args=(p,))
    x = sol[:, 0]
    y = sol[:, 1]
    z = sol[:, 2]
    plot_results(t, x, y, z, t_zoom=5.0)


if __name__ == "__main__":
    main(4)