import numpy as np
import matplotlib.pyplot as plt


def params_from_variant(N: int):
    A = np.array([
        [0.01 * N, 0.0001 * N],
        [0.0001 * N, 0.04 * N]
    ], dtype=float)

    x0 = 1000 - 10 * N
    y0 = 700 - 10 * N

    t0 = 0.0
    T = 150.0
    h = 0.1

    return A, np.array([x0, y0], dtype=float), t0, T, h


def make_time_grid(t0: float, T: float, h: float):
    steps = int(round((T - t0) / h))
    return np.linspace(t0, T, steps + 1, dtype=float)


def lotka_volterra_rhs(t: float, z: np.ndarray, A: np.ndarray):
    x, y = z
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]
    dx = a11 * x - a12 * x * y
    dy = -a22 * y + a21 * x * y
    return np.array([dx, dy], dtype=float)


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


def plot_results(t: np.ndarray, x: np.ndarray, y: np.ndarray):
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("t, дні")
    plt.ylabel("x(t), жертви")
    plt.title("Залежність x(t)")
    plt.grid(True)

    plt.figure()
    plt.plot(t, y)
    plt.xlabel("t, дні")
    plt.ylabel("y(t), хижаки")
    plt.title("Залежність y(t)")
    plt.grid(True)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x, жертви")
    plt.ylabel("y, хижаки")
    plt.title("Фазовий портрет y(x)")
    plt.grid(True)

    plt.show()


def main(N: int = 4):
    A, z0, t0, T, h = params_from_variant(N)
    t = make_time_grid(t0, T, h)
    z = rk4_solve(lotka_volterra_rhs, t, z0, args=(A,))
    x = z[:, 0]
    y = z[:, 1]
    plot_results(t, x, y)

if __name__ == "__main__":
    main(4)