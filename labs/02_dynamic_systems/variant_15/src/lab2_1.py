import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class PredatorPreyConfig:
    variant: int = 15
    t_start: float = 0.0
    t_end: float = 150.0
    step: float = 0.1


class LotkaVolterraModel:
    def __init__(self, config: PredatorPreyConfig):
        self.config = config
        self.N = config.variant
        self.A = self._build_matrix()
        self.initial_state = self._build_initial_state()
        self.time = self._build_time_grid()

    def _build_matrix(self):
        return np.array([
            [0.01 * self.N, 0.0001 * self.N],
            [0.0001 * self.N, 0.04 * self.N]
        ], dtype=float)

    def _build_initial_state(self):
        return np.array([
            1000 - 10 * self.N,
            700 - 10 * self.N
        ], dtype=float)

    def _build_time_grid(self):
        steps = int(round((self.config.t_end - self.config.t_start) / self.config.step))
        return np.linspace(self.config.t_start, self.config.t_end, steps + 1)

    def rhs(self, t, state):
        x, y = state
        a11 = self.A[0, 0]
        a12 = self.A[0, 1]
        a21 = self.A[1, 0]
        a22 = self.A[1, 1]

        dx = a11 * x - a12 * x * y
        dy = a21 * x * y - a22 * y

        return np.array([dx, dy], dtype=float)

    def solve_rk4(self):
        result = np.zeros((len(self.time), 2), dtype=float)
        result[0] = self.initial_state

        for i in range(len(self.time) - 1):
            t = self.time[i]
            h = self.time[i + 1] - self.time[i]
            z = result[i]

            k1 = self.rhs(t, z)
            k2 = self.rhs(t + h / 2, z + h * k1 / 2)
            k3 = self.rhs(t + h / 2, z + h * k2 / 2)
            k4 = self.rhs(t + h, z + h * k3)

            result[i + 1] = z + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return result

    def equilibrium_point(self):
        a11 = self.A[0, 0]
        a12 = self.A[0, 1]
        a21 = self.A[1, 0]
        a22 = self.A[1, 1]

        x_eq = a22 / a21
        y_eq = a11 / a12

        return x_eq, y_eq


def visualize(model, solution):
    t = model.time
    x = solution[:, 0]
    y = solution[:, 1]
    x_eq, y_eq = model.equilibrium_point()

    plt.figure()
    plt.plot(t, x, linewidth=2)
    plt.fill_between(t, x, alpha=0.15)
    plt.title("x(t) — жертви")
    plt.xlabel("Час")
    plt.ylabel("Кількість")
    plt.grid(True)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "x_t.png"), dpi=300)

    plt.figure()
    plt.plot(t, y, linewidth=2)
    plt.fill_between(t, y, alpha=0.15)
    plt.title("y(t) — хижаки")
    plt.xlabel("Час")
    plt.ylabel("Кількість")
    plt.grid(True)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "y_t.png"), dpi=300)

    plt.figure()
    plt.plot(x, y, linewidth=2)

    step = max(1, len(x)//25)
    plt.quiver(
        x[:-1:step],
        y[:-1:step],
        x[1::step] - x[:-1:step],
        y[1::step] - y[:-1:step],
        angles="xy",
        scale_units="xy",
        scale=1
    )

    plt.scatter(x[0], y[0], label="start")
    plt.scatter(x_eq, y_eq, marker="x", label="equilibrium")

    plt.title("Фазовий портрет")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "phase.png"), dpi=300)

    plt.show()

def print_summary(model, solution):
    x = solution[:, 0]
    y = solution[:, 1]
    x_eq, y_eq = model.equilibrium_point()

    print("=== Параметри моделі ===")
    print(f"N = {model.N}")
    print(f"A =\n{model.A}")
    print(f"x0 = {model.initial_state[0]:.0f}")
    print(f"y0 = {model.initial_state[1]:.0f}")
    print(f"Стаціонарна точка: x* = {x_eq:.2f}, y* = {y_eq:.2f}")

    print("\n=== Результати моделювання ===")
    print(f"min x(t) = {x.min():.2f}")
    print(f"max x(t) = {x.max():.2f}")
    print(f"min y(t) = {y.min():.2f}")
    print(f"max y(t) = {y.max():.2f}")
    print(f"x(T) = {x[-1]:.2f}")
    print(f"y(T) = {y[-1]:.2f}")


def main():
    config = PredatorPreyConfig(variant=15)
    model = LotkaVolterraModel(config)
    solution = model.solve_rk4()

    print_summary(model, solution)
    visualize(model, solution)


if __name__ == "__main__":
    main()