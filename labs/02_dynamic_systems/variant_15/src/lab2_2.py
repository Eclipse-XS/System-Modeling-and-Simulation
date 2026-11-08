import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class EpidemicConfig:
    variant: int = 15
    t_start: float = 0.0
    t_end: float = 40.0
    step: float = 0.1


class SIRModel:
    def __init__(self, config: EpidemicConfig):
        self.config = config
        self.N = config.variant
        self.H = 1000 - self.N
        self.beta = 25 - self.N
        self.gamma_days = self.N
        self.gamma = 1 / self.gamma_days
        self.initial_state = self._build_initial_state()
        self.time = self._build_time_grid()
        self._validate()

    def _build_initial_state(self):
        x0 = 900 - self.N
        y0 = 90 - self.N
        z0 = self.H - x0 - y0
        return np.array([x0, y0, z0], dtype=float)

    def _build_time_grid(self):
        steps = int(round((self.config.t_end - self.config.t_start) / self.config.step))
        return np.linspace(self.config.t_start, self.config.t_end, steps + 1)

    def _validate(self):
        if self.H <= 0:
            raise ValueError("H має бути додатним")
        if self.beta < 0:
            raise ValueError("beta має бути невід’ємним")
        if self.gamma_days <= 0:
            raise ValueError("gamma_days має бути додатним")
        if np.any(self.initial_state < 0):
            raise ValueError("Початкові умови мають бути невід’ємними")
        if abs(np.sum(self.initial_state) - self.H) > 1e-9:
            raise ValueError("x0 + y0 + z0 має дорівнювати H")

    def rhs(self, t, state):
        x, y, z = state

        infection_rate = (self.beta / self.H) * x * y

        dx = -infection_rate
        dy = infection_rate - self.gamma * y
        dz = self.gamma * y

        return np.array([dx, dy, dz], dtype=float)

    def solve_rk4(self):
        result = np.zeros((len(self.time), 3), dtype=float)
        result[0] = self.initial_state

        for i in range(len(self.time) - 1):
            t = self.time[i]
            h = self.time[i + 1] - self.time[i]
            state = result[i]

            k1 = self.rhs(t, state)
            k2 = self.rhs(t + h / 2, state + h * k1 / 2)
            k3 = self.rhs(t + h / 2, state + h * k2 / 2)
            k4 = self.rhs(t + h, state + h * k3)

            result[i + 1] = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return result


def print_summary(model, solution):
    x = solution[:, 0]
    y = solution[:, 1]
    z = solution[:, 2]
    t = model.time

    peak_index = np.argmax(y)

    print("=== Параметри SIR-моделі ===")
    print(f"N = {model.N}")
    print(f"H = {model.H}")
    print(f"beta = {model.beta}")
    print(f"gamma = 1 / {model.gamma_days} = {model.gamma:.4f}")
    print(f"x0 = {model.initial_state[0]:.0f}")
    print(f"y0 = {model.initial_state[1]:.0f}")
    print(f"z0 = {model.initial_state[2]:.0f}")

    print("\n=== Результати моделювання ===")
    print(f"Максимальна кількість хворих: {y[peak_index]:.2f}")
    print(f"День піку епідемії: {t[peak_index]:.2f}")
    print(f"Здорових наприкінці: {x[-1]:.2f}")
    print(f"Хворих наприкінці: {y[-1]:.2f}")
    print(f"Одужалих наприкінці: {z[-1]:.2f}")
    print(f"Перевірка суми x + y + z: {(x[-1] + y[-1] + z[-1]):.2f}")


def visualize(model, solution):
    t = model.time
    x = solution[:, 0]
    y = solution[:, 1]
    z = solution[:, 2]

    peak_index = np.argmax(y)

    plt.figure(figsize=(10, 5))
    plt.plot(t, x, linewidth=2)
    plt.fill_between(t, x, alpha=0.15)
    plt.title("Динаміка здорових людей x(t)")
    plt.xlabel("Час, дні")
    plt.ylabel("Кількість здорових")
    plt.grid(True, alpha=0.35)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "sir_healthy_x.png"), dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    plt.plot(t, y, linewidth=2)
    plt.fill_between(t, y, alpha=0.15)
    plt.scatter(t[peak_index], y[peak_index], s=80, label="Пік епідемії")
    plt.title("Динаміка хворих людей y(t)")
    plt.xlabel("Час, дні")
    plt.ylabel("Кількість хворих")
    plt.legend()
    plt.grid(True, alpha=0.35)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "sir_infected_y.png"), dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    plt.plot(t, z, linewidth=2)
    plt.fill_between(t, z, alpha=0.15)
    plt.title("Динаміка одужалих людей z(t)")
    plt.xlabel("Час, дні")
    plt.ylabel("Кількість одужалих")
    plt.grid(True, alpha=0.35)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "sir_recovered_z.png"), dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    plt.plot(t, x, linewidth=2, label="Здорові x(t)")
    plt.plot(t, y, linewidth=2, label="Хворі y(t)")
    plt.plot(t, z, linewidth=2, label="Одужалі z(t)")
    plt.scatter(t[peak_index], y[peak_index], s=80, label="Пік епідемії")
    plt.title(f"Загальна динаміка SIR-моделі для варіанта N = {model.N}")
    plt.xlabel("Час, дні")
    plt.ylabel("Кількість людей")
    plt.legend()
    plt.grid(True, alpha=0.35)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "sir_all_states.png"), dpi=300, bbox_inches="tight")

    plt.show()


def main():
    config = EpidemicConfig(variant=15)
    model = SIRModel(config)
    solution = model.solve_rk4()

    print_summary(model, solution)
    visualize(model, solution)


if __name__ == "__main__":
    main()