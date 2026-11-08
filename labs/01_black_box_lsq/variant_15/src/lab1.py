import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


VARIANT_NUMBER = 15


def create_experimental_table(variant):
    input_1 = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=float)
    input_2 = np.array([1.5, 2.5, 3.5, 1.5, 3.5, 1.5, 2.5, 2.5], dtype=float)

    output = np.array([
        2.3,
        4 + 0.3 * variant,
        2 - 0.1 * variant,
        5 - 0.2 * variant,
        4 - 0.2 * variant,
        6.1 + 0.2 * variant,
        6.5 - 0.1 * variant,
        7.2
    ], dtype=float)

    return input_1, input_2, output


def form_coefficient_matrix(input_1, input_2, output):
    observations_count = len(output)

    matrix = np.array([
        [observations_count, np.sum(input_1), np.sum(input_2)],
        [np.sum(input_1), np.sum(input_1 ** 2), np.sum(input_1 * input_2)],
        [np.sum(input_2), np.sum(input_1 * input_2), np.sum(input_2 ** 2)]
    ], dtype=float)

    vector = np.array([
        np.sum(output),
        np.sum(input_1 * output),
        np.sum(input_2 * output)
    ], dtype=float)

    return matrix, vector


def solve_linear_equations(matrix, vector):
    return np.linalg.solve(matrix, vector)


def calculate_model_values(coefficients, input_1, input_2):
    a0, a1, a2 = coefficients
    return a0 + a1 * input_1 + a2 * input_2


def calculate_determination_coefficient(actual_values, predicted_values):
    residual_sum = np.sum((actual_values - predicted_values) ** 2)
    total_sum = np.sum((actual_values - np.mean(actual_values)) ** 2)
    return 1 - residual_sum / total_sum


def print_input_data(input_1, input_2, output):
    table = []

    for index in range(len(output)):
        table.append([
            index + 1,
            input_1[index],
            input_2[index],
            output[index]
        ])

    print("\nВхідні експериментальні дані")
    print(tabulate(
        table,
        headers=["№", "x1", "x2", "y"],
        tablefmt="grid",
        floatfmt=".4f"
    ))


def print_normal_system(matrix, vector):
    table = []

    for index in range(len(vector)):
        table.append([
            matrix[index, 0],
            matrix[index, 1],
            matrix[index, 2],
            vector[index]
        ])

    print("\nСистема нормальних рівнянь")
    print(tabulate(
        table,
        headers=["a0", "a1", "a2", "Права частина"],
        tablefmt="grid",
        floatfmt=".4f"
    ))


def print_model_results(coefficients, r_squared, control_value):
    a0, a1, a2 = coefficients

    table = [
        ["a0", a0],
        ["a1", a1],
        ["a2", a2],
        ["R²", r_squared],
        ["y(1.5; 3)", control_value]
    ]

    print("\nРезультати побудови моделі")
    print(tabulate(
        table,
        headers=["Показник", "Значення"],
        tablefmt="grid",
        floatfmt=".6f"
    ))

    print("\nОтримана модель:")
    print(f"y = {a0:.6f} + ({a1:.6f})x1 + ({a2:.6f})x2")


def print_comparison_table(input_1, input_2, actual_values, predicted_values):
    table = []

    for index in range(len(actual_values)):
        table.append([
            index + 1,
            input_1[index],
            input_2[index],
            actual_values[index],
            predicted_values[index],
            actual_values[index] - predicted_values[index]
        ])

    print("\nПорівняння фактичних та розрахункових значень")
    print(tabulate(
        table,
        headers=["№", "x1", "x2", "y фактичне", "y модельне", "Похибка"],
        tablefmt="grid",
        floatfmt=".6f"
    ))


def draw_regression_surface(input_1, input_2, output, coefficients):
    a0, a1, a2 = coefficients

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    predicted_output = a0 + a1 * input_1 + a2 * input_2

    x1_vals = np.linspace(min(input_1) - 0.2, max(input_1) + 0.2, 35)
    x2_vals = np.linspace(min(input_2) - 0.2, max(input_2) + 0.2, 35)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Y = a0 + a1 * X1 + a2 * X2

    ax.plot_surface(
        X1,
        X2,
        Y,
        cmap="viridis",
        alpha=0.25,
        linewidth=0,
        antialiased=True
    )

    ax.scatter(
        input_1,
        input_2,
        output,
        color="red",
        s=90,
        edgecolor="black",
        label="Фактичні значення y"
    )

    ax.scatter(
        input_1,
        input_2,
        predicted_output,
        color="blue",
        s=45,
        marker="x",
        label="Значення на площині"
    )

    for i in range(len(output)):
        linestyle = "-" if output[i] >= predicted_output[i] else "--"

        ax.plot(
            [input_1[i], input_1[i]],
            [input_2[i], input_2[i]],
            [output[i], predicted_output[i]],
            color="black",
            linestyle=linestyle,
            linewidth=1.4
        )

        ax.text(
            input_1[i],
            input_2[i],
            output[i] + 0.25,
            str(i + 1),
            fontsize=9
        )

    ax.set_title("Апроксимація залежності методом найменших квадратів")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")

    ax.view_init(elev=24, azim=135)
    ax.legend()

    plt.tight_layout()
    plt.show()

def run_calculation():
    input_1, input_2, output = create_experimental_table(VARIANT_NUMBER)

    normal_matrix, normal_vector = form_coefficient_matrix(input_1, input_2, output)
    coefficients = solve_linear_equations(normal_matrix, normal_vector)

    predicted_output = calculate_model_values(coefficients, input_1, input_2)
    r_squared = calculate_determination_coefficient(output, predicted_output)
    control_value = calculate_model_values(coefficients, 1.5, 3.0)

    print(f"Номер варіанта: {VARIANT_NUMBER}")

    print_input_data(input_1, input_2, output)
    print_normal_system(normal_matrix, normal_vector)
    print_model_results(coefficients, r_squared, control_value)
    print_comparison_table(input_1, input_2, output, predicted_output)

    draw_regression_surface(input_1, input_2, output, coefficients)


run_calculation()