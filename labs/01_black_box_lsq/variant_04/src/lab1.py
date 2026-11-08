import matplotlib.pyplot as plt
import numpy as np

def generate_data(n):
    x1 = [0, 0, 0, 1, 1, 2, 2, 2]
    x2 = [1.5, 2.5, 3.5, 1.5, 3.5, 1.5, 2.5, 2.5]
    y = [2.3, 4 + 0.3*n, 2 - 0.1*n, 5 - 0.2*n, 4 - 0.2*n, 6.1 + 0.2*n, 6.5 - 0.1*n, 7.2]
    return x1, x2, y

def build_normal_system(x1, x2, y):
    N = len(x1)

    Sx1 = sum(x1)
    Sx2 = sum(x2)
    Sy = sum(y)

    Sx1x1 = sum(xi*xi for xi in x1)
    Sx2x2 = sum(xi*xi for xi in x2)
    Sx1x2 = sum(x1[i]*x2[i] for i in range(N))

    Sx1y = sum(x1[i]*y[i] for i in range(N))
    Sx2y = sum(x2[i]*y[i] for i in range(N))

    A = [ [N, Sx1, Sx2],
          [Sx1, Sx1x1, Sx1x2],
          [Sx2, Sx1x2, Sx2x2] ]

    B = [Sy, Sx1y, Sx2y]

    return A, B

def solve_gauss(A, B):
    n = len(A)
    M = [A[i][:] + [B[i]] for i in range(n)]

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot_row][k]) < 1e-12:
            raise ValueError("СЛР невизначена")

        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]

        pivot = M[k][k]
        for j in range(k, n + 1):
            M[k][j] /= pivot

        for i in range(k + 1, n):
            factor = M[i][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))

    return x[0], x[1], x[2]

def predict(a0, a1, a2, x1, x2):
    return a0 + a1*x1 + a2*x2

def r2_score(y, y_hat):
    n = len(y)
    y_bar = sum(y) / n
    MSE = sum((y[i] - y_hat[i])**2 for i in range(n)) / n
    VAR = sum((y[i] - y_bar)**2 for i in range(n)) / n
    return 1 - MSE / VAR

def plot_model(x1, x2, y, a0, a1, a2):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x1, x2, y)

    x1g = np.linspace(min(x1), max(x1), 30)
    x2g = np.linspace(min(x2), max(x2), 30)
    X1g, X2g = np.meshgrid(x1g, x2g)
    Yg = a0 + a1*X1g + a2*X2g

    ax.plot_surface(X1g, X2g, Yg, alpha=0.5)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")

    plt.show()

def main():
    n = 4
    x1, x2, y = generate_data(n)
    A, B = build_normal_system(x1, x2, y)
    a0, a1, a2 = solve_gauss(A, B)

    print("a0 =", a0)
    print("a1 =", a1)
    print("a2 =", a2)

    y_hat = [predict(a0, a1, a2, x1[i], x2[i]) for i in range(len(x1))]
    R2 = r2_score(y, y_hat)

    print("R^2 =", R2)

    y_q = predict(a0, a1, a2, 1.5, 3.0)
    print("y(1.5, 3) =", y_q)

    plot_model(x1, x2, y, a0, a1, a2)

if __name__ == "__main__":
    main()