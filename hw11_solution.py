import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve

a = 0
b = 1
h = 0.1
N = int((b - a) / h)
x = np.linspace(a, b, N + 1)
ya = 1
yb = 2

def f(x, y, yp):
    return (1 + np.exp(-x) - 2 * (1 + x) * yp) / (1 + x)**2

def shooting_method():
    def ode_system(x, Y):
        return [Y[1], f(x, Y[0], Y[1])]

    def boundary_error(slope_guess):
        sol = solve_ivp(ode_system, [a, b], [ya, slope_guess], t_eval=[b])
        return sol.y[0, -1] - yb

    s0, s1 = 0, 1
    tol = 1e-6
    for _ in range(100):
        error0 = boundary_error(s0)
        error1 = boundary_error(s1)
        s_new = s1 - error1 * (s1 - s0) / (error1 - error0)
        if abs(s_new - s1) < tol:
            break
        s0, s1 = s1, s_new

    sol_full = solve_ivp(ode_system, [a, b], [ya, s_new], t_eval=x)
    return sol_full.y[0]

def finite_difference_method_fixed():
    A = np.zeros((N - 1, N - 1))
    B = np.zeros(N - 1)

    for i in range(1, N):
        xi = x[i]
        coeff = (1 + xi)**2
        p = -2 * (1 + xi)
        A[i-1, i-1] = -2 / h**2 + p / (h * coeff)
        if i != 1:
            A[i-1, i-2] = 1 / h**2 - p / (2 * h * coeff)
        if i != N-1:
            A[i-1, i] = 1 / h**2 + p / (2 * h * coeff)
        B[i-1] = -(1 + np.exp(-xi)) / coeff

    B[0] -= (1 / h**2 - (-2 * (1 + x[1])) / (2 * h * (1 + x[1])**2)) * ya
    B[-1] -= (1 / h**2 + (-2 * (1 + x[N-1])) / (2 * h * (1 + x[N-1])**2)) * yb

    y_inner = solve(A, B)
    y_full = np.concatenate(([ya], y_inner, [yb]))
    return y_full

def variation_method():
    A = np.zeros((N - 1, N - 1))
    B = np.zeros(N - 1)

    for i in range(1, N):
        xi = x[i]
        w = 1 + xi
        A[i-1, i-1] = 2 * w**2 / h**2
        if i != 1:
            A[i-1, i-2] = -w**2 / h**2 - w / (2 * h)
        if i != N-1:
            A[i-1, i] = -w**2 / h**2 + w / (2 * h)
        B[i-1] = 1 + np.exp(-xi)

    B[0] += (1 + x[0])**2 / h**2 * ya
    B[-1] += (1 + x[-1])**2 / h**2 * yb

    y_inner = solve(A, B)
    y_full = np.concatenate(([ya], y_inner, [yb]))
    return y_full

if __name__ == "__main__":
    y_shooting = shooting_method()
    print("Q1: Shooting Method")
    for xi, yi in zip(x, y_shooting):
        print(f"x = {xi:.1f}, y = {yi:.6f}")

    y_fd_fixed = finite_difference_method_fixed()
    print("\nQ2: Finite Difference Method (Fixed)")
    for xi, yi in zip(x, y_fd_fixed):
        print(f"x = {xi:.1f}, y = {yi:.6f}")

    y_variation = variation_method()
    print("\nQ3: Variation Method")
    for xi, yi in zip(x, y_variation):
        print(f"x = {xi:.1f}, y = {yi:.6f}")
