from NonLinearSolver import NonLinearSolver
import numpy as np

# ---- Alias for numpy functions ----
COS: callable = np.cos
SIN: callable = np.sin

E: float = np.e
PI: float = np.pi

def PlotSolution(diff_Newton, diff_Broyden, diff_log_Newton, diff_log_Broyden) -> None:
    """
    Plot the solution
    """
    import matplotlib.pyplot as plt
    x = range(len(diff_Newton))
    x2 = range(len(diff_log_Newton))

    plt.figure()
    plt.semilogy(x, diff_Newton, marker="o", label="Newton")
    plt.semilogy(x, diff_Broyden[:-1], marker="o", label="Broyden")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x2, diff_log_Newton, marker="o", label="Newton")
    plt.plot(x2, diff_log_Broyden[:-1], marker="o", label="Broyden")
    plt.xlabel("Iteration")
    plt.ylabel("Convergence rate")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    system = 2
    if system == 1:
        eq1 = lambda x, y: E ** (x * y) + x ** 2 + y - 1.2
        eq2 = lambda x, y: x ** 2 + y ** 2 + x - 0.55

        grad_eq1 = lambda x, y: [y * E ** (x * y) + 2 * x, x * E ** (x * y) + 1]
        grad_eq2 = lambda x, y: [2 * x + 1, 2 * y]

        G = [eq1, eq2]
        grad_G = [grad_eq1, grad_eq2]

        x0 = [.1 for _ in range(2)]
        exact = [0.39384945283486894, 0.032142738943740555]

        n_iter = 10

    elif system == 2:
        eq1 = lambda x, y, _: -x * COS(y) - 1
        eq2 = lambda x, y, z: x * y + z - 2
        eq3 = lambda x, y, z: E ** (-z) * SIN(x + y) + x ** 2 + y ** 2 - 1

        grad_eq1 = lambda x, y, _: [-COS(y), x * SIN(y), 0]
        grad_eq2 = lambda x, y, _: [y, x, 1]
        grad_eq3 = lambda x, y, z: [E ** (-z) * COS(x + y) + 2 * x, E ** (-z) * COS(x + y) + 2 * y, -E ** (-z) * SIN(x + y)]

        G = [eq1, eq2, eq3]
        grad_G = [grad_eq1, grad_eq2, grad_eq3]

        x0 = [0.1 for _ in range(3)]
        exact = [-1.0202573099857735, 0.1996053775230006, 2.2036488455303114]

        n_iter = 25

    solver_newton = NonLinearSolver(G, grad_G, x0)
    solver_newton.SetMaxIteration(n_iter)
    solver_newton.SetMethod("Newton")
    solver_newton.SetExactSolution(exact)

    solver_newton.Solve()
    diff_Newton, diff_log_Newton = solver_newton.GetError()

    solver_broyden = NonLinearSolver(G, grad_G, x0)
    solver_broyden.SetMaxIteration(n_iter)
    solver_broyden.SetMethod("Broyden")
    solver_broyden.SetExactSolution(exact)

    solver_broyden.Solve()
    diff_Broyden, diff_log_Broyden = solver_broyden.GetError()

    solver_newton.WriteSolution(f"List5/Newton{system}.txt")
    solver_broyden.WriteSolution(f"List5/Broyden{system}.txt")

    PlotSolution(diff_Newton, diff_Broyden, diff_log_Newton, diff_log_Broyden)

if __name__ == "__main__":
    main()