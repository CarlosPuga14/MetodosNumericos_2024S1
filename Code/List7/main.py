from RungeKutta import RungeKutta
from Galerkin import Galerkin
from LeastSquare import LeastSquare

import numpy as np
from matplotlib import pyplot as plt

def RungeKuttaMethod()->None:
    """
    Runs the Runge-Kutta method for exercise 1
    """
    ft = lambda t, y: 1 + (t - y) ** 2
        
    rk = RungeKutta(ft, 2, 3, 0.1, 1)
    
    # 3/8 rule 4th order Runge-Kutta method
    rk.SetButcherTableau(method = "ThreeEights")
    rk.Run()

    print(rk.sol)

    rk.WriteResults("List7/Results_RK.txt")

    return

def GalerkinMethod()->None:
    """
    Runs the Galerkin Method for exercise 2
    """
    uex = lambda x, y: (x + 1) * (y + 1) * np.arctan(x - 1) * np.arctan(y - 1)

    dudx = lambda x, y: -((1 + x) * (1 + y) * np.arctan(1 - y)) / (1 + (1 - x) ** 2) + (1 + y) * np.arctan(1 - x) * np.arctan(1 - y)
    dudy = lambda x, y: -((1 + x) * (1 + y) * np.arctan(1 - x)) / (1 + (1 - y) ** 2) + (1 + x) * np.arctan(1 - x) * np.arctan(1 - y)

    b = lambda x, y: (2 * (1 + x) * np.arctan(1 - x)) / (1 + (1 - y) ** 2) + (2 * (1 + x) * (1 - y) * (1 + y) * np.arctan(1 - x)) / (1 + (1 - y) ** 2) ** 2 + (2 * (1 + y) * np.arctan(1 - y)) / (1 + (1 - x) ** 2) + (2 * (1 + y) * (1 - x) * (1 + x) * np.arctan(1 - y)) / (1 + (1 - x) ** 2) ** 2

    g = Galerkin(uex, b, dudx, dudy, 5)

    g.Run()

    print(g.alpha)
    print(g.error)

    return 

def LeastSquareMethod()->None:
    """ 
    Runs the Least Square Method for exercise 3
    """
    data = [
    [0.017, 0.154],
    [0.02, 0.181],
    [0.025, 0.23],
    [0.085, 0.26],
    [0.087, 0.296],
    [0.111, 0.357],
    [0.119, 0.299],
    [0.171, 0.334],
    [0.174, 0.363],
    [0.21, 0.428],
    [0.211, 0.366],
    [0.233, 0.537],
    [0.783, 1.47],
    [0.999, 0.771],
    [1.11, 0.531],
    [1.29, 0.87],
    [1.32, 1.15],
    [1.35, 2.48],
    [1.69, 1.44],
    [1.74, 2.23],
    [2.75, 1.84],
    [3.02, 2.01],
    [3.04, 3.59],
    [3.34, 2.83],
    [4.09, 3.58],
    [4.28, 3.28],
    [4.29, 3.4],
    [4.58, 2.96],
    [4.68, 5.1],
    [4.83, 4.66],
    [5.3, 3.88],
    [5.45, 3.52],
    [5.48, 4.15],
    [5.53, 6.94],
    [5.96, 2.4]
    ]

    squares_non = LeastSquare(*zip(*data), "NonLinear")
    squares_non.Run()

    squares_log = LeastSquare(*zip(*data), "Logarithmic")
    squares_log.Run()

    dx = [d[0] for d in data]
    dy = [d[1] for d in data]

    plt.figure()
    plt.scatter(dx, dy, color = "black", label = "Data")
    plt.scatter(dx, squares_non.approx_solution, color = "red", label = "Non-Linear Approximation")
    plt.scatter(dx, squares_log.approx_solution, color = "blue", label = "Logarithmic Approximation")
    plt.grid(True)
    plt.xlabel(r"$x_i$")
    plt.ylabel(r"$y_i$")
    plt.legend()
    plt.show()

    print(squares_non.alpha)
    print(squares_non.total_error)
    print()
    print(squares_log.alpha)
    print(squares_log.total_error)

    return

def main()->None:
    exercise = 3
    if exercise == 1:
        RungeKuttaMethod()

    elif exercise == 2:
        GalerkinMethod()

    elif exercise == 3:
        LeastSquareMethod()

    else:
        raise ValueError("Invalid exercise number")


    return 

if __name__ == "__main__":
    main()