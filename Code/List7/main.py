from RungeKutta import RungeKutta
from Galerkin import Galerkin
import numpy as np

def RungeKuttaMethod()->None:
    """
    Runs the Runge-Kutta method for exercise 1
    """
    ft = lambda t, y: 1 + (t - y) ** 2
        
    rk = RungeKutta(ft, 2, 3, 0.1, 1)
    
    # 3/8 rule 4th order Runge-Kutta method
    rk.SetButcherTableau(method = "ThreeEights")
    rk.RungeKuttaMethod()

    print(rk.sol)

    rk.WriteResults("List7/Results_RK.txt")

    return

def GalerkinMethod()->None:
    """
    Runs the Galerkin Method for exercise 2
    """
    uex = lambda x, y: (x + 1) * (y + 1) * np.arctan(x-1) * np.arctan(y-1)

    dudx = lambda x, y: -((1 + x) * (1 + y) * np.arctan(1 - y)) / (1 + (1 - x) ** 2) + (1 + y) * np.arctan(1 - x) * np.arctan(1 - y)
    dudy = lambda x, y: -((1 + x) * (1 + y) * np.arctan(1 - x)) / (1 + (1 - y) ** 2) + (1 + x) * np.arctan(1 - x) * np.arctan(1 - y)

    laplacian = lambda x, y: (2 * (1 + x) * np.arctan(1 - x)) / (1 + (1 - y) ** 2) + (2 * (1 + x) * (1 - y) * (1 + y) * np.arctan(1 - x)) / (1 + (1 - y) ** 2) ** 2 + (2 * (1 + y) * np.arctan(1 - y)) / (1 + (1 - x) ** 2) + (2 * (1 + y) * (1 - x) * (1 + x) * np.arctan(1 - y)) / (1 + (1 - x) ** 2) ** 2

    g = Galerkin(uex, laplacian, dudx, dudy, 5)

    g.Run()

    print(g.alpha)
    print(g.error)

def main()->None:
    exercise = 1
    if exercise == 1:
        RungeKuttaMethod()

    if exercise == 2:
        GalerkinMethod()

    else:
        raise ValueError("Invalid exercise number")


    return 

if __name__ == "__main__":
    main()