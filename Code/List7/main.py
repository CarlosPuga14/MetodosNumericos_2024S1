from RungeKutta import RungeKutta
import numpy as np
from scipy.special import eval_legendre

def RungeKuttaExample()->None:
    """
    Run the Runge-Kutta method for exercise 1
    """
    ft = lambda t, y: 1 + (t - y) ** 2
        
    rk = RungeKutta(ft, 2, 3, 0.1, 1)
    
    # 3/8 rule 4th order Runge-Kutta method
    rk.SetButcherTableau(method = "ThreeEights")
    rk.RungeKuttaMethod()

    print(rk.sol)

    rk.WriteResults("List7/Results_RK.txt")

    return

def main()->None:
    exercise = 1

    if exercise == 1:
        RungeKuttaExample()

    else:
        raise ValueError("Invalid exercise number")


    return 

if __name__ == "__main__":
    main()