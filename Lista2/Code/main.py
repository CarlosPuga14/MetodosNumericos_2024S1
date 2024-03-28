"""
The main code to numerically integrate equations. 
This code has been used during the course IC639 - Metodos Numericos para Engenharia,
under the supervision of Prof. Dr. Philippe Devloo and Dr. Giovane Avancini

Created by Carlos Puga - 03/23/2024
"""
from Interval import Interval
from scipy import special
import numpy as np

#-------------------
#CONSTANTS & FUNCTIONS
#-------------------
PI: float = np.pi
COS: callable = np.cos
SIN: callable = np.sin
SQRT: callable = np.sqrt

ERF: callable = special.erf
SICI: callable = special.sici

#-------------------
#    PARAMETERS
#-------------------
epot: int = 4
method: int = 3
npoints: int = 3
ref_level: int = 5
n_divisions: int = 5

a: float = 0.
b: float = 8/PI

#-------------------
#    FUNCTION 1
#-------------------
def Function1(x:float)->float:
    return np.exp(-(x - 1)**2 /10**(-epot))

def IntFunction1(x:float)->float:
    const = (SQRT(10)**epot)
    return  SQRT(PI) * (ERF(const * (-1 + x)))/(2*const)

#-------------------
#    FUNCTION 2
#-------------------
def Function2(x:float)->float:
    return x * SIN(1 / x)

def IntFunction2(x:float)->float:
    arg = 1 / x
    si, _ = SICI(arg)

    return x / 2 * COS(arg) + x ** 2 / 2 * SIN(arg) + 1 / 2 * si

#-------------------
#    FUNCTION 3
#-------------------
def Function3(x:float)->float:
    if x >= 0 and x <= 1/PI:
        return 5 + 2 * x

    elif x <= 2 / PI:
        return (2 * x + 10 * PI * x - 5 * PI ** 2 * (-5 - 2 * x + x ** 2)) / (1 + 5 * PI ** 2)
    
    elif x <= 8 / PI:
        return (4 + 20 * PI ** 2 + 25 * PI ** 3) / (PI + 5 * PI ** 3) - 2 * SIN(4 / PI) + 2 * SIN(2 * x)
    
    else: 
        raise ValueError("Invalid x value")

def IntFunction3(x:float)->float:
    # integral of 5 + 2 * x
    def intf31(x:float)->float: return 5 * x + x ** 2

    # integral of (2 * x + 10 * PI * x - 5 * PI ** 2 * (-5 - 2 * x + x ** 2)) / (1 + 5 * PI ** 2)
    def intf32(x:float)->float: return (25 * (PI ** 2) * x + x ** 2 + 5 * PI * (x ** 2) + 5 * (PI ** 2) * (x ** 2) - (5 * (PI ** 2) * (x ** 3)) / 3) / (1 + 5 * PI ** 2)

    # integral of (4 + 20 * PI ** 2 + 25 * PI ** 3) / (PI + 5 * PI ** 3) - 2 * SIN(4 / PI) + 2 * SIN(2 * x)
    def intf33(x:float)->float: return x * (4 + 20 * PI ** 2 + 25 * PI ** 3) / (PI + 5 * PI ** 3) - COS(2 * x) - 2 * x * SIN(4 / PI)
    
    if x >= 0 and x <= 1/PI:
        return intf31(x)
    
    elif x <= 2 / PI:
        return IntFunction3(1/PI) - intf32(1/PI) + intf32(x)
    
    elif x <= 8 / PI:
        return IntFunction3(2/PI) - intf33(2/PI) + intf33(x)

#-------------------
#    MAIN FUNCTION
#-------------------
def main()->int:
    # defining the refinement level (how many intervals are going to be used to integrate the function)
    interval = Interval(_a = a, _b = b, _n_refinements = n_divisions, _n_points = npoints)

    # defining the function to be integrated and the exact solution
    func = Function3
    exact = IntFunction3

    printTxt = True

    # defining the numerical method to be used
    match method:
        case 0:
            interval.SetTrapezoidalRule()
            results_file = "Trapezoidal.txt" if printTxt else None

        case 1:
            interval.SetSimpsonOneThirdRule()
            results_file = "Simpson13.txt" if printTxt else None

        case 2:
            interval.SetSimpsonThreeEighthsRule()
            results_file = "Simpson38.txt" if printTxt else None

        case 3:
            interval.SetGaussLegendreRule()
            results_file = "Gauss.txt" if printTxt else None

        case _:
            raise ValueError("Invalid method")

    # integrating the function
    interval.NumericalIntegrate(func, ref_level)
    
    # computing the error
    interval.SetExactSolution(exact, ref_level)
    interval.ComputeError(ref_level)

    # printing the results
    interval.Print(file = results_file, print_sub_intervals = False)

    return 0

if __name__ == "__main__":
    main()