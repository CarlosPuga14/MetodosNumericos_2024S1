"""
The main code to numerically integrate equations. 
This code has been used during the course IC639 - Metodos Numericos para Engenharia,
under the supervision of Prof. Dr. Philippe Devloo and Dr. Giovane Avancini

Created by Carlos Puga - 03/23/2024
"""
from Interval import Interval
from scipy import special
import numpy as np

epot:int = 4
method: int = 3

def Function1(x:float)->float:
    return np.exp(-(x - 1)**2 /10**(-epot))

def IntFunction1(x:float)->float:
    const = (np.sqrt(10)**epot)
    return  np.sqrt(np.pi) * (special.erf(const * (-1 + x)))/(2*const)


def main():
    # defining the integration interval
    n_divisions = 4
    npoints = 4
    a = 0.0
    b = 2.0

    # defining the refinement level (how many intervals are going to be used to integrate the function)
    ref_level = 4

    interval = Interval(_a = a, _b = b, _n_refinements = n_divisions, _n_points = npoints)

    # defining the function to be integrated and the exact solution
    func = Function1
    exact = IntFunction1

    printTxt = True

    # defining the numerical method to be used
    match method:
        case 0:
            interval.SetSimpsonOneThirdRule()
            results_file = "Simpson13.txt" if printTxt else None

        case 1:
            interval.SetSimpsonThreeEighthsRule()
            results_file = "Simpson38.txt" if printTxt else None

        case 2:
            interval.SetTrapezoidalRule()
            results_file = "Trapezoidal.txt" if printTxt else None

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
    interval.Print(file = results_file, print_sub_intervals = True)

    return 0

if __name__ == "__main__":
    main()
