"""
The main code to numerically integrate equations. 
This code has been used during the course IC639 - Metodos Numericos para Engenharia,
under the supervision of Prof. Dr. Philippe Devloo and Dr. Giovane Avancini

Created by Carlos Puga - 03/23/2024
"""
from Interval import Interval
import numpy as np

def main():
    # defining the integration interval
    n_divisions = 2
    npoints = 4
    interval = Interval(_a = 0.0, _b = 1.0, _n_refinements = n_divisions, _n_points = npoints)

    # defining the refinement level (how many intervals are going to be used to integrate the function)
    ref_level = 2

    # defining the function to be integrated and the exact solution
    p = 0
    func = lambda x: x ** p
    exact = lambda x: 1/(p + 1) * x ** (p + 1)

    printTxt = True

    # defining the numerical method to be used
    method: int = 3
    match method:
        case 0:
            interval.SetSimpsonOneThirdRule()

        case 1:
            interval.SetSimpsonThreeEighthsRule()

        case 2:
            interval.SetTrapezoidalRule()

        case 3:
            interval.SetGaussLegendreRule()

        case _:
            raise ValueError("Invalid method")

    # integrating the function
    interval.NumericalIntegrate(func, ref_level)
    
    # computing the error
    interval.SetExactSolution(exact, ref_level)
    interval.ComputeError()

    # printing the results
    results_file = "Gauss.txt" if printTxt else None
    interval.Print(file = results_file, print_sub_intervals = True)

if __name__ == "__main__":
    main()