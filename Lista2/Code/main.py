"""
The main code to numerically integrate equations. 
This code has been used during the course IC639 - Metodos Numericos para Engenharia,
under the supervision of Prof. Dr. Philippe Devloo and Dr. Giovane Avancini

Created by Carlos Puga - 03/23/2024
"""
from Interval import Interval
import numpy as np

def main():
    interval = Interval(0.0, 1, 2)

    func = lambda x: np.sin(x)

    ref_level = 1

    exact = lambda x: -np.cos(x)
    interval.SetExactSolution(exact, ref_level)
    
    interval.SetTrapezoidalRule()
    interval.NumericalIntegrate(func, ref_level)
    
    interval.ComputeError()

    interval.Print()

if __name__ == "__main__":
    main()