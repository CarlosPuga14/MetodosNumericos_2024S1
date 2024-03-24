"""
The main code to numerically integrate equations. 
This code has been used during the course IC639 - Metodos Numericos para Engenharia,
under the supervision of Prof. Dr. Philippe Devloo and Dr. Giovane Avancini

Created by Carlos Puga - 03/23/2024
"""
from Interval import Interval
import numpy as np

def main():
    interval = Interval(0.0, 1, 0)

    func = lambda x: x + 1

    exact = lambda x: x**2/2 + x
    interval.SetExactSolution(exact)
    
    interval.SetSimpsonThreeEighthsIntegration()
    interval.NumericalIntegrate(func)
    
    interval.Print()

if __name__ == "__main__":
    main()