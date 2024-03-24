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

    func = lambda x: x

    exact = lambda x: x**2/2
    interval.SetExactSolution(exact)
    
    interval.SetTrapezoidalRule()
    interval.NumericalIntegrate(func, 2)
    
    interval.Print()

if __name__ == "__main__":
    main()