"""
The main code to numerically integrate equations. 
This code has been used during the course IC639 - Metodos Numericos para Engenharia,
under the supervision of Prof. Dr. Philippe Devloo and Dr. Giovane Avancini

Created by Carlos Puga - 03/23/2024
"""
from Interval import Interval

def main():
    interval = Interval(0.0, 2.0, 2)
    interval.SetSimpsonOneThirdIntegration()

    func = lambda x: 1
    exact = lambda x: x

    interval.NumericalIntegrate(func, 2)
    interval.ExactIntegrate(exact)

    interval.Print()

if __name__ == "__main__":
    main()