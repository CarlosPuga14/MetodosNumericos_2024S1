"""
Main file for List 4 - Conjugate Gradient Mehtod

Created by Carlos Puga - 05/09/2024
"""


import sys
sys.path.insert(1, "/Users/CarlosPuga/Latex/MetodosNumericos_2024S1/Lista3/Code")

import numpy as np  

from SparseMatrix import SparseMatrix

def Main()->None:
    matrix = SparseMatrix()
    matrix.ParseFromFile("matrix.dat")

    print(f"{matrix.cols = }")

if __name__ == "__main__":
    Main()