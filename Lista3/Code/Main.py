"""
Main file for List 3 - Matrices Decomposition
"""
from Matrix import Matrix
import numpy as np

INVERSE: callable = np.linalg.inv

def Main()->None:
    myMatrix = Matrix([[1.0, 3.0, 5.0], [7.0, 9.0, 2.0], [4.0, 6.0, 8.0]], pivoting=False)

    myMatrix.SetDecomposition("LU")
    myMatrix.Decompose()

    print(INVERSE(myMatrix.L @ myMatrix.U) - INVERSE(myMatrix.A))

if __name__ == "__main__":
    Main()