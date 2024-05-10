"""
Main file for List 4 - Conjugate Gradient Mehtod

Created by Carlos Puga - 05/09/2024
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np  
from matplotlib import pyplot as plt
from List3.SparseMatrix import SparseMatrix

def ParseVector(file: str)->np.ndarray:
    with open(file, "r") as f:
        vector = np.array([float(x) for x in f.read().split(",")])      
    
    return vector

def Jacobi(matrix:SparseMatrix, rhs:np.array, solini:np.array, niter:int, omega:float)->tuple[float, list[float]]:
    """ 
    Evaluates the Jacobi method for a given matrix, right-hand side and initial solution
    """
    sol = solini.copy() 
    res = rhs - matrix.Multiply(sol)
    resnorm = [np.linalg.norm(res)]

    M = matrix.GetDiagonal()
    for _ in range(niter):
        reslocal = res 

        dx = omega * np.divide(reslocal, M)

        reslocal -= matrix.Multiply(dx)
        sol += dx

        resnorm.append(np.linalg.norm(reslocal))

    return sol, resnorm

def Main()->None:
    matrix_file = "List4/matrix_test.dat"
    rhs_file = "List4/rhs_test.dat"

    matrix = SparseMatrix()
    matrix.ParseFromFile(matrix_file)

    rhs = ParseVector(rhs_file)
    sol = np.zeros(len(rhs))

    niter = 10
    sol, resnorm = Jacobi(matrix, rhs, sol, niter, 1.0)

    plt.figure()
    plt.semilogy(range(niter+1), resnorm)
    plt.show()



if __name__ == "__main__":
    Main()