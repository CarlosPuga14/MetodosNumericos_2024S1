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

def Jacobi(matrix:SparseMatrix, sol:np.array, res:np.array, omega:float = 1.0)->tuple[float, list[float]]:
    """ 
    Evaluates the Jacobi method for a given matrix, right-hand side and initial solution
    """
    M = matrix.GetDiagonal()
    reslocal = res.copy()

    dx = omega * np.divide(reslocal, M)
    
    reslocal -= matrix.Multiply(dx)
    
    sol += dx

    return sol, reslocal

def GaussSeidelF(matrix:SparseMatrix, sol:np.array, res:np.array, omega:float=1.0)->np.array:
    """
    Evaluates the Gauss-Seidel (forward) method for a given matrix, right-hand side and initial solution
    """
    size = len(res)
    dx = np.zeros(size)

    reslocal = res.copy()

    dx[0] = omega * reslocal[0] / matrix.FindAij(0, 0)

    for i in range(1, size):
        reslocal[i] -= matrix.InnerProductLowerRows(dx, i)

        dx[i] += omega * reslocal[i] / matrix.FindAij(i, i)

    reslocal = res - matrix.Multiply(dx)
    sol += dx

    return sol, reslocal

def GaussSeidelB(matrix:SparseMatrix, sol:np.array, res:np.array, omega:float=1.0)->np.array:
    """
    Evaluates the Gauss-Seidel (backward) method for a given matrix, right-hand side and initial solution
    """
    size = len(res)
    dx = np.zeros(size)

    reslocal = res.copy()

    dx[size-1] = omega * reslocal[size-1] / matrix.FindAij(size-1, size-1)

    for i in range(size-2, -1, -1):
        reslocal[i] -= matrix.InnerProductUpperRows(dx, i)
        dx[i] += omega * reslocal[i] / matrix.FindAij(i, i)

    reslocal = res - matrix.Multiply(dx)
    sol += dx

    return sol, reslocal

def SSOR(matrix:SparseMatrix, sol:np.array, res:np.array, omega:float=1.0)->np.array:
    """
    Evaluates the Symmetric Successive Over-Relaxation method for a given matrix, right-hand side and initial solution
    """
    dx1, res1 = GaussSeidelF(matrix, sol, res, omega)
    dx2, res2 = GaussSeidelB(matrix, dx1, res1, omega)

    return dx2, res2

def ConjugateGradient(matrix:SparseMatrix, rhs:np.array, solini:np.array, niter:int)->tuple[float, list[float]]:
    """
    Evaluates the Conjugate Gradient method for a given matrix, right-hand side and initial solution
    """
    x_k = solini.copy()
    res = rhs - matrix.Multiply(x_k)
    p_k = res.copy()

    res_k = res.copy()

    resnorm = [np.linalg.norm(res_k)]

    for _ in range(niter):
        alpha_k = np.dot(res_k.T, res_k) / np.dot(matrix.Multiply(p_k.T), p_k)
        x_k += alpha_k * p_k

        res = res_k - alpha_k * matrix.Multiply(p_k)

        beta_k = np.dot(res.T, res) / np.dot(res_k.T, res_k)

        p_k = res + beta_k * p_k

        res_k = res.copy()

        resnorm.append(np.linalg.norm(res))

    return x_k, resnorm

def PreconditionedConjugateGradient(matrix:SparseMatrix, rhs:np.array, sol:np.array, niter:int, method:callable)->tuple[float, list[float]]:
    """
    Evaluates the Preconditioned Conjugate Gradient method for a given matrix, right-hand side and initial solution
    """
    x_k = sol.copy()
    res = rhs - matrix.Multiply(x_k)
    
    z, _ = method(matrix, np.zeros(len(res)), res)

    p_k = z.copy()
    z_k = z.copy()
    res_k = res.copy()

    resnorm = [np.linalg.norm(res)]

    for _ in range(niter):
        alpha_k = np.dot(res_k.T, z) / np.dot(matrix.Multiply(p_k.T), p_k)

        x_k += alpha_k * p_k
        res = res_k - alpha_k * matrix.Multiply(p_k)

        z, _ = method(matrix, np.zeros(len(res)), res)

        beta = np.dot(res.T, z) / np.dot(res_k.T, z_k)

        p_k = z + beta * p_k
        z_k = z.copy()
        res_k = res.copy()

        resnorm.append(np.linalg.norm(res))

    return x_k, resnorm

def IndirectSolver(matrix:SparseMatrix, rhs:np.array, solini:np.array, niter:int, method:callable, omega:float = 1)->tuple[float, list[float]]:
    sol = solini.copy() 
    res = rhs - matrix.Multiply(sol)

    resnorm = [np.linalg.norm(res)]

    for _ in range(niter):
        sol, res = method(matrix, sol, res, omega)
        resnorm.append(np.linalg.norm(res))

    return sol, resnorm

def Main()->None:
    matrix_file = "List4/matrix_test.dat"
    rhs_file = "List4/rhs_test.dat"

    matrix = SparseMatrix()
    matrix.ParseFromFile(matrix_file)

    rhs = ParseVector(rhs_file)
    sol = np.zeros(len(rhs))

    niter = 100
    solCG, resnormCG = ConjugateGradient(matrix, rhs, sol, niter)
    solCGJ, resnormCGJ = PreconditionedConjugateGradient(matrix, rhs, sol, niter, Jacobi)
    solCGJ, resnormCGSSOR = PreconditionedConjugateGradient(matrix, rhs, sol, niter, SSOR)

    plt.figure()
    plt.semilogy(range(niter+1), resnormCG)
    plt.semilogy(range(niter+1), resnormCGJ)
    plt.semilogy(range(niter+1), resnormCGSSOR)
    plt.legend(["CG", "CG-Jacobi", "CG-SSOR"])
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Residual Norm vs Iteration")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    Main()