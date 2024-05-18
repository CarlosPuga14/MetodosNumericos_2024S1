"""
Main file for List 4 - Conjugate Gradient Mehtod

Created by Carlos Puga - 05/09/2024
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np  
from List3.SparseMatrix import SparseMatrix
from IndirectSolver import IndirectSolver

def ParseVector(file: str)->np.array:
    with open(file, "r") as f:
        vector = np.array([float(x) for x in f.read().split(",")])      
    
    return vector

def Main()->None:
    matrix_file = "List4/matrix.dat"
    rhs_file = "List4/rhs.dat"
    results_file = "List4/Results.py"
    write_results = False

    niter = 500

    print("------ Reading matrix and right-hand side ------")
    matrix = SparseMatrix()
    matrix.ParseFromFile(matrix_file)

    rhs = ParseVector(rhs_file)

    solver = IndirectSolver(matrix, rhs, niter = niter)

    print("------ CONJUGATE GRADIENT ------")
    solver.Set_method("ConjugateGradient")
    solver.Solve()
    resnormCG = solver.resnorm

    print("------ CG JACOBI ------")
    solver.Set_preconditioner("Jacobi")
    solver.Solve()
    resnormCGJ = solver.resnorm

    print("------ CG SSOR ------")
    solver.Set_preconditioner("SSOR")
    solver.Solve()
    resnormCGSSOR = solver.resnorm

    if write_results:
        print("------ Saving results ------")
        with open(results_file, 'w') as f:
            print(f"resnormCG = {str(resnormCG)}", file=f)
            print(f"resnormCGJ = {str(resnormCGJ)}", file=f)
            print(f"resnormCGSSOR = {str(resnormCGSSOR)}", file=f)

    print("------ Simulation ended withou errors ------")

if __name__ == "__main__":
    Main()