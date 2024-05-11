"""
Class to solve linear systems of equations using the indirect method.

Created by Carlos Puga - 05/11/2024
"""
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np 
from dataclasses import dataclass, field
from List3.SparseMatrix import SparseMatrix

NORM: callable = np.linalg.norm

@dataclass
class IndirectSolver:
    A: SparseMatrix
    rhs: np.ndarray
    ninter: int = 100
    omega: float = 1.0
    method: callable = None

    x_k: np.ndarray = field(init=False, default_factory=list)
    resnorm: list[float] = field(init=False, default_factory=list)

    def Set_number_of_iterations(self, ninter: int):
        """ 
        Sets the number of iterations for the solver
        """
        self.ninter = ninter

    def Set_method(self, method: str):
        """ 
        Sets the method to be used by the solver. Currently available methods are:

            - Jacobi
            - GaussSeidelF
            - GaussSeidelB
            - SSOR
            - Conjugate Gradient
            - Preconditioned Conjugate Gradient
        """
        if method == "Jacobi":
            self.method = self.Jacobi

        elif method == "GaussSeidelF":
            self.method = self.GaussSeidelF

        elif method == "GaussSeidelB":
            self.method = self.GaussSeidelB

        elif method == "SSOR":
            self.method = self.SSOR

        elif method == "Conjugate Gradient":
            self.method = self.ConjugateGradient

        elif method == "Preconditioned Conjugate Gradient":
            self.method = self.PreconditionedConjugateGradient

        else:
            raise ValueError("Invalid method")
        
    def Jacobi(self)->None:
        """
        Solves the linear system of equations using the Jacobi method
        """
        M = self.A.GetDiagonal()

        dx = self.omega * np.divide(self.res, M)

        self.res -= self.A.Multiply(dx)
        self.x_k += dx 
    
    def GaussSeidelF(self)->np.array:
        """
        Solves the linear system of equations using the Gauss-Seidel Forward method
        """
        dx = np.zeros(self.A.size)
        reslocal = self.res.copy()

        dx[0] = self.omega * reslocal[0] / self.A.FindAij(0, 0)

        for i in range(1, self.A.size):
            reslocal[i] -= self.A.InnerProductLowerRows(dx, i)

            dx[i] += self.omega * reslocal[i] / self.A.FindAij(i, i)

        self.res -= self.A.Multiply(dx)
        self.x_k += dx
    
    def GaussSeidelB(self)->np.array:
        raise NotImplementedError
    
    def SSOR(self)->np.array:
        raise NotImplementedError
    
    def ConjugateGradient(self)->tuple[float, list[float]]:
        raise NotImplementedError
    
    def PreconditionedConjugateGradient(self)->tuple[float, list[float]]:
        raise NotImplementedError
    
    def Solve(self)->None:
        """
        Solves the linear system of equations using the selected method
        """
        if not self.method:
            raise ValueError("Method not set")
        
        self.x_k = np.zeros(len(self.rhs))
        self.res = self.rhs - self.A.Multiply(self.x_k)
        self.resnorm = [NORM(self.res)]

        for _ in range(self.ninter):
            self.method()
            self.resnorm.append(NORM(self.res))