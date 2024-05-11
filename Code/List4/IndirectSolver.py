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
INNER: callable = np.dot
ZEROS: callable = np.zeros

@dataclass
class IndirectSolver:
    A: SparseMatrix
    rhs: np.ndarray
    niter: int
    omega: float = 1.0
    method: callable = None

    resnorm: list[float] = field(init=False, default_factory=list)
    
    p_k: np.array = field(init=False, default=list)
    res_k: np.array = field(init=False, default=list)
    
    preconditioner: callable = field(init=False, default=None)
    z: np.array = field(init=False, default=None)
    z_k: np.array = field(init=False, default=None)

    def Set_number_of_iterations(self, ninter: int):
        """ 
        Sets the number of iterations for the solver
        """
        self.niter = ninter

    def Set_omega(self, omega: float):
        """ 
        Sets the relaxation factor for the solver
        """
        self.omega = omega

    def Set_method(self, method: str):
        """ 
        Sets the method to be used by the solver. Currently available methods are:

            - Jacobi
            - GaussSeidelF
            - GaussSeidelB
            - SSOR
            - Conjugate Gradient
        """
        if method == "Jacobi":
            self.method = self.Jacobi

        elif method == "GaussSeidelF":
            self.method = self.GaussSeidelF

        elif method == "GaussSeidelB":
            self.method = self.GaussSeidelB

        elif method == "SSOR":
            self.method = self.SSOR

        elif method == "ConjugateGradient":
            self.method = self.ConjugateGradient

        else:
            raise ValueError("Invalid method")
        
    def Set_preconditioner(self, preconditioner: str):
        """ 
        Sets the preconditioner for the solver
        """
        if preconditioner == "Jacobi":
            self.preconditioner = self.Jacobi

        elif preconditioner == "SSOR":
            self.preconditioner = self.SSOR

        else:
            raise ValueError("Invalid preconditioner")
        
    def Jacobi(self, sol:np.array, res:np.array)->None:
        """
        Solves the linear system of equations using the Jacobi method
        """
        M = self.A.GetDiagonal()
        reslocal = res.copy()

        dx = self.omega * np.divide(res, M)

        reslocal -= self.A.Multiply(dx)

        return sol + dx, reslocal
    
    def GaussSeidelF(self, sol:np.array, res:np.array)->np.array:
        """
        Solves the linear system of equations using the Gauss-Seidel Forward method
        """
        dx = ZEROS(self.A.size)
        reslocal = res.copy()

        dx[0] = self.omega * reslocal[0] / self.A.FindAij(0, 0)

        for i in range(1, self.A.size):
            reslocal[i] -= self.A.InnerProductLowerRows(dx, i)
            dx[i] += self.omega * reslocal[i] / self.A.FindAij(i, i)

        reslocal = res - self.A.Multiply(dx)

        return sol + dx, reslocal
    
    def GaussSeidelB(self, sol:np.array, res:np.array)->np.array:
        """ 
        Solves the linear system of equations using the Gauss-Seidel Backward method
        """
        dx = ZEROS(self.A.size)
        reslocal = res.copy()

        dx[self.A.size-1] = self.omega * reslocal[self.A.size-1] / self.A.FindAij(self.A.size-1, self.A.size-1)

        for i in range(self.A.size-2, -1, -1):
            reslocal[i] -= self.A.InnerProductUpperRows(dx, i)
            dx[i] += self.omega * reslocal[i] / self.A.FindAij(i, i)

        reslocal = res - self.A.Multiply(dx)

        return sol + dx, reslocal
    
    def SSOR(self, sol:np.array, res:np.array)->np.array:
        """ 
        Solves the linear system of equations using the Symmetric Successive Over-Relaxation method
        """
        sol, res = self.GaussSeidelF(sol, res)
        sol, res = self.GaussSeidelB(sol, res)

        return sol, res 

    def ConjugateGradient(self, sol:np.array, res:np.array)->tuple[float, list[float]]:
        """
        Solves the linear system of equations using the Conjugate Gradient method
        """
        alpha_k = INNER(res.T, res) / INNER(self.A.Multiply(self.p_k.T), self.p_k) if not self.preconditioner else INNER(self.res_k.T, self.z) / INNER(self.A.Multiply(self.p_k.T), self.p_k)

        sol += alpha_k * self.p_k
        res = self.res_k - alpha_k * self.A.Multiply(self.p_k)

        if not self.preconditioner:
            beta_k = INNER(res.T, res) / INNER(self.res_k.T, self.res_k)

            self.p_k = res + beta_k * self.p_k
        
        else:
            self.z, _ = self.preconditioner(ZEROS(self.A.size), res)
            beta_k = INNER(res.T, self.z) / INNER(self.res_k.T, self.z_k)

            self.z_k = self.z.copy()

            self.p_k = self.z + beta_k * self.p_k

        self.res_k = res.copy()

        return sol, res
    
    def Solve(self)->None:
        """
        Solves the linear system of equations using the selected method
        """
        if not self.method:
            raise ValueError("Method not set")
        
        sol = ZEROS(len(self.rhs))
        res = self.rhs - self.A.Multiply(sol)

        if self.method == self.ConjugateGradient:
            if not self.preconditioner:
                self.p_k = res.copy()
            
            else:
                self.z, _ = self.preconditioner(ZEROS(self.A.size), res)
                self.p_k = self.z.copy()
                self.z_k = self.z.copy()

            self.res_k = res.copy()
            
        self.resnorm = [NORM(res)]
        for i in range(self.niter):
            print(f"Method: {self.method} - Iteration {i}")
            sol, res = self.method(sol, res)
            self.resnorm.append(NORM(res))