""" 
Class for solving nonlinear systems of equations
Available methods: 
    - Newton-Raphson
    - Broyden

Created by Carlos Puga - 05/29/2024
"""
from dataclasses import dataclass, field
import numpy as np

# ---- Alias for numpy functions ----
ARRAY: callable = np.array
LINSOLVE: callable = np.linalg.solve
NORM: callable = np.linalg.norm
OUTER: callable = np.outer
LOG: callable = np.log

@dataclass
class NonLinearSolver:
    equations: list[callable] = field(default_factory=list)
    gradients: list[callable] = field(default_factory=list)
    x0: list[float] = field(default_factory=list)

    max_iter: int = 50
    tolerance: float = 1e-15

    x_list: list[list] = field(init=False, default_factory=list)
    diff: list[float] = field(init=False, default_factory=list)
    diff_log: list[float] = field(init=False, default_factory=list)
    residual: list[float] = field(init=False, default_factory=list)

    exact_solution: list[float] = field(default_factory=list)   

    def __post_init__(self):
        self.equations = ARRAY(self.equations)
        self.gradients = ARRAY(self.gradients)
        self.x0 = ARRAY(self.x0)

        self.x_list.append(self.x0)

    def SetMaxIteration(self, iter:int) -> None:
        """     
        Set the maximum number of iterations
        """
        self.max_iter = iter

    def SetMethod(self, method:str)->None:
        """
        Set the method to solve the system of equations
        """
        str_to_method = {
            "Newton": self.Newton,
            "Broyden": self.Broyden
        }

        self.method = str_to_method[method]

    def SetTolerance(self, tol:float) -> None:
        """
        Set the tolerance for the solution
        """
        self.tolerance = tol

    def SetExactSolution(self, sol:list[float]) -> None:
        """
        Set the exact solution for the system of equations
        """
        self.exact_solution = ARRAY(sol)

    def GetError(self) -> list[float]:
        """
        Get the error and convergence rate
        """
        return self.diff, self.diff_log, self.residual
    
    def SaveResidual(self, xval) -> None:
        """
        Save the residual of the system of equations
        """
        res = ARRAY(list(map(lambda eq: eq(*xval), self.equations)))
        self.residual.append(NORM(res))


    def Newton(self) -> None:
        """
        Solve the system of equations using Newton-Raphson method
        """
        xval = self.x0

        if any(self.exact_solution):
                self.diff.append(NORM(xval - self.exact_solution))

        self.SaveResidual(xval)
                    
        for _ in range(self.max_iter):
            G = ARRAY(list(map(lambda eq: eq(*xval), self.equations)))
            Grad_G = ARRAY(list(map(lambda grad: grad(*xval), self.gradients)))

            x_next = xval - LINSOLVE(Grad_G, G)

            self.x_list.append(x_next)
            self.SaveResidual(x_next)

            if any(self.exact_solution):
                self.diff.append(NORM(x_next - self.exact_solution))

            if NORM(self.residual[-1]) < self.tolerance:
                break

            xval = x_next

    def Broyden(self) -> None:
        """
        Solve the system of equations using Broyden method
        """
        v0 = self.x0.copy()
        self.SaveResidual(v0)

        G0 = ARRAY(list(map(lambda eq: eq(*v0), self.equations)))
        Grad_G0 = ARRAY(list(map(lambda grad: grad(*v0), self.gradients)))

        v1 = v0 - LINSOLVE(Grad_G0, G0)
    
        self.x_list.append(v1)
        self.SaveResidual(v1)

        G1 = ARRAY(list(map(lambda eq: eq(*v1), self.equations)))
        
        del_x = v1 - v0 

        if any(self.exact_solution):
            self.diff.append(NORM(v1 - self.exact_solution))

        del_G = G1 - G0

        for _ in range(self.max_iter-1):
            grad_G1 = Grad_G0 + OUTER(del_G - Grad_G0 @ del_x, del_x) / (del_x @ del_x)

            Grad_G0 = grad_G1

            xnext = v1 - LINSOLVE(grad_G1, G1)

            v0 = v1 
            v1 = xnext

            self.SaveResidual(v1)

            G0 = G1
            G1 = ARRAY(list(map(lambda eq: eq(*v1), self.equations)))

            del_x = v1 - v0
            del_G = G1 - G0

            self.x_list.append(v1)

            if any(self.exact_solution):
                self.diff.append(NORM(v1 - self.exact_solution))

            if NORM(self.residual[-1]) < self.tolerance:
                break

    def Solve(self) -> None:
        """
        Solve the system of equations
        """
        self.method()

        for i in range(1, len(self.diff)):
            self.diff_log.append(LOG(self.diff[i]) / LOG(self.diff[i-1]))

    def WriteSolution(self, file)->None:
        """
        Write the solution to a file
        """
        def printVector(v: list[float]) -> str:
            return " ".join(map(str, v))
        
        def printMatrix(M: list[list[float]]) -> str:
            return "\n".join(map(printVector, M))

        with open(file, "w") as f:
            f.write(f"Solution: \n")
            f.write(printMatrix(self.x_list))
            f.write("\n\n")

            f.write(f"Error: \n")
            f.write(printVector(self.diff))
            f.write("\n\n")

            f.write(f"Convergence rate: ")
            f.write(printVector(self.diff_log))

            f.write(f"Residual: \n")
            f.write(printVector(self.residual))