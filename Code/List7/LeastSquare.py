from dataclasses import dataclass, field
import numpy as np

@dataclass
class LeastSquare:
    """
    Least Square class
    """
    x: list
    y: list 
    approximation_type: str
    
    order: int = 1

    K: list = field(init=False, default_factory=list)
    F: list = field(init=False, default_factory=list)
    alpha: list = field(init=False, default_factory=list)

    approx_solution: list = field(init=False, default_factory=list)

    errors: list = field(init=False, default_factory=list)
    total_error: float = field(init=False, default=0.0)

    def SetOrder(self, order:int)->None:
        """
        Sets the order of the polynomial
        """
        self.order = order

    def SetMethod(self, method:str)->None:
        """
        Sets the approximation method. Avaliable methods are:
        - Polynomial
        - Logarithmic
        - Exponential 
        """
        self.approximation_type = method

    def PolynomialApproximation(self)->None:
        """
        Approximates the set of points with a polynomial of degree n
        """
        n = self.order + 1
        m = len(self.x)

        self.K = np.zeros((n, n))
        self.F = np.zeros(n)

        for i in range(n):
            self.F[i] = sum([self.y[k] * self.x[k] ** i for k in range(m)])

            for j in range(n):
                self.K[i, j] = sum([self.x[k] ** (i + j) for k in range(m)])

    def LogarithmicApproximation(self)->None:
        """
        Approximates the set of points with a logarithmic function
        """
        self.y = np.log(self.y)

        self.PolynomialApproximation()

        self.y = np.exp(self.y)

    def CalcApproxSolution(self)->None:
        """
        Calculates the approximation solution
        """
        m = len(self.x)
        n = self.order + 1

        self.approx_solution = np.zeros(m)

        for i in range(m):
            if self.approximation_type == "Polynomial":
                self.approx_solution[i] = sum([self.alpha[j] * self.x[i] ** j for j in range(n)])

            elif self.approximation_type == "Logarithmic":
                self.approx_solution[i] = self.alpha[0] * np.exp(self.alpha[1] * self.x[i])

            else: 
                raise ValueError("Invalid approximation type")
            
    def Solver(self)->None:
        """
        Solves the system of equations
        """
        self.alpha = np.linalg.solve(self.K, self.F)

        if self.approximation_type == "Logarithmic":
            self.alpha[0] = np.exp(self.alpha[0])

    def Error(self)->None:
        """
        Calculates the error of the approximation
        """
        m = len(self.x)

        self.errors = np.zeros(m)

        for i in range(m):
            self.errors[i] = (self.y[i] - self.approx_solution[i]) ** 2

        self.total_error = sum(self.errors)

    def Run(self)->None:
        """
        Runs the approximation
        """
        method = {
            "Polynomial": self.PolynomialApproximation,
            "Logarithmic": self.LogarithmicApproximation
        }

        method[self.approximation_type]()

        self.Solver()

        self.CalcApproxSolution()

        self.Error()

        print(f"{self.K=}")
        print(f"{self.F=}")
        print(f"{self.alpha=}")
        print(f"{self.approx_solution=}")
        print(f"{self.errors=}")
        print(f"{self.total_error=}")

        return