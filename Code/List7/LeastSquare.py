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
        self.x = np.log(self.x)

        self.PolynomialApproximation()

        self.y = np.exp(self.y)
        self.x = np.exp(self.x)

    def NonLinearApproximation(self)->None:
        """"
        Employs the Newton-Raphson method to approximate the solution
        """
        def dEda(a, b, xi, yi):
            if xi == 0:
                return 2 * (yi - b * xi ** a) * (-b  * xi ** a)
            
            return 2 * (yi - b * xi ** a) * (-b * np.log(xi) * xi ** a)
        
        def dEdb(a, b, xi, yi):
            return 2 * (yi - b * xi ** a) * (-xi ** a)
        
        def grad_dEda(a, b, xi, yi):
            if xi == 0:
                return [2 * b * xi ** a * (2 * b * xi ** a - yi) ** 2, 2 * xi ** a * (2 * b * xi ** a - yi)]
            
            return [2 * b * xi ** a * (2 * b * xi ** a - yi) * np.log(xi) ** 2, 2 * xi ** a * (2 * b * xi ** a - yi) * np.log(xi)]
        
        def grad_dEdb(a, b, xi, yi):
            if xi == 0:
                return [2 * xi ** a * (2 * b * xi ** a - yi) ** 2, 2 * xi ** (2 * a)]
            
            return [2 * xi ** a * (2 * b * xi ** a - yi) * np.log(xi), 2 * xi ** (2 * a)]

        xval = [1, 1]

        n_iter = 100

        for _ in range(n_iter):
            equations = np.zeros(2)
            grad_eq = np.zeros((2, 2))

            for x, y in zip(self.x, self.y):
                equations += np.array([dEda(*xval, x, y), dEdb(*xval, x, y)])
                grad_eq += np.array([grad_dEda(*xval, x, y), grad_dEdb(*xval, x, y)])

            x_next = xval - np.linalg.solve(grad_eq, equations) 

            residual = np.zeros(2)
            for x, y in zip(self.x, self.y):
                residual += np.array([dEda(*x_next, x, y), dEdb(*x_next, x, y)])

            residual = np.linalg.norm(residual)

            xval = x_next

            if residual < 1e-14:
                break

        self.alpha = xval

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
                self.approx_solution[i] = self.alpha[0] * self.x[i] ** self.alpha[1]

            elif self.approximation_type == "NonLinear":
                self.approx_solution[i] = self.alpha[1] * self.x[i] ** self.alpha[0]

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
            "Logarithmic": self.LogarithmicApproximation,
            "NonLinear": self.NonLinearApproximation
        }

        method[self.approximation_type]()

        if self.approximation_type in ["Polynomial", "Logarithmic"]:
            self.Solver()

        self.CalcApproxSolution()

        self.Error()

        return