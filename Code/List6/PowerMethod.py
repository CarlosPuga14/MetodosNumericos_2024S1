from dataclasses import dataclass, field
import numpy as np

# ----- Alias -----
ARRAY: callable = np.array
NORM: callable = np.linalg.norm
OUTER: callable = np.outer

@dataclass
class PowerMethod:
    A: np.array
    
    number_of_iterations: int = 20
    precision: float = 1e-8

    size: int = field(init=False)
    B: np.array = field(init=False)

    eigenval: list[float] = field(init=False, default_factory=list)
    eigenvec: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.A = ARRAY(self.A)
        self.B = self.A.copy()
        self.size = len(self.A)

    def SetNumberOfIterations(self, number_of_iterations:int)->None:
        """
        Set the number of iterations for the Power Method algorithm.
        """
        self.number_of_iterations = number_of_iterations

    def SetPrecision(self, precision:float)->None:
        """
        Set the precision for the Power Method algorithm.
        """
        self.precision = precision

    def GetEigenvector(self)->list:
        """
        Get the eigenvector.
        """
        return self.eigenvec
    
    def GetEigenvalue(self)->list:
        """
        Get the eigenvalue.
        """
        return self.eigenval
    
    def GetEigensystem(self)->tuple:
        """
        Get the eigenvalue and eigenvector.
        """
        return self.eigenval, self.eigenvec

    def FindEigenSystem(self)->tuple:
        """
        Find the proeminent eigenvalue and repective eigenvector.
        """
        v0 = np.ones(self.size)

        lambda_i = [NORM(v0)]
        v_i = [v0/lambda_i[0]]

        for i in range(1, self.number_of_iterations):
            previous = self.B @ v_i[i-1]

            lambda_i.append(NORM(previous))
            v_i.append(previous/lambda_i[i])

            sign = previous @ v_i[i-1]
            if sign < 0:
                lambda_i[i] *= -1

            if abs(lambda_i[i] - lambda_i[i-1]) < self.precision:
                break

        return lambda_i, v_i

    def UpdateMatrix(self, eigenval, eigenvec)->None:
        """
        Update the matrix B.
        """
        self.B = self.B - eigenval * OUTER(eigenvec, eigenvec)

    def Run(self)->None:
        """
        Run the simulation of the Power Method algorithm.
        """
        for _ in range(self.size):
            lambda_, v = self.FindEigenSystem()

            self.eigenval.append(lambda_)
            self.eigenvec.append(v)

            self.UpdateMatrix(lambda_[-1], v[-1])

    def WriteResults(self, file:str)->None:
        """
        Write the results in a file.
        """
        with open(file, "w") as f:
            for i, eigensystem in enumerate(zip(self.eigenval, self.eigenvec), 1):
                f.write(f"{'-'*20} {i} {'-'*20}\n")
                eigenval, eigenvec = eigensystem
                for j, (val, vec) in enumerate(zip(eigenval, eigenvec),1):
                    f.write(f"{j}. Eigenvalue: {val}\n")
                    f.write(f"   Eigenvector: {vec}\n")
                    f.write("\n")
