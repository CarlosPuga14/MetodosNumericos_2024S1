"""
The matrix class is implemented here. 

Created by Carlos Puga - 04/05/2024
"""
from dataclasses import dataclass, field
import numpy as np

# -----------------
# ---- ALIASES ----
# -----------------
ZEROS: callable = np.zeros_like
OUTER: callable = np.outer
ARRAY: callable = np.array
IDENTITY: callable = np.identity
SQRT: callable = np.sqrt
INVERSE: callable = np.linalg.inv

# -----------------
# ----- CLASS -----
# -----------------
@dataclass
class Matrix:
    """
    Class to represent a matrix and its decomposition
    """
    A: np.array # Matrix
    pivoting: bool = False # Pivoting flag
    decomposition_type: str = None # Decomposition type
    size: int = field(init=False) # Matrix size

    A_Decomposed: np.array = field(init=False) # Decomposed matrix
    L: np.array = field(init=False) # Lower matrix
    U: np.array = field(init=False) # Upper matrix
    D: np.array = field(init=False) # Diagonal matrix

    inverseA: np.array = field(init=False) # Inverse of A
    decomposd_inverse: np.array = field(init=False) # Inverse of A by decomposition

    def __post_init__(self):
        self.size = len(self.A)

        self.A = ARRAY(self.A)
        self.A_Decomposed = self.A.copy()
        self.inverseA = INVERSE(self.A)

        self.L = IDENTITY(self.size)
        self.U = IDENTITY(self.size)
        self.D = IDENTITY(self.size)
        self.decomposd_inverse = ZEROS(self.A)

    # -----------------
    # ---- METHODS ----
    # -----------------
    def SetDecompositionMethod(self, decomposition_type: str)->None:
        """
        Set the decomposition type
        """        
        if not decomposition_type in ["LU", "LDU", "LLt", "LDLt"]:
            raise Exception(f"The '{decomposition_type}' decomposition is not valid. Please choose one of the following: LU, LDU, LLt, LDLt")
        
        self.decomposition_type = decomposition_type

    def PivotMatrix(self)->None:
        raise NotImplementedError("Pivoting not implemented yet. Please implement me!")
    
    def RankOneUpdate(self)->None:
        """
        Perform a rank 1 update
        """
        for i in range(self.size):
            matii_correction = 1.0

            if self.decomposition_type in ["LDU", "LDLt"]:
                matii_correction = self.A_Decomposed[i, i]
                self.A_Decomposed[i, i+1::] /= self.A_Decomposed[i, i]

            elif self.decomposition_type == "LLt":
                self.A_Decomposed[i, i] /= SQRT(self.A_Decomposed[i, i])
                self.A_Decomposed[i, i+1::] /= self.A_Decomposed[i, i]

            self.A_Decomposed[i+1::, i] /= self.A_Decomposed[i, i]
            self.A_Decomposed[i+1::, i+1::] -= OUTER(self.A_Decomposed[i+1::, i], self.A_Decomposed[i, i+1::]) * matii_correction

    def Fill_L(self)->None:
        """
        Fill the lower matrix
        """
        diagonal_aux1 = 0 if self.decomposition_type == "LLt" else 1 # row used to start filling the matrix
        diagonal_aux2 = 1 if self.decomposition_type == "LLt" else 0 # column used to start filling the matrix

        for i in range(diagonal_aux1, self.size):
            for j in range(i + diagonal_aux2):
                self.L[i, j] = self.A_Decomposed[i, j]

    def Fill_U(self)->None:
        """
        Fill the upper matrix
        """
        diagonal_aux1 = 0 if self.decomposition_type == "LU" else 1 # row used to start filling the matrix
        diagonal_aux2 = 1 if self.decomposition_type == "LU" else 0 # column used to start filling the matrix

        for j in range(diagonal_aux1, self.size):
            for i in range(j + diagonal_aux2):
                self.U[i, j] = self.A_Decomposed[i, j]

    def Fill_D(self)->None:
        """
        Fill the diagonal matrix
        """
        for i in range(self.size):
            self.D[i, i] = self.A_Decomposed[i, i]

    def LU_Decomposition(self)->None:
        """
        Decompose the matrix using LU decomposition
        """
        if self.pivoting:
            self.PivotMatrix()
        
        self.RankOneUpdate()
        self.Fill_L()
        self.Fill_U()

    def LDU_Decomposition(self)->None:
        """
        Decompose the matrix using LDU decomposition
        """
        self.RankOneUpdate()
        self.Fill_L()
        self.Fill_D()
        self.Fill_U()
    
    def LLt_Decomposition(self)->None:
        """
        Decompose the matrix using LLt (Cholesky) decomposition
        """
        self.RankOneUpdate()
        self.Fill_L()
        self.U = self.L.T
    
    def LDLt_Decomposition(self)->None:
        """
        Decompose the matrix using LDLt decomposition
        """
        self.RankOneUpdate()
        self.Fill_L()
        self.Fill_D()
        self.U = self.L.T

    def Check_Symmetry(self)->bool:
        """
        Check if the matrix is symmetric
        """
        return np.allclose(self.A, self.A.T)
    
    def Check_PositiveDefinite(self)->bool:
        """
        Check if the matrix is positive definite
        """
        return np.all(np.linalg.eigvals(self.A) > 0)

    def Decompose(self)->None:
        """
        Decompose the matrix using the specified decomposition
        """
        if (self.decomposition_type in ["LDU", "LLt", "LDLt"]) and (self.pivoting):
            raise Exception(f"Pivoting not defined for the '{self.decomposition_type}' decomposition. Please try again with 'LU' decomposition.")

        if self.decomposition_type == "LU":
            self.LU_Decomposition()

        elif self.decomposition_type == "LDU":
            self.LDU_Decomposition()

        elif self.decomposition_type == "LLt":
            if (not self.Check_Symmetry()) and (not self.Check_PositiveDefinite()):
                raise Exception("The matrix is not symmetric positive definite. Please choose another decomposition type.")
            
            self.LLt_Decomposition()

        elif self.decomposition_type == "LDLt":
            if not self.Check_Symmetry():
                raise Exception("The matrix is not symmetric positive definite. Please choose another decomposition type.")
            
            self.LDLt_Decomposition()

        else:
            text = f"The '{self.decomposition_type}' decomposition is not valid. Please choose one of the following: "
            text += "LU, LDU, LLt, LDLt"

            raise Exception(text)
        
    def FindInverse(self)->None:
        """
        Find the inverse of the matrix using the decomposition
        """
        if self.decomposition_type in ["LU", "LLt"]:
            self.decomposd_inverse = INVERSE(self.L @ self.U)
        
        elif self.decomposition_type in ["LDU", "LDLt"]:
            self.decomposd_inverse = INVERSE(self.L @ self.D @ self.U) 
        
        else:
            raise Exception(f"The '{self.decomposition_type}' decomposition is not valid. Please choose one of the following: LU, LDU, LLt, LDLt")
    
    def Print(self, file)->None:
        """
        Print the matrix and its decomposition
        """
        with open(file, "w") as f:
            print(f"********** {self.decomposition_type} Decomposition **********\n", file=f)
            print(f"Matrix A: \n{self.A}\n", file=f)
            print(f"Matrix A Decomposed: \n{self.A_Decomposed}\n", file=f)

            print(f"Matrix L: \n{self.L}\n", file=f)
            print(f"Matrix U: \n{self.U}\n", file=f)
            print(f"Matrix D: \n{self.D}\n", file=f)

            print(f"Inverse A: \n{INVERSE(self.A)}\n", file=f)
            print(f"Inverse L: \n{INVERSE(self.L)}\n", file=f)
            print(f"Inverse U: \n{INVERSE(self.U)}\n", file=f)
            print(f"Inverse D: \n{INVERSE(self.D)}\n", file=f)
            print(f"Inverse of A Decomposed: \n{self.decomposd_inverse}\n", file=f)

            print(f"(Inverse of A) - (Decomposed Inverse): \n{self.inverseA - self.decomposd_inverse}\n", file=f)
