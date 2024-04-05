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

@dataclass
class Matrix:
    """
    Class to represent a matrix and its decomposition
    """
    A: np.array # Matrix
    pivoting: bool = False # Pivoting flag
    A_Decomposed: np.array = field(init=False) # Decomposed matrix
    decomposition_type: str = None # Decomposition type
    
    L: np.array = field(init=False) # Lower matrix
    U: np.array = field(init=False) # Upper matrix
    D: np.array = field(init=False) # Diagonal matrix

    def __post_init__(self):
        size = len(self.A)

        self.A = ARRAY(self.A)
        self.L = IDENTITY(size)
        self.U = IDENTITY(size)
        self.D = IDENTITY(size)
        self.A_Decomposed = self.A.copy()

    # -----------------
    # ---- METHODS ----
    # -----------------
    def SetDecomposition(self, decomposition_type: str)->None:
        """
        Set the decomposition type
        """
        if not isinstance(decomposition_type, str):
            raise Exception("The decomposition type must be a string")
        
        if not decomposition_type in ["LU", "LDU", "LLt", "LDLt"]:
            raise Exception(f"The '{decomposition_type}' decomposition is not valid. Please choose one of the following: LU, LDU, LLt, LDLt")
        
        self.decomposition_type = decomposition_type

    def PivotMatrix(self)->None:
        raise NotImplementedError("Pivoting not implemented yet")

    def LU_Decomposition(self)->None:
        """
        Decompose the matrix using LU decomposition
        """
        if self.pivoting:
            self.PivotMatrix()

        size = len(self.A_Decomposed)
        
        # Rank 1 update
        for i in range(size):
            self.A_Decomposed[i+1::, i] /= self.A_Decomposed[i, i]
            self.A_Decomposed[i+1::, i+1::] -= OUTER(self.A_Decomposed[i+1::, i], self.A_Decomposed[i, i+1::])

        # Fill L 
        for i in range(1, size):
            for j in range(i):
                self.L[i, j] = self.A_Decomposed[i, j]
                
        # Fill U
        for j in range(size):
            for i in range(j+1):
                self.U[i, j] = self.A_Decomposed[i, j]

    def LDU_Decomposition(self)->None:
        """
        Decompose the matrix using LDU decomposition
        """
        raise NotImplementedError("LDU decomposition not implemented yet")
    
    def LLt_Decomposition(self)->None:
        """
        Decompose the matrix using LLt (Cholesky) decomposition
        """
        raise NotImplementedError("LLt decomposition not implemented yet")
    
    def LDLt_Decomposition(self)->None:
        """
        Decompose the matrix using LDLt decomposition
        """
        raise NotImplementedError("LDLt decomposition not implemented yet")

    def Decompose(self)->None:
        """
        Decompose the matrix using the specified decomposition
        """
        if self.decomposition_type in ["LDU", "LLt", "LDLt"] and self.pivoting:
            raise Exception(f"Pivoting not defined for the '{self.decomposition_type}' decomposition. Please try again with 'LU' decomposition.")

        if self.decomposition_type == "LU":
            self.LU_Decomposition()

        elif self.decomposition_type == "LDU":
            self.LDU_Decomposition()

        elif self.decomposition_type == "LLt":
            self.LLt_Decomposition()

        elif self.decomposition_type == "LDLt":
            self.LDLt_Decomposition()

        else:
            text = f"The '{self.decomposition_type}' decomposition is not valid. Please choose one of the following: "
            text += "LU, LDU, LLt, LDLt"
            
            raise Exception(text)