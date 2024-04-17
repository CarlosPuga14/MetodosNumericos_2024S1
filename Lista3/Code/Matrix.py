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

    permuted_rows: list = field(default_factory=list) # permuted rows
    permuted_cols: list = field(default_factory=list) # permuted columns

    def __post_init__(self):
        self.size = len(self.A)

        self.A = ARRAY(self.A)
        self.A_Decomposed = self.A.copy()
        self.inverseA = INVERSE(self.A)

        self.L = IDENTITY(self.size)
        self.U = IDENTITY(self.size)
        self.D = IDENTITY(self.size)
        self.decomposd_inverse = ZEROS(self.A)

        self.permuted_rows = [i for i in range(self.size)]
        self.permuted_cols = [i for i in range(self.size)]

    # -----------------
    # ---- SETTERS ----
    # -----------------
    def SetDecompositionMethod(self, decomposition_type: str)->None:
        """
        Set the decomposition type
        """        
        if not decomposition_type in ["LU", "LDU", "LLt", "LDLt"]:
            raise Exception(f"The '{decomposition_type}' decomposition is not valid. Please choose one of the following: LU, LDU, LLt, LDLt")
        
        self.decomposition_type = decomposition_type
    
    # -----------------
    # ---- GENERAL ----
    # ---- METHODS ----
    # -----------------
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

    # -----------------
    # ---- PIVOTING ---
    # ---- METHODS ----
    # -----------------
    def FindMaxRowAndCol(self, submatrix:np.array)->tuple[int]:
        """
        Find the maximum value in the submatrix
        """
        max_number = submatrix[0, 0]

        if self.decomposition_type == "LU":
            for i, row in enumerate(submatrix):
                for j, number in enumerate(row):
                    if abs(number) > max_number:
                            max_number = number
                            perm_row_index = i
                            perm_col_index = j

        elif self.decomposition_type == "LDLt":
            for i in range(len(submatrix)):
                number = submatrix[i, i]

                if abs(number) > max_number:
                    max_number = number
                    perm_row_index = i
                    perm_col_index = i

        else:
            raise Exception(f"Pivoting not defined for the '{self.decomposition_type}' decomposition. Please try again with 'LU' decomposition.")

        return perm_row_index, perm_col_index


    def PermutationMatrix(self, permutation_vector, rows=True)->np.array:
        """
        Create a permutation matrix
        """
        perm_matrix = np.zeros((self.size, self.size))

        for i, index in enumerate(permutation_vector):
            if rows:
                perm_matrix[i, index] = 1

            else:
                perm_matrix[index, i] = 1

        return perm_matrix

    def UpdatedPermutationVector(self, rows, cols, row_max, col_max, index)->tuple[list]:
        """
        Update the permutation vector
        """
        rows[index], rows[row_max] = rows[row_max], rows[index]
        cols[index], cols[col_max] = cols[col_max], cols[index]

        self.permuted_rows[index], self.permuted_rows[row_max] = self.permuted_rows[row_max], self.permuted_rows[index]
        self.permuted_cols[index], self.permuted_cols[col_max] = self.permuted_cols[col_max], self.permuted_cols[index]

        return rows, cols

    def Pivot(self, index:int)->None:
        """
        Perfom the pivoting
        """
        rows = [i for i in range(self.size)]
        cols = [i for i in range(self.size)]

        submatrix = self.A_Decomposed[index::, index::].copy()

        row_max, col_max = self.FindMaxRowAndCol(submatrix)
        row_max += index
        col_max += index

        rows, cols = self.UpdatedPermutationVector(rows, cols, row_max, col_max, index)

        perm_row = self.PermutationMatrix(rows)
        perm_col = self.PermutationMatrix(cols, rows=False)

        self.A_Decomposed = perm_row @ self.A_Decomposed @ perm_col

    # -----------------
    # - DECOMPOSITION -
    # ---- METHODS ----
    # -----------------
    def Pivot_Decomposition(self)->None:
        """
        Decompose the matrix using pivoting LU or LDLt decomposition
        """
        for index in range(self.size - 1):
            self.Pivot(index)
            
            self.A_Decomposed[index+1::, index] /= self.A_Decomposed[index, index]
            self.A_Decomposed[index+1::, index+1::] -= OUTER(self.A_Decomposed[index+1::, index], self.A_Decomposed[index, index+1::])

        if self.decomposition_type == "LU":
            self.Fill_L()
            self.Fill_U()

        elif self.decomposition_type == "LDLt":
            self.Fill_L()
            self.Fill_D()
            self.U = self.L.T

        self.permuted_rows = self.PermutationMatrix(self.permuted_rows)
        self.permuted_cols = self.PermutationMatrix(self.permuted_cols, rows=False)

    def LU_Decomposition(self)->None:
        """
        Decompose the matrix using LU decomposition
        """
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

    # -----------------
    # ---- CHEKING ----
    # ---- METHODS ----
    # -----------------
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

    # -----------------
    # ---- RESULTS ----
    # ---- METHODS ----
    # -----------------
    def Decompose(self)->None:
        """
        Decompose the matrix using the specified decomposition
        """
        if (self.decomposition_type in ["LDU", "LLt"]) and (self.pivoting):
            raise Exception(f"Pivoting not defined for the '{self.decomposition_type}' decomposition. Please try again with 'LU' decomposition.")
        
        if self.pivoting:
            self.Pivot_Decomposition()

        elif self.decomposition_type == "LU":
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
            text = f"The '{self.decomposition_type}' decomposition is not valid. Please choose one of the following: LU, LDU, LLt, LDLt "

            raise Exception(text)
        
    def FindInverse(self)->None:
        """
        Find the inverse of the matrix using the decomposition
        """
        if self.pivoting:
            self.decomposd_inverse = self.permuted_rows.T @ INVERSE(self.U) @ INVERSE(self.D) @ INVERSE(self.L) @ self.permuted_cols.T

        elif self.decomposition_type in ["LU", "LLt"]:
            self.decomposd_inverse = INVERSE(self.U) @ INVERSE(self.L)
        
        elif self.decomposition_type in ["LDU", "LDLt"]:
            self.decomposd_inverse = INVERSE(self.U) @ INVERSE(self.D) @ INVERSE(self.L)
        
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

            if self.pivoting:
                print(f"Permuted Rows: \n{self.permuted_rows}\n", file=f)
                print(f"Permuted Columns: \n{self.permuted_cols}\n", file=f)

            print(f"Inverse A: \n{INVERSE(self.A)}\n", file=f)
            print(f"Inverse L: \n{INVERSE(self.L)}\n", file=f)
            print(f"Inverse U: \n{INVERSE(self.U)}\n", file=f)
            print(f"Inverse D: \n{INVERSE(self.D)}\n", file=f)
            print(f"Inverse of A Decomposed: \n{self.decomposd_inverse}\n", file=f)

            print(f"(Inverse of A) - (Decomposed Inverse): \n{self.inverseA - self.decomposd_inverse}\n", file=f)