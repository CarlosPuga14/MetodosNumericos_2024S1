"""
The sparse matrix class is implemented here. 

Created by Carlos Puga - 04/17/2024
"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SparseMatrix:
    """
    Sparse matrix class. This class is used to store a matrix in a sparse format. 
    """
    data: list = field(init=False, default_factory=list)
    cols: list = field(init=False, default_factory=list)
    ptr: list = field(init=False, default_factory=list)

    def ParseFullMatrix(self, matrix: np.array)->None:
        """
        Parse a full matrix into a sparse matrix. 
        """
        self.ptr.append(0)

        for row in matrix:
            for j, item in enumerate(row):
                if item != 0:
                    self.data.append(item)
                    self.cols.append(j)

            self.ptr.append(len(self.data))

    def FindAij(self, i:int, j:int)->float:
        """
        Find the Aij term in the sparse matrix
        """
        for k in range(self.ptr[i], self.ptr[i+1]):
            if self.cols[k] == j:
                return self.data[k]
        
        return 0.0
    
    def Multiply(self, vector:np.array)->np.array:
        """
        Multiply the sparse matrix by a vector
        """
        prod = np.zeros(len(vector))
        for i in range(len(self.ptr) - 1):
            sum = 0.0
            for k in range(self.ptr[i], self.ptr[i+1]):
                sum += self.data[k] * vector[self.cols[k]]

            prod[i] = sum

        return prod
