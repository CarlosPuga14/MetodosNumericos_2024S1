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
    data: list = field(init=False, default_factory=list) # List of non-zero elements
    cols: list = field(init=False, default_factory=list) # List of columns
    ptr: list = field(init=False, default_factory=list) # List of cumulative sums of non-zero elements
    sparsity: float = field(init=False, default=0.0) # Sparsity of the matrix
    density: float = field(init=False, default=0.0) # Density of the matrix
    size: int = field(init=False, default=0) # Size of the full matrix

    data_memory: int = field(init=False, default=0) # Memory usage of the data list
    cols_memory: int = field(init=False, default=0)
    ptr_memory: int = field(init=False, default=0)
    total_memory: int = field(init=False, default=0)

    def ParseFullMatrix(self, matrix: np.array)->None:
        """
        Parse a full matrix into a sparse matrix. 
        """
        self.size = len(matrix)

        self.ptr.append(0)
        for row in matrix:
            for j, item in enumerate(row):
                if item != 0:
                    self.data.append(item)
                    self.cols.append(j)

            self.ptr.append(len(self.data))

    def ParseFromFile(self, file:str)->None:
        """
        Parse a full matrix from a file into a sparse matrix. 
        """
        row = 0
        col = 0
        self.ptr.append(0)

        with open(file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip("= { , \n")
                end_row = False

                for element in line.split(","):
                    if "}" in element:
                        element = element.strip("}")
                        end_row = True

                    try: element = float(element)
                    except ValueError: continue

                    if element: 
                        self.data.append(element)
                        self.cols.append(col)

                    col += 1

                    if end_row:
                        self.ptr.append(len(self.data))
                        row += 1
                        col = 0


    def FindAij(self, i:int, j:int)->float:
        """
        Find the Aij term in the sparse matrix notation
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
    
    def EvaluateSparsity(self)->None:
        """
        Evaluate the sparsity and density of the matrix
        """
        self.sparsity = 1 - len(self.data) / (self.size * self.size)
        self.density = 1 - self.sparsity

    def CalcMemoryUsage(self)->None:
        """
        Calculate the memory usage of the sparse matrix
        """
        memory_usage = lambda lst:  sum([4 if isinstance(item, int) else 8 for item in lst])
        
        self.data_memory = memory_usage(self.data)
        self.cols_memory = memory_usage(self.cols)
        self.ptr_memory = memory_usage(self.ptr)

        self.total_memory = self.data_memory + self.cols_memory + self.ptr_memory