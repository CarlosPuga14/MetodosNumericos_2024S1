"""
Main file for List 3 - Matrices Decomposition
"""
from Matrix import Matrix
import numpy as np

method: dict = {"LU": 0, "LDU": 1, "LLt": 2, "LDLt": 3}

def Main()->None:
    matrixtype = 2

    if matrixtype == method["LU"]:
        myMatrix = Matrix([[1.0, 3.0, 5.0], [7.0, 9.0, 2.0], [4.0, 6.0, 8.0]])
        myMatrix.SetDecompositionMethod("LU")
        output_file = "LU.txt"

    elif matrixtype == method["LDU"]:
        myMatrix = Matrix([[1.0, 3.0, 5.0], [7.0, 9.0, 2.0], [4.0, 6.0, 8.0]])
        myMatrix.SetDecompositionMethod("LDU")
        output_file = "LDU.txt"
    
    elif matrixtype == method["LLt"]:
        myMatrix = Matrix([[23., 10., 9.], [10., 54., 8.], [9., 8., 49.]])
        myMatrix.SetDecompositionMethod("LLt")
        output_file = "LLt.txt"

    elif matrixtype == method["LDLt"]:
        myMatrix = Matrix([[2., 10., 9.], [10., 18., 8.], [9., 8., 16.]])
        myMatrix.SetDecompositionMethod("LDLt")
        output_file = "LDLt.txt"

    else:
        raise Exception("Invalid matrix type")
    
    myMatrix.Decompose()
    myMatrix.FindInverse()
    myMatrix.Print(output_file)

if __name__ == "__main__":
    Main()