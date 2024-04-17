"""
Main file for List 3 - Matrices Decomposition

Created by Carlos Puga - 04/05/2024
"""
from Matrix import Matrix

def Main()->None:
    decomposition = "LU"
    pivoting = True

    if decomposition == "LU":
        myMatrix = Matrix([[1.0, 2.0, 8.0], [6.0, 4.0, 7.0], [5.0, 3.0, 9.0]], pivoting)

    elif decomposition == "LDU":
        myMatrix = Matrix([[1.0, 3.0, 5.0], [7.0, 9.0, 2.0], [4.0, 6.0, 8.0]])
    
    elif decomposition == "LLt":
        myMatrix = Matrix([[23., 10., 9.], [10., 54., 8.], [9., 8., 49.]])

    elif decomposition == "LDLt":
        myMatrix = Matrix([[2., 10., 9.], [10., 18., 8.], [9., 8., 16.]], pivoting)

    else:
        raise Exception("Invalid matrix type")
    
    myMatrix.SetDecompositionMethod(decomposition)
    
    if pivoting:
        decomposition += "_Pivoting"

    myMatrix.Decompose()
    myMatrix.FindInverse()

    output_file = f"{decomposition}.txt"
    myMatrix.Print(output_file)

if __name__ == "__main__":
    Main()