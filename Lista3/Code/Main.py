"""
Main file for List 3 - Matrices Decomposition

Created by Carlos Puga - 04/05/2024
"""
from Matrix import Matrix

def Main()->None:
    decomposiion = "LU"

    if decomposiion in ["LU", "LDU"]:
        myMatrix = Matrix([[1.0, 3.0, 5.0], [7.0, 9.0, 2.0], [4.0, 6.0, 8.0]])
    
    elif decomposiion == "LLt":
        myMatrix = Matrix([[23., 10., 9.], [10., 54., 8.], [9., 8., 49.]])

    elif decomposiion == "LDLt":
        myMatrix = Matrix([[2., 10., 9.], [10., 18., 8.], [9., 8., 16.]])

    else:
        raise Exception("Invalid matrix type")
    
    myMatrix.SetDecompositionMethod(decomposiion)
    myMatrix.Decompose()
    myMatrix.FindInverse()

    output_file = f"{decomposiion}.txt"
    myMatrix.Print(output_file)

if __name__ == "__main__":
    Main()