"""
Main file for List 3 - Matrices Decomposition
"""
from Matrix import Matrix
import numpy as np

INVERSE: callable = np.linalg.inv

def Main()->None:
    matrixtype = 2

    if matrixtype == 0 or matrixtype == 1:
        # ---- LU and LDU -----
        myMatrix = Matrix([[1.0, 3.0, 5.0], [7.0, 9.0, 2.0], [4.0, 6.0, 8.0]])
        myMatrix.SetDecomposition("LDU")
        myMatrix.Decompose()

        print(f"Matrix A: \n{myMatrix.A}")
        print(f"Matrix L: \n{myMatrix.L}")
        print(f"Matrix U: \n{myMatrix.U}")
        print(f"Matrix D: \n{myMatrix.D}")

        if myMatrix.decomposition_type == "LU":
            print("LU Decomposition:")
            print(INVERSE(myMatrix.A) - INVERSE(myMatrix.L @ myMatrix.U))

        elif myMatrix.decomposition_type == "LDU":
            print("LDU Decomposition:")
            print(INVERSE(myMatrix.A) - INVERSE(myMatrix.L @ myMatrix.D @ myMatrix.U))
    
    elif matrixtype == 2:
        # ------ LLt -----
        posdefMatrix = Matrix([[23., 10., 9.], [10., 54., 8.], [9., 8., 49.]])
        posdefMatrix.SetDecomposition("LLt")
        posdefMatrix.Decompose()

        print(f"Matrix A: \n{posdefMatrix.A}")
        print(f"Matrix L: \n{posdefMatrix.L}")
        print(f"Matrix L.T: \n{posdefMatrix.U}")

        print("LLt Decomposition:")
        print(INVERSE(posdefMatrix.A) - INVERSE(posdefMatrix.L @ posdefMatrix.U))

    elif matrixtype == 3:
        # ------ LDLt -----
        symMatrix = Matrix([[2., 10., 9.], [10., 18., 8.], [9., 8., 16.]])
        symMatrix.SetDecomposition("LDLt")
        symMatrix.Decompose()

        print(f"Matrix A: \n{symMatrix.A}")
        print(f"Matrix L: \n{symMatrix.L}")
        print(f"Matrix L.T: \n{symMatrix.U}")
        print(f"Matrix D: \n{symMatrix.D}")

        print("LDLt Decomposition:")
        print(INVERSE(symMatrix.A) - INVERSE(symMatrix.L @ symMatrix.D @ symMatrix.L.T))

    else:
        raise Exception("Invalid matrix type")

if __name__ == "__main__":
    Main()