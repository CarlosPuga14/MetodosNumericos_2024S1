from PowerMethod import PowerMethod
from matplotlib import pyplot as plt

def PlotConvergence(eigenvalues)->None:
    """
    Plot the convergence of the eigenvalues.
    """
    raise NotImplementedError

def main():
    """
    main function
    """
    A = [
        [1,2,3],
        [2,4,5],
        [3,5,6]
        ]

    pm = PowerMethod(A)

    pm.Run()

    pm.WriteResults("results.txt")

    eigenvalues = pm.GetEigenvalue()

if __name__ == "__main__":
    main()
