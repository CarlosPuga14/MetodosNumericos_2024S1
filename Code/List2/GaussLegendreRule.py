from IntegrationRule import IntegrationRule
from dataclasses import dataclass
import numpy as np

@dataclass
class GaussLegendreRule(IntegrationRule):
    """
    The GaussLegendreRule class implements the Gauss-Legendre quadrature rule.
    """
    # -------------------
    #      METHODS
    # -------------------
    def IntegrationPoints(self, _, __, npoints: int) -> None:
        match npoints:
            case 1:
                self.weights.append(2)
                
                self.points.append(0)

            case 2:
                self.weights.append(1)
                self.weights.append(1)

                self.points.append(-1/np.sqrt(3))
                self.points.append(1/np.sqrt(3))

            case 3:
                self.weights.append(5/9)
                self.weights.append(8/9)
                self.weights.append(5/9)

                self.points.append(-np.sqrt(3/5))
                self.points.append(0)
                self.points.append(np.sqrt(3/5))

            case 4:
                self.weights.append(0.2369268850561891)
                self.weights.append(0.4786286704993665)
                self.weights.append(0.5688888888888889)
                self.weights.append(0.4786286704993665)
                self.weights.append(0.2369268850561891)
                
                self.points.append(-0.9061798459386640)
                self.points.append(-0.5384693101056831)
                self.points.append(0)
                self.points.append(0.5384693101056831)
                self.points.append(0.9061798459386640)

            case _:
                raise ValueError("Invalid number of points")
            
    def Xmap(self, xi: float, a: float, b: float) -> None:
        """
        Method to map the integration points to the physical domain
        """
        self.Xmapped = a * (1 - xi) / 2 + b * (1 + xi) / 2

    def DetJac(self, a: float, b: float) -> float:
        """
        Method to compute the determinant of the Jacobian
        """
        self.detjac = (b - a) / 2
