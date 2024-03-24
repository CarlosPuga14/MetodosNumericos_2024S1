from IntegrationRule import IntegrationRule
from dataclasses import dataclass

@dataclass
class TrapezoidalRule(IntegrationRule):
    """
    Class to compute the integration points and weights for the Trapezoidal rule
    """
    # -------------------
    #      METHODS
    # -------------------
    def IntegrationPoints(self, a: float, b: float, _) -> None:
        """
        Method to compute the integration points
        """
        self.points = [a, b]
        self.weights = [1/2, 1/2]

    def Xmap(self, xi: float) -> None:
        """
        Method to map the integration points to the physical domain
        """
        self.Xmapped = xi

    def DetJac(self, _, a: float, b: float) -> float:
        self.detjac = (b - a)

    def ComputeRequiredData(self, point: float, weight: float, a: float, b: float) -> None:
        self.Xmap(point)
        self.DetJac(point, a, b)

        self.detjac *= weight