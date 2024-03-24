from IntegrationRule import IntegrationRule
from dataclasses import dataclass

@dataclass
class SimpsonOneThirdRule(IntegrationRule):
    """
    Class to compute the integration points and weights for the Simpson's 1/3 rule
    """
    # -------------------
    #      METHODS
    # -------------------
    def IntegrationPoints(self, a: float, b: float, _) -> None:
        """
        Method to compute the integration points
        """
        self.points = [a, (a+b)/2, b]
        self.weights = [1/3, 4/3, 1/3]

    def Xmap(self, xi: float) -> None:
        """
        Method to map the integration points to the physical domain
        """
        self.Xmapped = xi

    def DetJac(self, _, a: float, b: float) -> float:
        self.detjac = (b - a) / 2

    def ComputeRequiredData(self, point: float, weight: float, a: float, b: float) -> None:
        self.Xmap(point)
        self.DetJac(point, a, b)

        self.detjac *= weight

    