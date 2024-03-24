from IntegrationRule import IntegrationRule
from dataclasses import dataclass

@dataclass
class SimpsonThreeEighthsRule(IntegrationRule):
    """
    Class to compute the integration points and weights for the Simpson's 3/8 rule
    """
    # -------------------
    #      METHODS
    # -------------------
    def IntegrationPoints(self, a: float, b: float, _) -> None:
        """
        Method to compute the integration points
        """
        self.points = [a, (2* a + b) / 3, (a + 2 * b) / 3, b]
        self.weights = [1/8, 3/8, 3/8, 1/8]

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