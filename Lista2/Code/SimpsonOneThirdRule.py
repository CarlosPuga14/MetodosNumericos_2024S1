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

    def Xmap(self, xi: float, _, __)->None:
        """
        Method to map the integration points to the physical domain
        """
        self.Xmapped = xi

    def DetJac(self, a: float, b: float) -> float:
        """
        Method to compute the determinant of the Jacobian
        """
        self.detjac = (b - a) / 2


    