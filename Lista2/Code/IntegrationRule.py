"""
The base class for numerical integration.
This is an abstract class that defines the interface for numerical integration.
Integration Methods are supposed to inherit from this class and implement the integration rule.

Created by Carlos Puga - 03/23/2024
"""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

@dataclass
class IntegrationRule(metaclass=ABCMeta):
    """Base class for numerical integration"""
    _weights: list[float] = field(init=False, default_factory=list)
    _points: list[float] = field(init=False, default_factory=list)
    _detjac: float = field(init=False, default=0.0)
    _Xmapped: float = field(init=False, default=0.0)
    
    # ------------------- 
    #    PROPERTIES 
    # -------------------
    @property
    def weights(self)->list[float]: return self._weights
    @weights.setter
    def weights(self, value: list[float])->None: self._weights = value

    @property
    def points(self)->list[float]: return self._points
    @points.setter
    def points(self, value: list[float])->None: self._points = value

    @property
    def detjac(self)->float: return self._detjac
    @detjac.setter
    def detjac(self, value: float)->None: self._detjac = value

    @property
    def Xmapped(self)->float: return self._Xmapped
    @Xmapped.setter
    def Xmapped(self, value: float)->None: self._Xmapped = value

    # -------------------
    #      METHODS
    # -------------------
    @abstractmethod
    def IntegrationPoints(self, a: float, b:float, pOrder: int)->None:
        """
        Method to compute the integration points
        """
        raise NotImplementedError("Warning: You should not be here! Implement the method in the derived class")
    
    @abstractmethod
    def Xmap(self, xi: float)->None:
        """
        Method to compute the map from the reference element to the real element

        Parameters
        ----------
        xi : float
            The reference element coordinate
        """
        raise NotImplementedError("Warning: You should not be here! Implement the method in the derived class")
    
    @abstractmethod
    def DetJac(self, xi: float)->None:
        """
        Method to compute the determinant of the Jacobian

        Parameters
        ----------
        xi : float
            The reference element coordinate
        """
        raise NotImplementedError("Warning: You should not be here! Implement the method in the derived class")
    
    def ComputeRequiredData(self, point: float, weight: float, a: float, b: float)->None:
        """
        Method to compute the required data for the integration
        """
        self.Xmap(point)
        self.DetJac(point, a, b)

        self.detjac *= weight
            
    def Integrate(self, func: callable, a: float, b: float, pOrder: int)->float:
        """
        Method to compute the integral of the function

        Returns
        -------
        float
            The integral of the function in the interval
        """
        self.IntegrationPoints(a, b, pOrder)

        function_integrate: float = 0.0
        for point, weight in zip(self.points, self.weights):
            self.ComputeRequiredData(point, weight, a, b)

            function_integrate += self.detjac * func(self.Xmapped)

        return function_integrate