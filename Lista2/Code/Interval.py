"""
The Interval class operates the main data structure for the numerical integration.
In this class, parameters such as the integration domain, number of intervals, the function to be integrated,
the analytic solution, and the numerical method are stored.

Created by Carlos Puga - 03/23/2024 inspired by the code developed by Giovane Avancini
"""
from dataclasses import dataclass, field
from IntegrationRule import IntegrationRule
from SimpsonOneThirdRule import SimpsonOneThirdRule
from SimpsonThreeEighthsRule import SimpsonThreeEighthsRule
from TrapezoidalRule import TrapezoidalRule
from GaussLegendreRule import GaussLegendreRule

@dataclass
class Interval:
    """
    The Interval class stores the parameters for the numerical integration.
    The attributes are:
    _a: float - The lower limit of the integration domain.
    _b: float - The upper limit of the integration domain.
    _n_refinements: int - The number of refinements to be performed.
    _refinement_level: int - The current refinement level.
    _method: IntegrationRule - The numerical integration method to be used.
    _sub_intervals: list - The list of sub-intervals.
    _numerical_integral: float - The numerical integral of the function.
    _analytic_integral: float - The analytic integral of the function.
    _integration_error: float - The integration error.
    """
    # ------------------------------
    #         ATTRIBUTES
    # ------------------------------
    _a: float
    _b: float
    _n_refinements: int
    _refinement_level: int = field(default=0)
    _n_points: int = field(default=0)
    _method: IntegrationRule = field(init=False, default=None)
    _sub_intervals: list = field(init=False, default_factory=list)
    _numerical_integral: float = field(init=False, default=0.0)
    _analytic_integral: float = field(init=False, default=0.0)
    _integration_error: float = field(init=False, default=0.0)

    # ------------------------------
    #         GETTERS & SETTERS
    # ------------------------------

    @property
    def a(self) -> float: return self._a
    @a.setter
    def a(self, a: float): self._a = a
    
    @property
    def b(self) -> float: return self._b
    @b.setter
    def b(self, b: float): self._b = b

    @property
    def n_refinements(self) -> int: return self._n_refinements
    @n_refinements.setter
    def n_refinements(self, n_refinements: int): self._n_refinements = n_refinements

    @property
    def refinement_level(self) -> int: return self._refinement_level
    @refinement_level.setter
    def refinement_level(self, refinement_level: int): self._refinement_level = refinement_level

    @property
    def n_points(self) -> int: return self._n_points
    @n_points.setter
    def n_points(self, points: int): self._n_points = points

    @property
    def method(self) -> IntegrationRule: return self._method
    @method.setter
    def method(self, method: IntegrationRule): self._method = method

    @property
    def sub_intervals(self) -> list: return self._sub_intervals
    @sub_intervals.setter
    def sub_intervals(self, sub_intervals: list): self._sub_intervals = sub_intervals

    @property
    def numerical_integral(self) -> float: return self._numerical_integral
    @numerical_integral.setter
    def numerical_integral(self, numerical_integral: float): self._numerical_integral = numerical_integral

    @property
    def analytic_integral(self) -> float: return self._analytic_integral
    @analytic_integral.setter
    def analytic_integral(self, analytic_integral: float): self._analytic_integral = analytic_integral

    @property
    def integration_error(self) -> float: return self._integration_error
    @integration_error.setter
    def integration_error(self, integration_error: float): self._integration_error = integration_error

    # ------------------------------
    #         METHODS
    # ------------------------------
    def __post_init__(self):
        """
        The __post_init__ method initializes sub intervals
        """
        if self._n_refinements > 0:
            span: float = (self.b - self.a)

            left_subinterval: Interval = Interval(self.a, self.a + span/2, self.n_refinements - 1, self.refinement_level + 1, self.n_points)
            self.sub_intervals.append(left_subinterval)

            right_subinterval: Interval = Interval(self.a + span/2, self.b, self.n_refinements-1, self.refinement_level + 1, self.n_points)
            self.sub_intervals.append(right_subinterval)

    def NumericalIntegrate(self, func: callable, ref_level: int = 0)->float:
        """
        The numerical_integrate method calculates the numerical integral of the function.
        
        Parameters:
        func: function - The function to be integrated.
        ref_level: int - The refinement level.

        Returns:
        float - The numerical integral of the function.
        """

        if ref_level < 0:
            raise Exception("Warning: Refinement level must be greater than 0.")
        
        elif self.refinement_level == 0 and ref_level > self.n_refinements:
            raise Exception("Warning: Refinement level must be less than the number of refinements.")
        
        self.numerical_integral = 0.0
        if self.refinement_level == ref_level:
            self.numerical_integral = self.method.Integrate(func, self.a, self.b, self.n_points)

        else:
            interval: Interval
            for interval in self.sub_intervals:
                self.numerical_integral += interval.NumericalIntegrate(func, ref_level)

        return self.numerical_integral

    def SetExactSolution(self, exact_func: callable, ref_level)->None:
        """
        The exact_integrate method calculates the exact integral of the function.
        
        Parameters:
        func: function - The function to be integrated.

        Returns:
        float - The exact integral of the function.
        """
        self.analytic_integral = exact_func(self.b) - exact_func(self.a)

        interval: Interval
        for interval in self.sub_intervals:
            if self.refinement_level < ref_level:
                interval.SetExactSolution(exact_func, ref_level)

    def ComputeError(self)->float:
        """
        The compute_error method calculates the integration error.

        Returns:
        float - The integration error.
        """
        self.integration_error = abs(self.analytic_integral - self.numerical_integral)
        
        interval: Interval
        for interval in self.sub_intervals:
            interval.ComputeError()

        return self.integration_error

    def SetSimpsonOneThirdRule(self)->None:
        """
        The SetSimpsonOneThirdIntegration method sets the Simpson's 1/3 integration method.
        """
        self.method = SimpsonOneThirdRule()

        interval: Interval
        for interval in self.sub_intervals:
            interval.SetSimpsonOneThirdRule()

    def SetSimpsonThreeEighthsRule(self)->None:
        """
        The SetSimpsonThreeEighthsIntegration method sets the Simpson's 3/8 integration method.
        """
        self.method = SimpsonThreeEighthsRule()

        interval: Interval
        for interval in self.sub_intervals:
            interval.SetSimpsonThreeEighthsRule()

    def SetTrapezoidalRule(self)->None:
        """
        The SetTrapezoidalRule method sets the Trapezoidal rule.
        """
        self.method = TrapezoidalRule()

        interval: Interval
        for interval in self.sub_intervals:
            interval.SetTrapezoidalRule()

    def SetGaussLegendreRule(self)->None:
        """
        The SetGaussLegendreRule method sets the Gauss-Legendre rule.
        """
        self.method = GaussLegendreRule()

        interval: Interval
        for interval in self.sub_intervals:
            interval.SetGaussLegendreRule()

    def WriteResults(self, nprint: callable, print_sub_intervals: bool = True)->None:
            nprint("")
            nprint(f"Interval: [{self.a}, {self.b}]")
            nprint(f"Refinement Level: {self.refinement_level}")
            nprint(f"Numerical Integral: {self.numerical_integral}")
            nprint(f"Analytic Integral: {self.analytic_integral}")
            nprint(f"Integration Error: {self.integration_error}")
            nprint(f"Number of Sub-Intervals: {len(self.sub_intervals)}")
            nprint(f"Integration Method: {self.method.__class__.__name__}")

            if self.sub_intervals and print_sub_intervals:
                nprint("Sub-Intervals:")
                
                sub_interval: Interval
                for sub_interval in self.sub_intervals:
                    sub_interval.WriteResults(nprint=nprint, print_sub_intervals=print_sub_intervals)
    
    def Print(self, file: str = None, print_sub_intervals: bool = True)->None:
        """
        The Print method prints the Interval data and integration results.
        """
        def CurryPrint(file): return lambda text=" ": print(text, file=file)

        if file:
            with open(file, "w") as f:
                nprint = CurryPrint(f)
                self.WriteResults(nprint=nprint, print_sub_intervals=print_sub_intervals)
        else:
            nprint = print
            self.WriteResults(nprint=nprint, print_sub_intervals=print_sub_intervals)