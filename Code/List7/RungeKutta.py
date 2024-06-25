from dataclasses import dataclass, field
import numpy as np

@dataclass
class RungeKutta:
    ft: callable
    t0: float
    tf: float
    Dt: float
    y0: float

    ButcherTableau: list[float] = field(init=False, default_factory=list)
    c: list[float] = field(init=False, default_factory=list)
    a: list[list[float]] = field(init=False, default_factory=list)
    b: list[float] = field(init=False, default_factory=list)

    sol: list[float] = field(init=False, default_factory=list)
    step: float = field(init=False, default=0)

    def __post_init__(self)->None:
        self.sol.append((self.t0, self.y0))

        self.step = int((self.tf - self.t0)/self.Dt)

    def ConstantK(self, index, t, y, k)->float:
        """
        Evaluates the constant K of the Runge-Kutta method
        """
        a = t + self.c[index] * self.Dt
        b = y + self.Dt * sum([self.a[index][j] * k[j] for j in range(index)])

        return self.ft(a, b)
    
    def SetButcherTableau(self, **var)->None:
        """
        Set the Butcher Tableau of the Runge-Kutta method
        You can choose from the following methods:
        - EulerMethod (euler)
        - RK2 (rk2)
        - RK4 (rk4)
        - ThreeEights (ThreeEights)
        - Custom Butcher Tableau

        """ 
        butcher = {"euler": self.EulerMethod(), "rk2": self.RK2(), "rk4": self.RK4(), "ThreeEights": self.ThreeEighthRule()}

        if 'method' in var:
            butcher[var['method']]

        elif 'ButcherTableau' in var:
            self.ButcherTableau = var['ButcherTableau']

        else: 
            raise ValueError("Invalid Butcher Tableau")
    
    def EulerMethod(self)->None:
        """
        Apply the Euler method to solve the ODE
        """
        self.ButcherTableau = [
            [0],
            [[0]],
            [1]
        ]

    def RK2(self)->None:
        """
        Apply the Runge-Kutta 2nd order method to solve the ODE
        """
        self.ButcherTableau = [
            [0, 1/2],
            [[0,0],[1/2,0]],
            [0,1]
        ]

    def RK4(self)->None:
        """
        Apply the Runge-Kutta 4th order method to solve the ODE
        """
        self.ButcherTableau = [
            [0, 1/2, 1/2, 1],
            [[0,0,0,0],[1/2,0,0,0],[0,1/2,0,0],[0,0,1,0]],
            [1/6, 1/3, 1/3, 1/6]
        ]

    def ThreeEighthRule(self)->None:
        """
        Apply the 3/8 rule 4th order Runge-Kutta method to solve the ODE
        """
        self.ButcherTableau = [
            [0, 1/3, 2/3, 1],
            [[0, 0, 0, 0], [1/3, 0, 0, 0], [-1/3, 1, 0, 0], [1, -1, 1, 0]],
            [1/8, 3/8, 3/8, 1/8]
        ]

    def RungeKuttaMethod(self)->None:
        """
        Apply the Runge-Kutta method to solve the ODE 
        using a Butcher Tableau
        """
        t = self.t0
        y = self.y0

        self.c, self.a, self.b = self.ButcherTableau

        k = []
        for _ in range(self.step):
            for i in range(len(self.c)):
                k.append(self.ConstantK(i, t, y, k))

            y += self.Dt * sum([self.b[j] * k[j] for j in range(len(k))])
            t += self.Dt

            k.clear()
            self.sol.append((t, y))

    def WriteResults(self, file)->None:
        """
        Write the results to a file
        """
        with open(file, 'w') as f:
            for t, y in self.sol:
                f.write(f"{t} {y}\n")
            