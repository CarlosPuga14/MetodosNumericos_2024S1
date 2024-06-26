from dataclasses import dataclass, field
import numpy as np 


@dataclass
class Galerkin:
    domain: list[float]
    u_exact: callable
    p_order: int

    xi: list[float] = field(init=False, default_factory=list)
    phi: list[float] = field(init=False, default_factory=list)
    dphi: list[float] = field(init=False, default_factory=list)

    points: list[float] = field(init=False, default_factory=list)
    weights: list[float] = field(init=False, default_factory=list)

    K: list[float] = field(init=False, default_factory=list)
    F: list[float] = field(init=False, default_factory=list)

    def IntegrationRule(self)->None:
        """
        Define the integration rule for the Galerkin method
        """
        pol_order = 2 * self.p_order
        n_points = (pol_order + 1)/2

        if n_points <= 1:
            self.points = np.zeros((1, 2))
            self.weights = np.zeros(1)

            points = [0]
            weights = [2]

        elif n_points <= 2:
            self.points = np.zeros((4, 2))
            self.weights = np.zeros(4)

            points = [-0.5773502691896257, 0.5773502691896257]
            weights = [1, 1]


        elif n_points <= 3:
            self.points = np.zeros((9, 2))
            self.weights = np.zeros(9)

            points = [-0.7745966692414834, 0, 0.7745966692414834]
            weights = [5 / 9, 8 / 9, 5 / 9]

        elif n_points <= 4:
            self.points = np.zeros((16, 2))
            self.weights = np.zeros(16)

            points = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
            weights = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.347854845137]

        elif n_points <= 5:
            self.points = np.zeros((25, 2))
            self.weights = np.zeros(25)

            points = [-0.9061798459386640, -0.5384693101056831, 0, 0.5384693101056831, 0.9061798459386640]
            weights = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]

        elif n_points <= 6:
            self.points = np.zeros((36, 2))
            self.weights = np.zeros(36)

            points = [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.9324695142031521]
            weights = [0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 	0.4679139345726910, 0.3607615730481386, 0.1713244923791704]

        else:
            raise ValueError("Invalid number of points")
        
        for i, (p1, w1) in enumerate(zip(points, weights)):
            for j, (p2, w2) in enumerate(zip(points, weights)):
                self.points[i * len(points)+j][0] = p1
                self.points[i * len(points)+j][1] = p2 

                self.weights[i * len(points)+j] = w1 * w2

    def BasisFuntcion(self, x:float, y:float)->list[float]:
        """
        Define the basis function for the Galerkin method
        """
        self.phi = np.zeros((self.p_order) ** 2)
        self.dphi = np.zeros(((self.p_order) ** 2, 2))

        b0 = lambda x: (x+1)
        db0 = lambda _: 1

        basis = [
            lambda _: 1,
            lambda x: x,
            lambda x: 0.5 * (-1 + 3 * x ** 2),
            lambda x: 0.5 * (-3 * x + 5 * x ** 3),
            lambda x: 0.125 * (3 - 30 * x ** 2 + 35 * x ** 4),
        ]

        d_basis = [
            lambda _: 0,
            lambda _: 1,
            lambda x: 3 * x,
            lambda x: 0.5 * (-3 + 15 * x ** 2),
            lambda x: 0.125 * (-60 * x + 140 * x ** 3),
        ]

        if self.p_order > len(basis):
            raise ValueError("Invalid polynomial order")

        k = self.p_order
        for i in range(k):
            for j in range(k):
                self.phi[i * (k) + j] = b0(x) * b0(y) * basis[i](x) * basis[j](y)

        for i in range(k):
            for j in range(k):
                self.dphi[i * (k) + j, 0] = db0(x) * b0(y) * basis[i](x) * basis[j](y) + b0(x) * b0(y) * d_basis[i](x) * basis[j](y)
                self.dphi[i * (k) + j, 1] = b0(x) * db0(y) * basis[i](x) * basis[j](y) + b0(x) * b0(y) * basis[i](x) * d_basis[j](y)

    def Contribute(self, x:float, y:float, w:float)->list[float]:
        """
        Define the contribution of each integration point to the element matrix
        """
        kel = np.zeros((len(self.phi), len(self.phi)))
        for i, dphi_i in enumerate(self.dphi):
            for j, dphi_j in enumerate(self.dphi):
                kel[i, j] += np.dot(dphi_i, dphi_j) * w

        return kel

    def ContributeBC(self)->None:
        """
        Define the contribution of each integration point to the load vector
        """
        raise NotImplementedError("You must implement the contribution of the boundary conditions")
    
    def Run(self)->None:
        """
        Run the Galerkin method
        """
        self.IntegrationRule()

        for (x, y), w in zip(self.points, self.weights):
            self.BasisFuntcion(x, y)

            if not len(self.K):
                self.K = np.zeros((len(self.phi), len(self.phi)))

            if not any(self.F):
                self.F = np.zeros(len(self.phi))

            self.K += self.Contribute(x, y, w)

