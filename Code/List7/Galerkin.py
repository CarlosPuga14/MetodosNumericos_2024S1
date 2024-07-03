from dataclasses import dataclass, field
import numpy as np 


@dataclass
class Galerkin:
    u_exact: callable
    source_term: callable
    dudx: callable
    dudy: callable
    p_order: int

    n_points: float = field(init=False)

    xi: list[float] = field(init=False, default_factory=list)
    phi: list[float] = field(init=False, default_factory=list)
    dphi: list[float] = field(init=False, default_factory=list)

    points: list[float] = field(init=False, default_factory=list)
    weights: list[float] = field(init=False, default_factory=list)
    pointsBC: list[float] = field(init=False, default_factory=list)
    weightsBC: list[float] = field(init=False, default_factory=list)

    K: list[float] = field(init=False, default_factory=list)
    F: list[float] = field(init=False, default_factory=list)
    alpha: list[float] = field(init=False, default_factory=list)

    error: float = field(init=False, default=0.0)

    def __post_init__(self)->None: 
        self.n_points = (2 * self.p_order + 1)/2

    def SetNppoints(self, n_points:float)->None:
        """
        Set the number of points for the integration rule
        """
        self.n_points = n_points

    def IntegrationRule(self)->None:
        """
        Define the integration rule for the Galerkin method

        points taken from: https://pomax.github.io/bezierinfo/legendre-gauss.html
        """
        if self.n_points <= 1:
            self.points = np.zeros((1, 2))
            self.weights = np.zeros(1)

            points = [0]
            weights = [2]

        elif self.n_points <= 2:
            self.points = np.zeros((4, 2))
            self.weights = np.zeros(4)

            points = [-0.5773502691896257, 0.5773502691896257]
            weights = [1, 1]


        elif self.n_points <= 3:
            self.points = np.zeros((9, 2))
            self.weights = np.zeros(9)

            points = [-0.7745966692414834, 0, 0.7745966692414834]
            weights = [5 / 9, 8 / 9, 5 / 9]

        elif self.n_points <= 4:
            self.points = np.zeros((16, 2))
            self.weights = np.zeros(16)

            points = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
            weights = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.347854845137]

        elif self.n_points <= 5:
            self.points = np.zeros((25, 2))
            self.weights = np.zeros(25)

            points = [-0.9061798459386640, -0.5384693101056831, 0, 0.5384693101056831, 0.9061798459386640]
            weights = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]

        elif self.n_points <= 6:
            self.points = np.zeros((36, 2))
            self.weights = np.zeros(36)

            points = [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.9324695142031521]
            weights = [0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 	0.4679139345726910, 0.3607615730481386, 0.1713244923791704]

        elif self.n_points <= 7:
            self.points = np.zeros((49, 2))
            self.weights = np.zeros(49)

            points = [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585]
            weights = [0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.2797053914892766, 0.1294849661688697]

        elif self.n_points <= 8:
            self.points = np.zeros((64, 2))
            self.weights = np.zeros(64)

            points = [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498, 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363]
            weights = [0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763]

        elif self.n_points <= 9:
            self.points = np.zeros((81, 2))
            self.weights = np.zeros(81)

            points = [-0.9681602395076261, -0.8360311073266358, -0.6133714327005904, -0.3242534234038089, 0, 0.3242534234038089, 0.6133714327005904, 0.8360311073266358, 0.9681602395076261]
            weights = [0.0812743883615744, 0.1806481606948574, 0.2606106964029354, 0.3123470770400029, 0.3302393550012598, 0.3123470770400029, 0.2606106964029354, 0.1806481606948574, 0.0812743883615744]

        elif self.n_points <= 10:
            self.points = np.zeros((100, 2))
            self.weights = np.zeros(100)

            points = [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312, 0.1488743389816312, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717]
            weights = [0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881]

        else:
            raise ValueError("Invalid number of points")
        
        return points, weights
        
    def IntegrationRuleDomain(self)->None:
        points, weights = self.IntegrationRule()

        for i, (p1, w1) in enumerate(zip(points, weights)):
            for j, (p2, w2) in enumerate(zip(points, weights)):
                self.points[i * len(points)+j][0] = p1
                self.points[i * len(points)+j][1] = p2 

                self.weights[i * len(points)+j] = w1 * w2

    def IntegrationRuleBC(self)->None:
        self.pointsBC, self.BCweights = self.IntegrationRule()

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

    def Contribute(self)->None:
        """
        Define the stiffness matrix for the Galerkin method
        """
        self.SetNppoints((self.p_order * 2 + 1)/2)
        self.IntegrationRuleDomain()

        for (x, y), w in zip(self.points, self.weights):
            self.BasisFuntcion(x, y)

            if not len(self.K):
                self.K = np.zeros((len(self.phi), len(self.phi)))

            for i, dphi_i in enumerate(self.dphi):
                for j, dphi_j in enumerate(self.dphi):
                    self.K[i, j] += np.dot(dphi_i, dphi_j) * w

    def BodyForce(self)->None:
        """
        Define the body force for the Galerkin method
        """
        self.SetNppoints(10)
        self.IntegrationRuleDomain()
        for (x, y), w in zip(self.points, self.weights):
            self.BasisFuntcion(x, y)

            if not len(self.F):
                self.F = np.zeros(len(self.phi))

            for i, phi in enumerate(self.phi): 
                self.F[i] += phi * self.source_term(x, y) * w

    def ContributeBC(self)->list[float]:
        """
        Define the contribution of each integration point to the element matrix
        """
        self.IntegrationRuleBC()
        for point, w in zip(self.pointsBC, self.BCweights):
            self.BasisFuntcion(1, point)
            for i, phi in enumerate(self.phi):
                self.F[i] += phi * self.dudx(1, point) * w

            self.BasisFuntcion(point, 1)
            for i, phi in enumerate(self.phi):
                self.F[i] += phi * self.dudy(point, 1) * w
        
    def Error(self)->None:
        """ 
        Evaluates the error of the Galerkin method
        """
        self.SetNppoints(10)
        self.IntegrationRuleDomain()

        for (x, y), w in zip(self.points, self.weights):
            self.BasisFuntcion(x, y)
                
            u_h = self.alpha @ self.phi

            self.error += ((self.u_exact(x, y) - u_h) ** 2) * w 

        self.error = np.sqrt(self.error)

    def Run(self)->None:
        """
        Run the Galerkin method
        """
        self.Contribute()
        self.BodyForce()
        self.ContributeBC()

        self.alpha = np.linalg.solve(self.K, self.F)

        self.Error()