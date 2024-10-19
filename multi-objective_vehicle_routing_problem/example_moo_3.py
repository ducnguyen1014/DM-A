# A simple example

# Find solution for the following optimization problem

# min f1(x) = 100 * (x_1^2 + x_2^2)
# max f2(x) = -(x_1 - 1)^2 - x_2^2
#
# s.t.  g1(x) = 2 * (x_1 - 0.1) * (x_1 - 0.9) <= 0
#       g2(x) = 20 * (x_1 - 0.4) * (x_1 - 0.6) >= 0
#       -2 <= x_1 <= 2
#       -2 <= x_2 <= 2
#       x in R

# Problem definition

# Most optimization frameworks commit to either minimize or maximize all objectives and to have only ≤ or ≥ constraints. In pymoo, each objective function is supposed to be minimized, and each constraint needs to be provided in the form of ≤ 0.


# In this example, we are going to use pymoo library to find the solution.

# Before using the library, we need to normalize the constraints by dividing g1 and g2 by its corresponding coefficients.

# Coefficients of g1 is |2 * -0.1 * -0.9| = 0.18
# Coefficients of g2 is |20 * -0.4 * -0.6| = 4.8

# We get a normalized problem

# min f1(x) = 100 * (x_1^2 + x_2^2)
# max f2(x) = -(x_1 - 1)^2 - x_2^2
#
# s.t.  g1(x) = 2 * (x_1 - 0.1) * (x_1 - 0.9) / 0.18 <= 0
#       g2(x) = -20 * (x_1 - 0.4) * (x_1 - 0.6) / 0.48 <= 0
#       -2 <= x_1 <= 2
#       -2 <= x_2 <= 2
#       x in R


import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights


# 1. Define the problem
class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=2, n_obj=2, n_ieq_constr=2, xl=np.array([-2, -2]), xu=np.array([2, 2])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Objectives
        f1 = 100 * (x[0] ** 2 + x[1] ** 2)
        f2 = (x[0] - 1) ** 2 + x[1] ** 2

        # Inequality Constraints
        g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        g2 = -20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        # Equality Constraints
        # h1 = ...
        # h2 = ...

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]
        # out["H"] = [h1, h2]


problem = MyProblem()


# 2. Define algorithm: NSGA-II
algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
)


# 3. Define termination criterion
termination = get_termination("n_gen", 40)


# 4. Optimization
result = minimize(
    problem, algorithm, termination, seed=1, save_history=True, verbose=True
)

X = result.X
F = result.F

# print("\nResult:")
# print(f"X: {X}")
# print(f"F: {F}")


# 5. Visualization
# 5.1 Visualize X
# xl, xu = problem.bounds()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ax1.scatter(X[:, 0], X[:, 1], s=30, facecolors="none", edgecolors="r")
# ax1.set_xlim(xl[0], xu[0])
# ax1.set_ylim(xl[1], xu[1])
# ax1.set_title("Design Space")

# ax2.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue")
# ax2.set_title("Objective Space")

# plt.tight_layout()
# plt.show()


# 6. Multi-Criteria Decision Making
F = result.F
xl, xu = problem.bounds()
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue")
plt.title("Objective Space")
plt.show()

fl = F.min(axis=0)
fu = F.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

# WARNING:
# Without normalization, we are comparing oranges with apples. The first objective will dominate any distance calculation in the objective space because of its larger scale. Handling different scales of objectives is an inherent part of any multi-objective algorithms, and, thus, we need to do the same for post-processing.

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue")
plt.scatter(
    approx_ideal[0],
    approx_ideal[1],
    facecolors="none",
    edgecolors="red",
    marker="*",
    s=100,
    label="Ideal Point (Approx)",
)
plt.scatter(
    approx_nadir[0],
    approx_nadir[1],
    facecolors="none",
    edgecolors="black",
    marker="p",
    s=100,
    label="Nadir Point (Approx)",
)
plt.title("Objective Space")
plt.legend()
plt.show()

nF = (F - approx_ideal) / (approx_nadir - approx_ideal)  # Normalized f function

fl = nF.min(axis=0)
fu = nF.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

# plt.figure(figsize=(7, 5))
# plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors="none", edgecolors="blue")
# plt.title("Objective Space")
# plt.show()


# 7. Compromise Programming¶
weights = np.array([0.2, 0.8])

decomp = ASF()

i = decomp.do(nF, 1 / weights).argmin()
print("Best regarding ASF: Point \ni = %s\nF = %s" % (i, F[i]))

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue")
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
plt.title("Best regarding ASF (0.2, 0.8)")
plt.show()


# 8. Pseudo-Weights
i = PseudoWeights(weights).do(nF)
print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue")
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
plt.title("Best regarding Pseudo Weights (0.2, 0.8)")
plt.show()
