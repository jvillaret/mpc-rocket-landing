import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


def max_invariant_set(A_cl, X: Polyhedron, max_iter=100) -> Polyhedron:
    """Compute maximal invariant set for autonomous system x+ = A_cl x inside polyhedron X."""
    O = X
    for i in range(max_iter):
        O_prev = O
        F, f = O.A, O.b
        O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.hstack((f, f)))
        O.minHrep(True)
        _ = O.Vrep
        if O == O_prev:
            return O
    return O


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # 1. Define Decision Variables
        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')

        # 2. Define Parameters (values updated in get_u)
        self.x_init = cp.Parameter(self.nx, name='x0')

        # 3. Define Weights and Dynamics
        #Q = np.diag([2.0, 53.0, 0.1])  # Weights for [wy, beta, vx]
        #R = np.diag([7.0])  
        Q = np.diag([1.0, 5.0, 0.1])  # Weights for [wy, beta, vx]
        R = np.diag([20.0])               # Weight for input [d2]

        # Compute LQR terminal cost and gain
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K  # dlqr returns positive K, we need u = Kx

        # Compute terminal invariant set Xf
        # State constraints: |beta| <= 10° (beta is state index 1)
        F_x = np.array([[0, 1, 0], [0, -1, 0]])
        f_x = np.array([np.deg2rad(10), np.deg2rad(10)])
        X = Polyhedron.from_Hrep(F_x, f_x)

        # Input constraints: |d2| <= 15°
        u_max = np.deg2rad(15)
        F_u = np.array([[1], [-1]])
        f_u = np.array([u_max, u_max])
        U = Polyhedron.from_Hrep(F_u, f_u)

        # Closed-loop system and terminal set
        A_cl = self.A + self.B @ K
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        Xf = max_invariant_set(A_cl, X.intersect(KU))
        self.F_xf, self.f_xf = Xf.A, Xf.b

        # Affine term for absolute dynamics
        affine_term = self.xs - self.A @ self.xs - self.B @ self.us

        cost = 0.0
        constraints = [self.x_var[:, 0] == self.x_init]

        for k in range(self.N):
            # Cost: Tracking error + Input effort
            cost += cp.quad_form(self.x_var[:, k] - self.xs, Q)
            cost += cp.quad_form(self.u_var[:, k] - self.us, R)

            # Dynamics constraint
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k] + affine_term
            )

            # Input constraints (approx +/- 15 degrees for d2)
            constraints.append(self.u_var[:, k] <= 0.2618)
            constraints.append(self.u_var[:, k] >= -0.2618)
            
            # State constraints: |beta| <= 10° (beta is state index 1)
            beta_max = np.deg2rad(10)
            constraints.append(self.x_var[1, k] <= beta_max)
            constraints.append(self.x_var[1, k] >= -beta_max)

        # Terminal cost (LQR-based Qf)
        cost += cp.quad_form(self.x_var[:, self.N] - self.xs, Qf)

        # Terminal set constraint: x_N ∈ Xf (defined around xs)
        constraints.append(self.F_xf @ (self.x_var[:, self.N] - self.xs) <= self.f_xf)

        # 4. Create the Optimization Problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        # 1. Update Parameters
        self.x_init.value = x0

        # 2. Solve the Optimization Problem
        self.ocp.solve(solver=cp.PIQP)

        # 3. Extract Results
        if self.u_var.value is not None:
            u0 = self.u_var[:, 0].value
            x_traj = self.x_var.value
            u_traj = self.u_var.value
        else:
            print("MPC Solver failed, returning trim values.")
            u0 = self.us
            x_traj = np.tile(self.xs[:, None], (1, self.N + 1))
            u_traj = np.tile(self.us[:, None], (1, self.N))

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj