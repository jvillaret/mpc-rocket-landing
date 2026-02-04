import numpy as np
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # 1. Define Decision Variables
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        # Slack variable for soft constraint on alpha (state index 1 in local coordinates)
        self.epsilon_alpha = cp.Variable(self.N + 1, name='epsilon_alpha')

        # 2. Define Parameters (values updated in get_u)
        self.x_init = cp.Parameter(self.nx)
        self.x_target = cp.Parameter((self.nx, self.N + 1))
        self.u_target = cp.Parameter((self.nu, self.N))


        #TUNE THESE 
        # 3. Define Weights and Dynamics
        # Tuning parameters from Deliverable 3.1
        Q = np.diag([1.0, 5.0, 0.1])  # Weights for [wx, alpha, vy]
        R = np.diag([20.0])           # Weight for input [d1]
        
        # Compute LQR terminal cost Qf using dlqr
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K  # dlqr returns positive K, we need u = Kx
        
        # Penalty weight for slack variable (soft constraint on alpha)
        # Soft constraint: allows violations when necessary but penalizes them
        # Lower value allows more flexibility, higher value enforces constraint more strictly
        rho_alpha = 50.0  # Penalty weight for slack variable - soft constraint (not hard)

        # Affine term for absolute dynamics: x_next = A(x-xs)+B(u-us)+xs
        affine_term = self.xs - self.A @ self.xs - self.B @ self.us

        cost = 0.0
        constraints = [self.x_var[:, 0] == self.x_init]

        # Alpha constraint limit: 10 degrees
        alpha_max = np.deg2rad(10.0)
        
        for k in range(self.N):
            # Cost: Tracking error + Input effort
            cost += cp.quad_form(self.x_var[:, k] - self.x_target[:, k], Q)
            cost += cp.quad_form(self.u_var[:, k] - self.u_target[:, k], R)
            
            # Penalty for slack variable (linear penalty only)
            cost += rho_alpha * self.epsilon_alpha[k]

            # Dynamics constraint
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k] + affine_term
            )

            # Input constraints (slightly tighter than 15 degrees to account for numerical precision)
            u_max = np.deg2rad(14.9)  # slightly less than 15 degrees for safety margin
            constraints.append(self.u_var[:, k] <= u_max)
            constraints.append(self.u_var[:, k] >= -u_max)
            
            # Soft constraint on alpha: |alpha| <= alpha_max + epsilon_alpha
            # alpha is at index 1 in local state vector
            constraints.append(self.x_var[1, k] <= alpha_max + self.epsilon_alpha[k])
            constraints.append(self.x_var[1, k] >= -alpha_max - self.epsilon_alpha[k])
            
            # Slack variable must be non-negative
            constraints.append(self.epsilon_alpha[k] >= 0)

        # Terminal cost (LQR-based Qf for stability)
        cost += cp.quad_form(self.x_var[:, self.N] - self.x_target[:, self.N], Qf)
        
        # Terminal soft constraint on alpha
        cost += rho_alpha * self.epsilon_alpha[self.N]
        constraints.append(self.x_var[1, self.N] <= alpha_max + self.epsilon_alpha[self.N])
        constraints.append(self.x_var[1, self.N] >= -alpha_max - self.epsilon_alpha[self.N])
        constraints.append(self.epsilon_alpha[self.N] >= 0)

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

        if x_target is not None:
            if x_target.ndim == 1:
                # Broadcast single target vector across horizon
                self.x_target.value = np.tile(x_target[:, None], (1, self.N + 1))
            else:
                self.x_target.value = x_target
        else:
            # Default to trim state
            self.x_target.value = np.tile(self.xs[:, None], (1, self.N + 1))

        if u_target is not None:
            if u_target.ndim == 1:
                self.u_target.value = np.tile(u_target[:, None], (1, self.N))
            else:
                self.u_target.value = u_target
        else:
            # Default to trim input
            self.u_target.value = np.tile(self.us[:, None], (1, self.N))

        # 2. Solve the Optimization Problem
        self.ocp.solve(solver=cp.PIQP, warm_start=True)

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
