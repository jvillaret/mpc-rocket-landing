"""
MPCControl_xvel - X-velocity subsystem controller for Deliverable 4.1

This controller uses SOFT CONSTRAINTS with slack variables to handle model mismatch.
Tuning parameters:
    - Q = diag([200.0, 100.0, 10.0]) for states [wy, beta, vx]
    - R = [100.0] for input [d2]
    - Input constraints: ±15° (hard constraints)
    - State constraints: beta within ±10° (SOFT constraints with slack variables)
    - Slack penalty: 50000 (very heavy penalty to minimize violations)
    - Warm start enabled

Soft constraints ensure the problem remains feasible even under model mismatch,
while very heavy penalties (rho_slack) and increased Q/R weights strongly discourage constraint violations.
"""
import numpy as np
import cvxpy as cp

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # 1. Define Decision Variables
        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')
        
        # Slack variables for soft state constraints (beta)
        # One slack variable per time step for upper and lower bounds
        self.epsilon_beta_pos = cp.Variable(self.N + 1, nonneg=True, name='epsilon_beta_pos')
        self.epsilon_beta_neg = cp.Variable(self.N + 1, nonneg=True, name='epsilon_beta_neg')

        # 2. Define Parameters (values updated in get_u)
        self.x_init = cp.Parameter(self.nx, name='x0')
        self.x_target = cp.Parameter((self.nx, self.N + 1))
        self.u_target = cp.Parameter((self.nu, self.N))


        # 3. Define Weights and Dynamics
        # Tuned for Deliverable 4.1: Nonlinear simulation with model mismatch + soft constraints
        # Strategy: Even higher weights to work with soft constraints and minimize violations
        # Q = diag([wy_weight, beta_weight, vx_weight])
        Q = np.diag([100.0, 0.1, 0.1])  # Weights for [wy, beta, vx]
        # Increased weight on wy (200.0) to strongly penalize aggressive rotational maneuvers
        # Increased weight on beta (100.0) to very strongly penalize large angles
        # Moderate weight on vx (10.0) for velocity tracking
        
        R = np.diag([1.0])  # Weight for input [d2]
        # Increased input penalty (100.0) to encourage even smoother control actions
        
        # Penalty weight for slack variables (soft constraints)
        # Very high penalty to strongly discourage constraint violations
        # Increased from 10000 to 50000 to make violations extremely expensive
        rho_slack = 200

        # Affine term for absolute dynamics: x_next = A*x + B*u + (xs - A*xs - B*us) = A(x-xs)+B(u-us)+xs
        affine_term = self.xs - self.A @ self.xs - self.B @ self.us

        cost = 0.0
        constraints = [self.x_var[:, 0] == self.x_init]

        for k in range(self.N):
            # Cost: Tracking error + Input effort
            cost += cp.quad_form(self.x_var[:, k] - self.x_target[:, k], Q)
            cost += cp.quad_form(self.u_var[:, k] - self.u_target[:, k], R)
            
            # Penalty for slack variables (soft constraints)
            # L2 penalty (quadratic) + L1 penalty (linear) for better constraint satisfaction
            cost += rho_slack * (cp.square(self.epsilon_beta_pos[k + 1]) + cp.square(self.epsilon_beta_neg[k + 1]))
            cost += rho_slack * (self.epsilon_beta_pos[k + 1] + self.epsilon_beta_neg[k + 1])

            # Dynamics constraint
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k] + affine_term
            )

            # Input constraints: ±15° (hard constraints)
            u_max = np.deg2rad(15.0)
            constraints.append(self.u_var[:, k] <= u_max)
            constraints.append(self.u_var[:, k] >= -u_max)
            
            # State constraints: beta (index 1 in [wy, beta, vx]) - SOFT CONSTRAINTS
            # Use original constraint limit (±10°) but allow violations through slack variables
            beta_max = np.deg2rad(9.9)
            # Soft upper bound: beta <= 10° + epsilon_pos (epsilon_pos >= 0)
            constraints.append(self.x_var[1, k + 1] <= beta_max + self.epsilon_beta_pos[k + 1])
            # Soft lower bound: beta >= -10° - epsilon_neg (epsilon_neg >= 0)
            constraints.append(self.x_var[1, k + 1] >= -beta_max - self.epsilon_beta_neg[k + 1])

        # Terminal cost
        cost += cp.quad_form(self.x_var[:, self.N] - self.x_target[:, self.N], Q)
        
        # Penalty for terminal slack variables
        cost += rho_slack * (cp.square(self.epsilon_beta_pos[self.N]) + cp.square(self.epsilon_beta_neg[self.N]))
        cost += rho_slack * (self.epsilon_beta_pos[self.N] + self.epsilon_beta_neg[self.N])
        
        # Terminal state constraint: beta - SOFT CONSTRAINT at terminal step
        beta_max = np.deg2rad(10.0)
        constraints.append(self.x_var[1, self.N] <= beta_max + self.epsilon_beta_pos[self.N])
        constraints.append(self.x_var[1, self.N] >= -beta_max - self.epsilon_beta_neg[self.N])

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