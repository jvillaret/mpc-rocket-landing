import numpy as np
import cvxpy as cp

from .MPCControl_base import MPCControl_base


class MPCControl_y(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7, 10])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        #################################################
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.s_y = cp.Variable(1, nonneg=True)   # Slack for soft constraints

        # Define optimization parameters
        self.x_init = cp.Parameter(self.nx)
        self.x_ref = cp.Parameter(self.nx)

        # Define the cost function
        Q = np.diag([100, 1, 0.1, 50.0])  # Penalize alpha and y position
        R = np.diag([1])
        S = 1e6

        # Define cost function
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(self.x_var[:, k] - self.x_ref, Q) + cp.quad_form(self.u_var[:, k], R)
        cost += cp.quad_form(self.x_var[:, self.N] - self.x_ref, Q)    # Terminal cost
        cost += S * self.s_y    # Add slack penalty

        # Define constraints
        constraints = []
        # Initial state constraint
        constraints += [self.x_var[:, 0] == self.x_init]

        # System dynamics and operational constraints
        for k in range(self.N):
            # System dynamics
            constraints += [self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]]

            # Soft state constraints |alpha| <= 10 deg (0.1745 rad)
            alpha_idx = 1  # alpha is the second state in sys_y
            constraints += [self.x_var[alpha_idx, k] <= 0.1745 + self.s_y]
            constraints += [self.x_var[alpha_idx, k] >= -0.1745 - self.s_y]

            # Hard input constraints |d1| <= 15 deg (0.26 rad)
            constraints += [self.u_var[:, k] <= 0.26]
            constraints += [self.u_var[:, k] >= -0.26]
        # Create the optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        
        # Calculate delta states
        delta_x0 = x0 - self.xs
        delta_xref = x_target - self.xs

        # Set the parameter values for the OCP
        self.x_init.value = delta_x0
        self.x_ref.value = delta_xref

        # Solve the OCP
        self.ocp.solve(solver=cp.PIQP, warm_start=True)

        # Handle solver failure
        if self.ocp.status != cp.OPTIMAL and self.ocp.status != cp.OPTIMAL_INACCURATE:
            print("[MPC_y] OCP FAILED TO SOLVE!")
            u0_delta = np.zeros(self.nu)
            x_traj_delta = np.tile(delta_x0, (self.N + 1, 1)).T
            u_traj_delta = np.zeros((self.nu, self.N))
        else:
            u0_delta = self.u_var.value[:, 0]
            x_traj_delta = self.x_var.value
            u_traj_delta = self.u_var.value

        # Return the absolute control input and predicted trajectories
        u0 = u0_delta + self.us
        x_traj = x_traj_delta + np.tile(self.xs, (self.N + 1, 1)).T
        u_traj = u_traj_delta + np.tile(self.us, (self.N, 1)).T

        #################################################

        return u0, x_traj, u_traj
