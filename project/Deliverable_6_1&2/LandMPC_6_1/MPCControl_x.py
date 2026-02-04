import numpy as np
import cvxpy as cp

from .MPCControl_base import MPCControl_base


class MPCControl_x(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6, 9])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        #################################################

        # Define optimization variables
        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.N))
        s_x = cp.Variable(1, nonneg=True)   # Slack for soft constraints

        # Define optimization parameters
        x_init = cp.Parameter(self.nx)
        x_ref = cp.Parameter(self.nx)

        # Tuning matrices
        Q = np.diag([100.0, 1, 0.1, 50.0])  # Penalize beta and x position
        R = np.diag([1])                
        S = 1e6                           
 
        # Define cost function
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(x[:, k] - x_ref, Q) + cp.quad_form(u[:, k], R)
        cost += cp.quad_form(x[:, self.N] - x_ref, Q)    # Terminal cost
        cost += S * s_x    # Add slack penalty

        # Define constraints
        constraints = []
        # Initial state constraint
        constraints += [x[:, 0] == x_init]

        # System dynamics and operational constraints
        for k in range(self.N):
            # System dynamics
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]

            # Soft state constraints |beta| <= 10 deg (0.1745 rad)
            beta_idx = 1  # beta is the second state in sys_x
            constraints += [x[beta_idx, k] <= 0.1745 + s_x]
            constraints += [x[beta_idx, k] >= -0.1745 - s_x]

            # Hard input constraints |d2| <= 15 deg (0.26 rad)
            constraints += [u[:, k] <= 0.26]
            constraints += [u[:, k] >= -0.26]

        # Create the optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # for access in get_u
        self._x = x
        self._u = u
        self._s_x = s_x
        self._x_init = x_init
        self._x_ref = x_ref
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
         
        # Calculate delta states
        delta_x0 = x0 - self.xs
        delta_xref = x_target - self.xs
 
        # Set the parameter values for the OCP
        self._x_init.value = delta_x0
        self._x_ref.value = delta_xref

        # Solve the OCP
        self.ocp.solve(solver=cp.PIQP, warm_start=True)

        # Handle solver failure
        if self.ocp.status != cp.OPTIMAL and self.ocp.status != cp.OPTIMAL_INACCURATE:
            print("[MPC_x] OCP FAILED TO SOLVE!")
            # Return a safe control action (e.g., zero delta input)
            u0_delta = np.zeros(self.nu)
            x_traj_delta = np.tile(delta_x0, (self.N + 1, 1)).T
            u_traj_delta = np.zeros((self.nu, self.N))
        else:
            # Extract the first optimal control input
            u0_delta = self._u.value[:, 0]
            x_traj_delta = self._x.value
            u_traj_delta = self._u.value

        # Return the absolute control input and predicted trajectories
        u0 = u0_delta + self.us
        x_traj = x_traj_delta + np.tile(self.xs, (self.N + 1, 1)).T
        u_traj = u_traj_delta + np.tile(self.us, (self.N, 1)).T

        #################################################

        return u0, x_traj, u_traj