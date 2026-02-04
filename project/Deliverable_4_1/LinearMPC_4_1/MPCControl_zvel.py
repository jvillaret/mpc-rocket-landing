
import cvxpy as cp
import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # 1. Define Decision Variables
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))

        # 2. Define Parameters
        self.x_init = cp.Parameter(self.nx)
        self.x_target = cp.Parameter((self.nx, self.N + 1))
        self.u_target = cp.Parameter((self.nu, self.N))
      
        # 3. Define Weights and Dynamics
        Q = np.diag([25.0])  # Weight for [vz]
        R = np.diag([1.0])   # Weight for input [Pavg]

        # Affine term for absolute dynamics
        affine_term = self.xs - self.A @ self.xs - self.B @ self.us

        cost = 0.0
        constraints = [self.x_var[:, 0] == self.x_init]

        for k in range(self.N):
            # Cost
            cost += cp.quad_form(self.x_var[:, k] - self.x_target[:, k], Q)
            cost += cp.quad_form(self.u_var[:, k] - self.u_target[:, k], R)

            # Dynamics constraint
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k] + affine_term 
            )

            # Input constraints (Pavg between 40 and 80)
            constraints.append(self.u_var[:, k] <= 80.0)
            constraints.append(self.u_var[:, k] >= 40.0)

        # Terminal cost
        cost += cp.quad_form(self.x_var[:, self.N] - self.x_target[:, self.N], Q)

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
                self.x_target.value = np.tile(x_target[:, None], (1, self.N + 1))
            else:
                self.x_target.value = x_target
        else:
            self.x_target.value = np.tile(self.xs[:, None], (1, self.N + 1))

        if u_target is not None:
            if u_target.ndim == 1:
                self.u_target.value = np.tile(u_target[:, None], (1, self.N))
            else:
                self.u_target.value = u_target
        else:
            self.u_target.value = np.tile(self.us[:, None], (1, self.N))

        # 2. Solve
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

  
