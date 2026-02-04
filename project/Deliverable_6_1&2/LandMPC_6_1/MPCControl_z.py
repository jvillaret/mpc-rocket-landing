import numpy as np
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    d_estimate: np.ndarray
    d_gain: float

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        
        # Q and R for LQR gain computation
        Q = np.diag([55.0, 110.0])  # [vz, z]
        R = np.array([[1.8]])  # [Pavg]

        # Q and R for LQR gain computation
        #Q = np.diag([60.0, 160.0])  # [vz, z]
        #R = np.array([[0.68]])  # [Pavg]
        
        # Compute LQR gain K for tube controller
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K  # dlqr returns -K
        self.K = K
        self.Qf = Qf
        
        # Affine term for absolute dynamics: accounts for linearization point
        # x_next = A*(x-xs) + B*(u-us) + xs = A*x + B*u + (xs - A*xs - B*us)
        affine_term = self.xs - self.A @ self.xs - self.B @ self.us
        
        # Tube MPC variables: nominal trajectory z and nominal control v
        # Actual state x = z + e, actual control u = v + K*e
        # Use column-major format like other controllers: (nx, N+1) and (nu, N)
        self.z_var = cp.Variable((self.nx, self.N + 1))  # (nx, N+1)
        self.v_var = cp.Variable((self.nu, self.N))  # (nu, N)
        self.x_init = cp.Parameter(self.nx)
        
        # Cost function: penalize deviation from reference
        cost = 0.0
        for k in range(self.N):
            cost += cp.quad_form(self.z_var[:, k] - self.xs, Q)
            cost += cp.quad_form(self.v_var[:, k] - self.us, R)
        cost += cp.quad_form(self.z_var[:, self.N] - self.xs, self.Qf)
        
        # Constraints
        constraints = []
        
        # Initial condition: z0 = x0 (no initial error for simplicity)
        constraints.append(self.z_var[:, 0] == self.x_init)
        
        # Dynamics: z[k+1] = A @ z[k] + B @ v[k] + affine_term
        for k in range(self.N):
            constraints.append(
                self.z_var[:, k + 1] == self.A @ self.z_var[:, k] + self.B @ self.v_var[:, k] + affine_term
            )
        
        # State constraints: z[k] ≥ 0 (original, no tightening)
        # z is the second state (index 1)
        for k in range(self.N + 1):
            constraints.append(self.z_var[1, k] >= 0.0)  # z ≥ 0
        
        # Input constraints: 40 ≤ v[k] ≤ 80 (original, no tightening)
        for k in range(self.N):
            constraints.append(self.v_var[0, k] >= 40.0)
            constraints.append(self.v_var[0, k] <= 80.0)
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        
        # Set initial condition
        self.x_init.value = x0
        
        # Solve optimization problem
        try:
            self.ocp.solve(solver=cp.PIQP, warm_start=True, verbose=False)
        except:
            # Fallback to OSQP if PIQP fails
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5)
        
        if self.ocp.status == cp.OPTIMAL and self.z_var.value is not None:
            z0 = self.z_var[:, 0].value
            v0 = self.v_var[:, 0].value
            
            # Compute actual control: u = v + K*(x0 - z0)
            # Since z0 = x0 initially, e0 = 0, so u0 = v0
            e0 = x0 - z0
            u0 = v0 + self.K @ e0
            
            # Clip control to ensure it satisfies original constraints
            u0 = np.clip(u0, 40.0, 80.0)
            
            # Return trajectories (already in column-major format)
            x_traj = self.z_var.value  # (nx, N+1)
            u_traj = np.zeros((self.nu, self.N))
            for k in range(self.N):
                if k == 0:
                    ek = e0
                else:
                    # Approximate error evolution (for trajectory visualization)
                    ek = (self.A + self.B @ self.K) @ ek
                vk = self.v_var[:, k].value
                uk = vk + self.K @ ek
                u_traj[:, k] = np.clip(uk, 40.0, 80.0)
        else:
            print(f"MPC Solver failed with status: {self.ocp.status}, returning trim values.")
            u0 = self.us
            x_traj = np.tile(self.xs[:, None], (1, self.N + 1))
            u_traj = np.tile(self.us[:, None], (1, self.N))
        
        return u0, x_traj, u_traj

        # YOUR CODE HERE
        #################################################

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE

        self.d_estimate = ...
        self.d_gain = ...

        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        self.d_estimate = ...
        # YOUR CODE HERE
        ##################################################
