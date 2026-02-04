import numpy as np
import cvxpy as cp

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:
        #################################################
        # Define variables
        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.N))
        s_roll = cp.Variable(self.N + 1, nonneg=True)

        x_init = cp.Parameter(self.nx)
        x_ref = cp.Parameter(self.nx)

        # Tuning
        # Q: Penalize w_z and gamma. R: Penalize P_diff effort.
        Q = np.diag([1, 100.0])  
        R = np.diag([1.0])       
        S = 1e6 # High penalty for slack 

        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(x[:, k] - x_ref, Q) + cp.quad_form(u[:, k], R)
            cost += S * s_roll[k]
        cost += cp.quad_form(x[:, self.N] - x_ref, Q)
        cost += S * s_roll[self.N]

        constraints = [x[:, 0] == x_init]

        for k in range(self.N):
            # System dynamics [cite: 167, 168]
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]

            # Hard input constraints: |P_diff| <= 20 
            # u is delta_u, so: -20 <= delta_u + us <= 20
            constraints += [u[:, k] <= 20.0 - self.us]
            constraints += [u[:, k] >= -20.0 - self.us]

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        
        self._x, self._u, self._s_roll = x, u, s_roll
        self._x_init, self._x_ref = x_init, x_ref
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        delta_x0 = x0 - self.xs
        delta_xref = x_target - self.xs
 
        self._x_init.value = delta_x0
        self._x_ref.value = delta_xref

        # Solve optimization problem
        self.ocp.solve(solver=cp.PIQP, warm_start=True)

        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            u0_delta = np.zeros(self.nu)
            x_traj_delta = np.tile(delta_x0, (self.N + 1, 1)).T
            u_traj_delta = np.zeros((self.nu, self.N))
        else:
            u0_delta = self._u.value[:, 0]
            x_traj_delta = self._x.value
            u_traj_delta = self._u.value

        # Shift back to absolute coordinates
        u0 = u0_delta + self.us
        x_traj = x_traj_delta + np.tile(self.xs, (self.N + 1, 1)).T
        u_traj = u_traj_delta + np.tile(self.us, (self.N, 1)).T
        #################################################

        return u0, x_traj, u_traj
