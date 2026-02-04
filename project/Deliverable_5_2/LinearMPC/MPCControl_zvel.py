import cvxpy as cp
import numpy as np
from scipy.signal import place_poles
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])  # vz
    u_ids: np.ndarray = np.array([2])  # Pavg

    # Constraints
    u_min = 40.0
    u_max = 80.0

    def _setup_controller(self) -> None:
        #################################################
        # Disturbance model: x^+ = A x + B u + Bd d
        #                    d^+ = d
        #                    y   = C x (with C = I)
        Bd = self.B.copy()  # Disturbance enters same as input

        # Augmented system: A_hat = [A  Bd;  0  I]
        self.A_hat = np.vstack((
            np.hstack((self.A, Bd)),
            np.hstack((np.zeros((1, self.nx)), np.eye(1)))
        ))

        # B_hat = [B; 0]
        self.B_hat = np.vstack((self.B, np.zeros((1, self.nu))))

        # C_hat = [C  0] ==> C = I
        self.C_hat = np.hstack((np.eye(self.nx), np.zeros((self.nx, 1))))

        # Observer gain via pole placement
        poles = np.array([0.5, 0.6]) #faster = closer to zero
        res = place_poles(self.A_hat.T, self.C_hat.T, poles)
        self.L = -res.gain_matrix.T  

        # Observer state: [x_hat, d_hat] in DEVIATION coordinates
        self.x_hat = np.zeros(self.nx + 1)
        self.first_call = True  # Flag to initialize observer with first measurement

        # Logging for convergence analysis                                                                                                                   
        self.d_hat_history = []                                                                                                                              
        self.x_hat_history = [] 

        # MPC cost weights
        self.Q = np.diag([25.0])
        self.R = np.diag([1])

        # Compute LQR terminal cost
        _, self.Qf, _ = dlqr(self.A, self.B, self.Q, self.R)

        # MPC variables and parameters
        self.x_var = cp.Variable((self.N + 1, self.nx), name='x')
        self.u_var = cp.Variable((self.N, self.nu), name='u')

        self.xs_par = cp.Parameter(self.nx, name='xs')
        self.us_par = cp.Parameter(self.nu, name='us')
        self.x0_hat_par = cp.Parameter(self.nx, name='x0_hat')
        self.d_hat_par = cp.Parameter(1, name='d_hat')

        # Cost function
        mpc_obj = 0
        for k in range(self.N):
            mpc_obj += 0.5 * cp.quad_form(self.x_var[k] - self.xs_par, self.Q)
            mpc_obj += 0.5 * cp.quad_form(self.u_var[k] - self.us_par, self.R)

        # Terminal cost (LQR-based)
        mpc_obj += 0.5 * cp.quad_form(self.x_var[self.N] - self.xs_par, self.Qf)

        # Constraints
        mpc_cons = []
        mpc_cons.append(self.x_var[0] == self.x0_hat_par)
        mpc_cons.append(self.u_var <= self.u_max)
        mpc_cons.append(self.u_var >= self.u_min)

        # Dynamics: x^+ = A x + B (u - us_trim) + B d_hat  (u is ABSOLUTE)
        for k in range(self.N):
            mpc_cons.append(
                self.x_var[k + 1] == self.A @ self.x_var[k] + self.B @ (self.u_var[k] - self.us[0]) + self.B @ self.d_hat_par
            )

        self.ocp = cp.Problem(cp.Minimize(mpc_obj), mpc_cons)
        #################################################

    def compute_steady_state(self, d_hat: float):
        
        d_hat = np.array(d_hat).reshape((-1,))
        us_v = cp.Variable(self.nu)
        xs_v = cp.Variable(self.nx)
        
        obj = cp.Minimize(cp.quad_form(us_v, np.eye(self.nu)))

        cons = [
            us_v >= self.u_min,
            us_v <= self.u_max,
            xs_v == self.A @ xs_v + self.B @ (us_v - self.us[0]) + self.B @ d_hat,
        ]

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.PIQP)

        if prob.status != cp.OPTIMAL:
            print(f"Warning: Steady-state calculation not optimal. Status: {prob.status}.")
            return self.xs, self.us

        return xs_v.value, us_v.value
    
    def compute_steady_stateOLD(self, d_hat: float, x_target: float):
        """
        Compute steady-state xs and us (ABSOLUTE) that satisfy:
            xs = A @ xs + B @ (us - us_trim) + B @ d_hat
            xs = x_target
        """
        xs = x_target
        # Deviation steady-state input
        delta_us = (xs * (1 - self.A[0, 0]) - self.B[0, 0] * d_hat) / self.B[0, 0]
        # Convert to ABSOLUTE
        us = delta_us + self.us[0]
        us = np.clip(us, self.u_min, self.u_max)
        return np.array([xs]), np.array([us])

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # Measurement in deviation coordinates
        yk = x0 - self.xs

        # Initialize observer with first measurement
        if self.first_call:
            self.x_hat[:self.nx] = yk
            self.first_call = False

        # 1. Get CURRENT estimates (before update)
        x_hat_k = self.x_hat[:self.nx] + self.xs  # Convert to absolute
        d_hat_k = self.x_hat[self.nx:]

        # Log for convergence analysis
        self.d_hat_history.append(d_hat_k[0])
        self.x_hat_history.append(x_hat_k[0])

        # 2. Compute steady-state target using current d_hat
        xs, us = self.compute_steady_state(d_hat_k[0])

        # 3. Solve MPC using current estimates
        self.xs_par.value = xs
        self.us_par.value = us
        self.x0_hat_par.value = x_hat_k
        self.d_hat_par.value = d_hat_k

        self.ocp.solve(solver=cp.PIQP)

        # 4. Extract control
        if self.ocp.status == cp.OPTIMAL and self.u_var.value is not None:
            u0 = self.u_var[0].value
            x_traj = self.x_var.value.T
            u_traj = self.u_var.value.T
        else:
            u0 = us
            x_traj = np.tile(xs[:, None], (1, self.N + 1))
            u_traj = np.tile(us[:, None], (1, self.N))

        # 5. NOW update observer for next iteration (using current u and y)
        uk = u0 - self.us  # Current input in deviation
        x_hat_next = self.A_hat @ self.x_hat + self.B_hat.flatten() * uk[0] \
                     + self.L.flatten() * (self.C_hat @ self.x_hat - yk)
        self.x_hat = x_hat_next
        #################################################

        return u0, x_traj, u_traj