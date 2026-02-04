import numpy as np
import casadi as ca
from typing import Tuple
# from control import dlqr  # Not needed for simple version


class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """

    def __init__(self, rocket, H, xs, us):
        """
        Initialize NMPC controller.
        
        Args:
            rocket: Rocket object with dynamics
            H: Horizon time in seconds
            xs: Steady-state (reference) state vector (12,) - this is the TARGET state
            us: Steady-state input vector (4,) - this is the TARGET input
        """
        self.rocket = rocket
        self.H = H
        self.xs = xs
        self.us = us
        self.Ts = rocket.Ts
        self.N = int(H / self.Ts)  # Prediction horizon length
        
        # Symbolic dynamics f(x,u) from rocket
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]
        
        # Set controller type for constraint checking
        rocket.controller_type = 'NmpcCtrl'
        
        # No linearization needed for simple version
        # (Removed LQR terminal cost to simplify)
        
        # Setup the optimization problem
        self._setup_controller()
    
    def rk4(self, x, u, h):
        """
        Runge-Kutta 4th order discretization.
        
        Args:
            x: State vector (CasADi SX)
            u: Input vector (CasADi SX)
            h: Time step (scalar)
            
        Returns:
            Next state x_{k+1}
        """
        k1 = h * self.f(x, u)
        k2 = h * self.f(x + k1/2, u)
        k3 = h * self.f(x + k2/2, u)
        k4 = h * self.f(x + k3, u)
        return x + (k1 + 2*k2 + 2*k3 + k4) / 6

    def _setup_controller(self) -> None:
        """
        Set up the CasADi optimization problem for NMPC.
        """
        # Create optimization problem
        opti = ca.Opti()
        
        # Decision variables
        X = opti.variable(12, self.N + 1)  # State trajectory (12 states, N+1 steps)
        U = opti.variable(4, self.N)       # Input trajectory (4 inputs, N steps)
        
        # Parameters
        X0_param = opti.parameter(12)      # Initial state
        xs_param = opti.parameter(12)      # Equilibrium state (reference/target)
        us_param = opti.parameter(4)       # Equilibrium input (CRITICAL: balances gravity)
        
        # Weight matrices for cost function - TUNED FOR FAST SETTLING (≤4s)
        # Track to (xs, us) equilibrium pair
        # Increased position weights for faster convergence to target
        Q = np.diag([1.0,   1.0,   1.0,     # wx, wy, wz (angular velocities)
                     10.0,  10.0,  10.0,    # alpha, beta, gamma (attitude)
                     2.0,   2.0,   2.0,     # vx, vy, vz (velocities - increased for faster movement)
                     300.0, 300.0, 500.0])  # x, y, z (positions - significantly increased for faster settling)
        
        # R: Penalize deviation from equilibrium input us
        # Reduced R to allow more aggressive control for faster settling
        R = np.diag([0.5, 0.5, 0.3, 0.5])  # dR, dP, Pavg (lower), Pdiff
        
        # Extract diagonal elements for efficient computation (Q, R are diagonal)
        Q_diag = np.diag(Q)
        R_diag = np.diag(R)
        
        # Convert to CasADi
        Q_diag_ca = ca.DM(Q_diag)
        R_diag_ca = ca.DM(R_diag)
        
        # Cost function - SIMPLE QUADRATIC
        cost = 0.0
        
        # Stage cost: Track equilibrium (xs, us)
        for k in range(self.N):
            # State cost: (x - xs)^T Q (x - xs)
            dx = X[:, k] - xs_param
            cost += ca.sum1(Q_diag_ca * (dx**2))
            
            # Input cost: (u - us)^T R (u - us) 
            # us is the equilibrium input that balances gravity
            du = U[:, k] - us_param
            cost += ca.sum1(R_diag_ca * (du**2))
        
        # Terminal cost: Simple scaled Q (no LQR complexity)
        # Increased terminal cost scaling for faster convergence to target state
        dx_term = X[:, self.N] - xs_param
        cost += ca.sum1(Q_diag_ca * (dx_term**2)) * 4.0  # Scale by 4 for strong terminal emphasis
        
        # CRITICAL: Register the cost as the objective to minimize
        opti.minimize(cost)
        
        # Constraints
        
        # Initial condition
        opti.subject_to(X[:, 0] == X0_param)
        
        # Dynamics constraints (RK4 discretization)
        for k in range(self.N):
            opti.subject_to(X[:, k+1] == self.rk4(X[:, k], U[:, k], self.Ts))
        
        # Input constraints (from constraints (9) mentioned in task)
        # dR, dP: ±15 degrees
        dR_dP_max = np.deg2rad(15)
        opti.subject_to(U[0, :] >= -dR_dP_max)  # dR lower bound
        opti.subject_to(U[0, :] <= dR_dP_max)   # dR upper bound
        opti.subject_to(U[1, :] >= -dR_dP_max)  # dP lower bound
        opti.subject_to(U[1, :] <= dR_dP_max)   # dP upper bound
        
        # Pavg: [10, 90]
        opti.subject_to(U[2, :] >= 10.0)   # Pavg lower bound
        opti.subject_to(U[2, :] <= 90.0)   # Pavg upper bound
        
        # Pdiff: [-20, 20]
        opti.subject_to(U[3, :] >= -20.0)  # Pdiff lower bound
        opti.subject_to(U[3, :] <= 20.0)   # Pdiff upper bound
        
        # State constraints
        # beta (index 4): |β| ≤ 80° (task specifies safe numerical values, not 85°)
        beta_max = np.deg2rad(80)
        opti.subject_to(X[4, :] >= -beta_max)  # beta lower bound
        opti.subject_to(X[4, :] <= beta_max)   # beta upper bound
        
        # z (index 11): z >= 0 (rocket can't go below ground)
        opti.subject_to(X[11, :] >= 0.0)  # z lower bound
        
        # Set initial guess - simple: use trim state/input
        # (Warm start handled in get_u, but here just use trim)
        opti.set_initial(X, np.tile(self.xs.reshape(-1, 1), (1, self.N + 1)))
        opti.set_initial(U, np.tile(self.us.reshape(-1, 1), (1, self.N)))
        
        # Solver options (with expand=True for faster computation as suggested)
        options = {
            "expand": True,  # Expand CasADi expressions for faster computation
            "print_time": False,
            "ipopt": {
                "sb": "yes",
                "print_level": 0,
                "tol": 1e-3,
                "max_iter": 500  # Increase max iterations for better convergence
            }
        }
        opti.solver("ipopt", options)
        
        # Store optimization problem and variables
        self.ocp = opti
        self.X = X
        self.U = U
        self.X0_param = X0_param
        self.xs_param = xs_param
        self.us_param = us_param
        
        # Store previous solution for warm start
        self.last_x_sol = None
        self.last_u_sol = None

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the MPC control input and trajectories.
        
        Args:
            t0: Initial time
            x0: Initial state vector (12,)
            
        Returns:
            u0: First control input (4,)
            x_ol: Open-loop state trajectory (12, N+1)
            u_ol: Open-loop input trajectory (4, N)
            t_ol: Time trajectory (N+1,)
        """
        # Set parameter values
        self.ocp.set_value(self.X0_param, x0)
        self.ocp.set_value(self.xs_param, self.xs)  # Equilibrium state
        self.ocp.set_value(self.us_param, self.us)  # Equilibrium input (balances gravity)
        
        # Warm start: use previous solution if available
        if self.last_x_sol is not None and self.last_u_sol is not None:
            # Shift previous solution forward (receding horizon)
            x_warm = np.column_stack([
                self.last_x_sol[:, 1:],  # Shift states forward
                self.last_x_sol[:, -1:]  # Repeat last state
            ])
            u_warm = np.column_stack([
                self.last_u_sol[:, 1:],  # Shift inputs forward
                self.last_u_sol[:, -1:]  # Repeat last input
            ])
            self.ocp.set_initial(self.X, x_warm)
            self.ocp.set_initial(self.U, u_warm)
        else:
            # First iteration: use current state and transition to target
            x_init = np.zeros((12, self.N + 1))
            x_init[:, 0] = x0
            # Gradually transition from x0 to xs for smoother initial guess
            for k in range(1, self.N + 1):
                alpha = k / self.N
                x_init[:, k] = (1 - alpha) * x0 + alpha * self.xs
            self.ocp.set_initial(self.X, x_init)
            self.ocp.set_initial(self.U, np.tile(self.us.reshape(-1, 1), (1, self.N)))
        
        # Solve the optimization problem
        try:
            sol = self.ocp.solve()
            
            # Extract solution
            # X is (12, N+1) and U is (4, N) - no transpose needed
            x_ol = sol.value(self.X)  # Shape: (12, N+1)
            u_ol = sol.value(self.U)  # Shape: (4, N)
            u0 = u_ol[:, 0]  # First input, shape: (4,)
            
            # Store solution for warm start in next iteration
            self.last_x_sol = x_ol.copy()
            self.last_u_sol = u_ol.copy()
            
            # Time trajectory
            t_ol = np.linspace(t0, t0 + self.H, self.N + 1)
            
            return u0, x_ol, u_ol, t_ol
            
        except Exception as e:
            # If solver fails, return last known solution or default
            print(f"MPC solver failed: {e}")
            # Return trim input as fallback
            u0 = self.us.copy()
            # Predict forward using trim input
            x_ol = np.zeros((12, self.N + 1))
            x_ol[:, 0] = x0
            for k in range(self.N):
                x_ol[:, k+1] = x_ol[:, k]  # Simplified - just repeat state
            u_ol = np.tile(self.us.reshape(-1, 1), (1, self.N))
            t_ol = np.linspace(t0, t0 + self.H, self.N + 1)
            return u0, x_ol, u_ol, t_ol
