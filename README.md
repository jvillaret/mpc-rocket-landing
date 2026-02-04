# MPC for Autonomous Rocket Landing (EPFL ME-425)

Implementation of Model Predictive Control (MPC) strategies to land a thrust-controlled rocket prototype (rocket-shaped drone with gimbaled propellers).  
This project follows the EPFL **ME-425 Model Predictive Control** mini-project specification and focuses on constrained control, robustness, and nonlinear simulation validation.

## Project goal
Design stabilizing and tracking controllers that guide the vehicle through:
1) **Approach phase**: reach a stable hover / controlled descent region  
2) **Landing phase**: descend safely toward a low-altitude target while satisfying constraints

The provided model is a **12-state nonlinear system** (angular rates, Euler angles, velocities, positions) with **4 inputs** (servo deflections + average/differential throttle).  
The system is linearized around trim points and decomposed into independent subsystems for MPC design (x, y, z, roll), as described in the project statement.

## Implemented controllers (high level)
- **Linear MPC** for subsystem velocity tracking and stabilization  
- **Robust MPC / Tube MPC** for disturbance rejection and constraint satisfaction in the landing phase (notably for the z dimension)  
- **Offset-free tracking** in the presence of mass mismatch (disturbance estimation)  
- **Nonlinear MPC (NMPC)** using CasADi (full-state, nonlinear dynamics)

## Constraints handled
Examples from the specification include:
- Euler angle validity bounds for linearization (e.g., |α|, |β| limited)
- Input limits (servo angles, throttle ranges)
- Safety constraints during landing (e.g., height constraint z ≥ 0)
- Additional constraints for safe operation (minimum throttle requirement)

## Repository structure
