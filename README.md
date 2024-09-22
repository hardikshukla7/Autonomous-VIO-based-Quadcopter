# Autonomous VIO-based Quadcopter

## Project Overview

This project features a quadcopter controlled using a geometric nonlinear PD controller, designed to align the quadcopter's axis with the desired direction for agile maneuvers. To facilitate waypoint creation and enable autonomous navigation from the source to the target, I implemented the A* Algorithm, optimized with the RDP algorithm. The quadcopter follows a minimal jerk trajectory to enhance stability and reduce overshoot.

## State Estimation

The quadcopter's state is updated using an Extended Kalman Filter (EKF), which combines predictions from IMU sensors with observations from a stereo camera. The EKF process consists of three main steps:

1. **Nominal State Update**: Updates the nominal state of the quadcopter using measured angular velocity and linear acceleration, employing kinematic equations to propagate position, velocity, and orientation over time.

2. **Error Covariance Update**: Updates the error state covariance matrix based on the current nominal state and sensor measurements. It implements the process noise model and state transition matrix to propagate the error covariance matrix forward in time.

3. **Measurement Update**: Corrects the nominal state estimate using sensor measurements. It calculates the innovation as the difference between predicted and actual measurements. If the innovation is within a specified threshold, it computes the Kalman gain to update the state estimate and error covariance matrix based on the measurement residual and Jacobian. If the innovation exceeds the threshold, the update is rejected.

### Quadcopter in Action
![Quadcopter Movement](https://github.com/hardikshukla7/Autonomous-VIO-based-Quadcopter/blob/main/over_under.mp4?raw=true)
![Quadcopter Movement](https://github.com/hardikshukla7/Autonomous-VIO-based-Quadcopter/blob/main/maze.mp4?raw=true)

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries (e.g., NumPy, OpenCV, etc.)


