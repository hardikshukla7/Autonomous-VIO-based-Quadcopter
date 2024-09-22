# %% Imports

import numpy as np
import scipy as sp
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


# %% Functions


def nominal_state_update(nominal_state, w_m, a_m, dt):
    p, v, q, a_b, w_b, g = nominal_state

    p = p + v * dt + 0.5 * ((q.as_matrix() @ (a_m - a_b) + g) * dt ** 2)
    v = v + (q.as_matrix() @ (a_m - a_b) + g) * dt
    dcq = Rotation.from_rotvec((w_m - w_b).reshape(-1) * dt).as_quat()
    q = q * Rotation.from_quat(dcq)

    return p, v, q, a_b, w_b, g


def error_covariance_update(nom_state, err_state_cov, w_meas, a_meas, dt,
                            acc_noise_density, gyro_noise_density,
                            acc_random_walk, gyro_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal state tuple
    pos, vel, ori, acc_bias, gyro_bias, grav = nom_state

    Q_i = np.zeros((12, 12))
    Q_i[0:3, 0:3] = (acc_noise_density ** 2) * (dt ** 2) * np.eye(3)
    Q_i[3:6, 3:6] = (gyro_noise_density ** 2) * (dt ** 2) * np.eye(3)
    Q_i[6:9, 6:9] = acc_random_walk ** 2 * dt * np.eye(3)
    Q_i[9:12, 9:12] = gyro_random_walk ** 2 * dt * np.eye(3)

    a = (a_meas - acc_bias).flatten()
    w = (w_meas - gyro_bias).flatten() * dt

    a_skew = np.array([[0, -a[2], a[1]],
                       [a[2], 0, -a[0]],
                       [-a[1], a[0], 0]])

    F_x = np.eye(18)
    F_x[:3, 3:6] = np.eye(3) * dt
    F_x[3:6, 6:9] = -ori.as_matrix() @ a_skew * dt
    F_x[3:6, 9:12] = -ori.as_matrix() * dt
    F_x[3:6, 15:18] = np.eye(3) * dt
    F_x[6:9, 6:9] = Rotation.from_rotvec(w).as_matrix().T
    F_x[6:9, 12:15] = -np.eye(3) * dt

    F_i = np.zeros((18, 12))
    F_i[3:6, :3] = np.eye(3)
    F_i[6:9, 3:6] = np.eye(3)
    F_i[9:12, 6:9] = np.eye(3)
    F_i[12:15, 9:12] = np.eye(3)

    return F_x @ err_state_cov @ F_x.T + F_i @ Q_i @ F_i.T


def measurement_update_step(nom_state, err_state_cov, uv_meas, Pw_meas, err_thresh, Q_cov):
    """
    Function to update the error state covariance matrix

    Args:
        nom_state: Tuple representing the nominal state (position, velocity, orientation, accelerometer bias, gyroscope bias, gravity)
        err_state_cov: Initial error state covariance matrix
        uv_meas: Measured image coordinates
        Pw_meas: World coordinates
        err_thresh: Threshold for innovation
        Q_cov: Covariance matrix of measurement noise

    Returns:
        Updated nominal state, error state covariance, and innovation
    """
    # Extracting components from the nominal state tuple
    pos, vel, ori, acc_bias, gyro_bias, grav = nom_state
    R = ori.as_matrix()
    p_c = R.T @ (Pw_meas - pos)
    z_c = p_c[2, 0]
    c_0 = p_c[0, 0] / z_c
    c_1 = p_c[1, 0] / z_c

    # Reshaping into a column vector
    c = np.array([c_0, c_1]).reshape(2, 1)

    innovation = uv_meas.reshape(2, 1) - c

    if np.linalg.norm(innovation) > err_thresh:
        return nom_state, err_state_cov, innovation

    p_flat = p_c.flatten()
    dpc_dt = np.array([[0, -p_flat[2], p_flat[1]],
                       [p_flat[2], 0, -p_flat[0]],
                       [-p_flat[1], p_flat[0], 0]]).reshape(3, 3)

    p_c_00 = p_c[0, 0]
    p_c_10 = p_c[1, 0]
    inv_z_c = 1 / z_c
    c_0 = -p_c_00 * inv_z_c
    c_1 = -p_c_10 * inv_z_c
    matrix = np.array([[1, 0, c_0], [0, 1, c_1]])
    dzt_dpc = inv_z_c * matrix.reshape(2, 3)

    H = np.zeros((2, 18))
    H[0:2, 0:3] = dzt_dpc @ -R.T
    H[0:2, 6:9] = dzt_dpc @ dpc_dt

    K_t = err_state_cov @ H.T @ np.linalg.inv(H @ err_state_cov @ H.T + Q_cov)
    d_x = K_t @ innovation

    pos += d_x[0:3]
    vel += d_x[3:6]
    acc_bias += d_x[9:12]
    gyro_bias += d_x[12:15]
    grav += d_x[15:18]
    q_axis_angle = ori.as_rotvec()
    del_q_axis_angle = d_x[6:9].ravel()
    new_axis_angle = q_axis_angle + del_q_axis_angle
    new_q = Rotation.from_rotvec(new_axis_angle).as_quat()
    # Update quaternion
    ori = Rotation.from_quat(new_q)
    A = np.eye(18) - (K_t @ H)
    cov_mat = (A @ err_state_cov @ A.T) + (K_t @ Q_cov @ K_t.T)
    return (pos, vel, ori, acc_bias, gyro_bias, grav), cov_mat, innovation
