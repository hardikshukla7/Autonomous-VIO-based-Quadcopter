import numpy as np
from scipy.spatial.transform import Rotation
from math import cos, sin

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2
        self.gamma = self.k_drag / self.k_thrust

        # STUDENT CODE HERE
        self.Kp = 10*np.eye(3)
        self.Kd = 5*np.eye(3)
        self.Kr = 120*np.eye(3)
        self.Kw = 10*np.eye(3)

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        e_position = state["x"] - flat_output["x"]          #position error
        e_velocity = state["v"]       #velocity error doubt
        r_acc = flat_output["x_ddot"] - np.dot(self.Kd, e_velocity) - np.dot(self.Kp, e_position)
        F_des = self.mass*(r_acc.reshape(3,)) + np.array([ 0, 0 , self.mass * self.g]).reshape(3,)

        R = Rotation.from_quat(state['q']).as_matrix()
        b3 = np.dot(R, np.array([0, 0, 1]).T) #b3 = 3 X 1 vector
        cmd_thrust = np.dot(b3.T ,  F_des)
        b3_des = F_des/np.linalg.norm(F_des)

        si = flat_output['yaw']
        a_si = np.array([cos(si) , sin(si) , 0])
        prod = np.cross(b3_des , a_si.T)
        b2_des = prod / np.linalg.norm(prod)
        # print(b2_des)
        R_des = np.hstack((np.cross(b2_des,b3_des).reshape(3,1), b2_des.reshape(3,1), b3_des.reshape(3,1)))

        mat = (R_des.T @ R - R.T @ R_des) / 2

        e_R = np.array([mat[2, 1], mat[0, 2], mat[1, 0]]) #skew_symmetric matrix
        e_R = e_R.reshape((3,))
        val = -np.dot(self.Kr , e_R ) - np.dot(self.Kw, state["w"])
        cmd_moment = self.inertia @ val
        cmd_q = Rotation.from_matrix(R_des).as_quat()
        input = np.array([[ 1, 1, 1, 1],
                        [0, self.arm_length, 0, -self.arm_length],
                        [-self.arm_length, 0, self.arm_length, 0],
                        [self.gamma, -self.gamma, self.gamma, -self.gamma]])
        motor_thrust = np.dot(np.linalg.inv(input).reshape((4,4)), np.vstack((cmd_thrust.reshape((1,1)), cmd_moment.reshape((3,1)))))
        motor_thrust = motor_thrust.reshape((4,))
        print(motor_thrust)

        cmd_motor_speeds = np.zeros((4,))
        for i in range(4):
            if motor_thrust[i] < 0:
                cmd_motor_speeds[i] = 0
            else:
                cmd_motor_speeds[i] = np.sqrt(motor_thrust[i] / self.k_thrust)
        cmd_motor_speeds = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        ##########


        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
