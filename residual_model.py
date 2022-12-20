import numpy as np 
import cfusdlog
import matplotlib.pyplot as plt
import argparse
import numpy as np
import mplcursors
import functools
import math
from config.multirotor_config import MultirotorConfig
import rowan

def thrust_torque(pwm_1, pwm_2, pwm_3, pwm_4, mv):
    f_1 = 11.09-39.08*pwm_1-9.53*mv +20.57*pwm_1**2 + 38.43*pwm_1*mv
    f_2 = 11.09-39.08*pwm_2-9.53*mv +20.57*pwm_2**2 + 38.43*pwm_2*mv
    f_3 = 11.09-39.08*pwm_3-9.53*mv +20.57*pwm_3**2 + 38.43*pwm_3*mv
    f_4 = 11.09-39.08*pwm_4-9.53*mv +20.57*pwm_4**2 + 38.43*pwm_4*mv
    arm_length = 0.046 # m
    arm = 0.707106781 * arm_length
    t2t = 0.006 # thrust-to-torque ratio
    B0 = np.array([
			[1, 1, 1, 1],
			[-arm, -arm, arm, arm],
			[-arm, arm, arm, -arm],
			[-t2t, t2t, -t2t, t2t]
			])

    u = B0 @ np.array([f_1, f_2, f_3, f_4])
    return u 

def angular_acceleration(a_vel, prev_time, time):
    t = time - prev_time
    a_acc = a_vel/t
    return a_acc

def disturbance_forces(m, vel, R, f_u):
    g = np.array([0,0,9.81]).T
    f_a = m*vel - m*g - R*f_u
    return f_a


def disturbance_torques(I, a_acc, a_vel, tau_u):
    tau_a = I@a_acc - np.cross(I@a_vel, a_vel) - tau_u
    return tau_a

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("file_usd")
    #args = parser.parse_args()
    #data_usd = cfusdlog.decode(args.file_usd)
    data_usd = cfusdlog.decode("log01")
    data = data_usd['fixedFrequency']
    start_time = data['timestamp'][0]
    I = MultirotorConfig.INERTIA
    ct =MultirotorConfig.THRUST_C
    cq = MultirotorConfig.TORQUE_C
    d = MultirotorConfig.DISTANCE_M_C
    m = MultirotorConfig.MASS

    prev_time = start_time

    for i in range(1, len(data['timestamp'])):
        time = data['timestamp'][i]
        a_vel = np.array([data['gyro.x'][i], data['gyro.y'][i], data['gyro.z'][i]])
        vel = np.array([data['stateEstimate.vx'][i], data['stateEstimate.vy'][i], data['stateEstimate.vz'][i]])
        w = np.array([data['rpm.m1'][i], data['rpm.m2'][i], data['rpm.m3'][i], data['rpm.m4'][i]]) 
        R = rowan.to_matrix(np.array([data['stateEstimate.qw'][i],data['stateEstimate.qx'][i], data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]]))
        u = thrust_torque(data['pwm.m1_pwm'][i], data['pwm.m2_pwm'][i], data['pwm.m3_pwm'][i], data['pwm.m4_pwm'][i], data['pm.vbatMV'][i])
        a_acc = angular_acceleration(a_vel, prev_time, time)
        f_u = np.array([0,0, u[0]])
        f_a = disturbance_forces(m, vel, R, f_u)

        tau_u = np.array([u[1], u[2], u[3]])
        tau_a = disturbance_torques(I, a_acc, a_vel, tau_u)


