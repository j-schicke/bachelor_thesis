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
from Basis_Forward_Propagation import newton_euler

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


        a_acc = angular_acceleration(a_vel, prev_time, time)
        u = newton_euler(w, ct, cq, d)
        f_u = np.array([0,0, u[0]]).T
        f_a = disturbance_forces(m, vel, R, f_u)

        tau_u = np.array([u[1], u[2], u[3]]).T
        tau_a = disturbance_torques(I, a_acc, a_vel, tau_u)


