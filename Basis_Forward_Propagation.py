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
from mpl_toolkits.mplot3d import Axes3D
from plot_data import compare_data, errors, rpm


#def newton_euler(w, ct, cq, d):
#    T = np.array([[ct, ct, ct, ct], [0, d*ct, 0, -d*ct], [-d*ct, 0, d*ct, 0], [cq, -cq, cq, -cq]])
#    u = T @ w**2
#    return u
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


def acceleration(m, u, z_w, z_b):
    g = 9.81
    acc = (-m*g*z_w+u[0]*z_b)/m
    return acc


def velocity(acc, vel, time, prev_time):
    dt = time - prev_time
    v = vel + acc*dt
    return v

def position(vel, pos, time, prev_time):
    dt = time - prev_time
    y = pos + vel*dt
    return y

def angular_acc(u, wbw, I):
    wbw_2 = np.linalg.pinv(I)@(np.cross(-wbw,I)@wbw+np.array([u[1], u[2], u[3]]))
    return wbw_2

def angular_velocity(acc_a, vel_a, time, prev_time):
    dt = time - prev_time
    v = vel_a + acc_a*dt
    return v

def new_quaternion(v, q, time, prev_time):
    dt = time - prev_time
    quaternion = rowan.calculus.integrate(q, v, dt)
    return quaternion

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("file_usd")
    #args = parser.parse_args()
    #data_usd = cfusdlog.decode(args.file_usd)
    data_usd = cfusdlog.decode("log01")
    data = data_usd['fixedFrequency']

    I = MultirotorConfig.INERTIA
    ct =MultirotorConfig.THRUST_C
    cq = MultirotorConfig.TORQUE_C
    d = MultirotorConfig.DISTANCE_M_C
    m = MultirotorConfig.MASS
    acc = []
    z_w = np.array([0,0,1])
    prev_time = data['timestamp'][0]
    vel_a = []
    acc_a = []
    pos = []
    vel = []
    acc = []
    quaternions = []

    vel_a.append(np.array([data['gyro.x'][0], data['gyro.y'][0], data['gyro.z'][0]]))
    vel.append(np.array([data['stateEstimate.vx'][0], data['stateEstimate.vy'][0], data['stateEstimate.vz'][0]]))
    pos.append(np.array([data['stateEstimate.x'][0], data['stateEstimate.y'][0], data['stateEstimate.z'][0]]))
    quaternions.append(np.array([data['stateEstimate.qw'][0],data['stateEstimate.qx'][0], data['stateEstimate.qy'][0], data['stateEstimate.qz'][0]]))

    for i in range(len(data['timestamp'])):
        R = rowan.to_matrix(quaternions[i])
        z_b= np.asarray(R@z_w) 
        #w = np.array([data['rpm.m1'][i], data['rpm.m2'][i], data['rpm.m3'][i], data['rpm.m4'][i]]) 
        u = thrust_torque(data['pwm.m1_pwm'][i], data['pwm.m2_pwm'][i], data['pwm.m3_pwm'][i], data['pwm.m4_pwm'][i], data['pm.vbatMV'][i])

        time = data['timestamp'][i]

        ang_a = angular_acc(u, vel_a[i], I)
        acc_a.append(ang_a)

        a = acceleration(m ,u, z_w, z_b)  
        acc.append(a)

        v = velocity(acc[i], vel[i], time, prev_time)
        vel.append(v)

        y = position(vel[i], pos[i], time, prev_time)
        pos.append(y)

        v_a = angular_velocity(acc_a[i], vel_a[i] ,time, prev_time)
        vel_a.append(v_a)

        quat = new_quaternion(vel_a[i], quaternions[i], time, prev_time)
        quaternions.append(quat)

        prev_time = time

    acc = np.array(acc)

    vel.pop(0)
    vel = np.array(vel)

    vel_a.pop(0)
    vel_a = np.array(vel_a)

    quaternions.pop(0)
    quaternions = np.array(quaternions)

    pos.pop(0)
    pos = np.array(pos)

    compare_data(data, quaternions, acc, vel, vel_a)
    errors(data, acc, vel, vel_a, quaternions, pos)

