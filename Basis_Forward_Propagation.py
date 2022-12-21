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
from plot_data import compare_data, errors
from sklearn import preprocessing



#def newton_euler(w, ct, cq, d):
#    T = np.array([[ct, ct, ct, ct], [0, d*ct, 0, -d*ct], [-d*ct, 0, d*ct, 0], [cq, -cq, cq, -cq]])
#    u = T @ w**2
#    return u
def thrust_torque(pwm_1, pwm_2, pwm_3, pwm_4, mv):


    f_1 = (11.09-39.08*pwm_1-9.53*mv +20.57*pwm_1**2 + 38.43*pwm_1*mv)*0.0980665
    f_2 = (11.09-39.08*pwm_2-9.53*mv +20.57*pwm_2**2 + 38.43*pwm_2*mv)*0.0980665
    f_3 = (11.09-39.08*pwm_3-9.53*mv +20.57*pwm_3**2 + 38.43*pwm_3*mv)*0.0980665
    f_4 = (11.09-39.08*pwm_4-9.53*mv +20.57*pwm_4**2 + 38.43*pwm_4*mv)*0.0980665
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
    
    return acc*0.101972

def velocity(acc, vel, time, prev_time):
    dt = (time - prev_time)* 0.001
    v = vel + acc*dt
    return v

def position(vel, pos, time, prev_time):
    dt = (time - prev_time)*0.001
    y = pos + vel*dt
    return y

def angular_acc(u, wbw, I):
    wbw_2 = np.linalg.pinv(I)@(np.cross(-wbw,I)@wbw+np.array([u[1], u[2], u[3]]))
    return wbw_2

def angular_velocity(acc_a, vel_a, time, prev_time):
    dt = (time - prev_time)*0.001
    v = vel_a + acc_a*dt
    return v

def new_quaternion(v, q, time, prev_time):
    dt = (time - prev_time)*0.001
    quaternion = rowan.calculus.integrate(q, v, dt)
    return quaternion

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("file_usd")
    #args = parser.parse_args()
    #data_usd = cfusdlog.decode(args.file_usd)
    data_usd = cfusdlog.decode("hardware/data/jana02")
    data = data_usd['fixedFrequency']

    I = MultirotorConfig.INERTIA
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
    pwm_1 = preprocessing.normalize(data['pwm.m1_pwm'][None])[0]
    pwm_2 = preprocessing.normalize(data['pwm.m2_pwm'][None])[0]
    pwm_3 = preprocessing.normalize(data['pwm.m3_pwm'][None])[0]
    pwm_4 =preprocessing.normalize(data['pwm.m4_pwm'][None])[0]
    mv = preprocessing.normalize(data['pm.vbatMV'][None])[0]
    
    # vel_a.append(np.array([data['gyro.x'][0], data['gyro.y'][0], data['gyro.z'][0]]))
    # vel.append(np.array([data['stateEstimate.vx'][0], data['stateEstimate.vy'][0], data['stateEstimate.vz'][0]]))
    # pos.append(np.array([data['stateEstimate.x'][0], data['stateEstimate.y'][0], data['stateEstimate.z'][0]]))
    # quaternions.append(np.array([data['stateEstimate.qw'][0],data['stateEstimate.qx'][0], data['stateEstimate.qy'][0], data['stateEstimate.qz'][0]]))

    for i in range(len(data['timestamp'])):
        R = rowan.to_matrix(np.array([data['stateEstimate.qw'][i],data['stateEstimate.qx'][i], data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]]))
        #R = rowan.to_matrix(quaternions[i])
        z_b= np.asarray(R@z_w) 
        u = thrust_torque(pwm_1[i], pwm_2[i], pwm_3[i], pwm_4[i], mv[i])

        time = data['timestamp'][i]

        # ang_a = angular_acc(u, vel_a[i], I)
        ang_a = angular_acc(u, np.array([data['gyro.x'][i], data['gyro.y'][i], data['gyro.z'][i]]), I)

        acc_a.append(ang_a)

        a = acceleration(m ,u, z_w, z_b)
        acc.append(a)

        # v = velocity(acc[i], vel[i], time, prev_time)
        v = velocity(acc[i], np.array([data['stateEstimate.vx'][i], data['stateEstimate.vy'][i], data['stateEstimate.vz'][i]]), time, prev_time)
        vel.append(v)

        # y = position(vel[i], pos[i], time, prev_time)
        y = position(vel[i], np.array([data['stateEstimate.x'][i], data['stateEstimate.y'][i], data['stateEstimate.z'][i]]), time, prev_time)
        pos.append(y)

        # v_a = angular_velocity(acc_a[i], vel_a[i] ,time, prev_time)
        v_a = angular_velocity(acc_a[i], np.array([data['gyro.x'][i], data['gyro.y'][i], data['gyro.z'][i]]) ,time, prev_time)
        vel_a.append(v_a)

        #quat = new_quaternion(vel_a[i], quaternions[i], time, prev_time)
        quat = new_quaternion(vel_a[i], np.array([data['stateEstimate.qw'][i],data['stateEstimate.qx'][i], data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]]), time, prev_time)
        quaternions.append(quat)

        prev_time = time

        

    acc = np.array(acc)

    vel = np.array(vel)


    vel_a = np.array(vel_a)

    pos = np.array(pos)

    quaternions = np.array(quaternions)

    pos = np.array(pos)

    compare_data(data, quaternions, acc, vel, vel_a)
    errors(data, acc, vel, vel_a, quaternions, pos)

