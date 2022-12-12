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


def newton_euler(w, ct, cq, d):
    T = np.array([[ct, ct, ct, ct], [0, d*ct, 0, -d*ct], [-d*ct, 0, d*ct, 0], [cq, -cq, cq, -cq]])
    u = T @ w**2
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
    start_time = data['timestamp'][0]

    I = MultirotorConfig.INERTIA
    ct =MultirotorConfig.THRUST_C
    cq = MultirotorConfig.TORQUE_C
    d = MultirotorConfig.DISTANCE_M_C
    m = MultirotorConfig.MASS
    acc = []
    z_w = np.array([0,0,1])
    prev_time = 0
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
        w = np.array([data['rpm.m1'][i], data['rpm.m2'][i], data['rpm.m3'][i], data['rpm.m4'][i]]) 
        u = newton_euler(w, ct, cq, d)
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
    quaternions.pop(len(data['timestamp']))
    quaternions = np.array(quaternions)
    

    fig, ax = plt.subplots(2)
    ax[0].plot(data['timestamp'], data['stateEstimate.qw'], '-', label='w')
    ax[0].plot(data['timestamp'], data['stateEstimate.qx'], '-', label='x')
    ax[0].plot(data['timestamp'], data['stateEstimate.qy'], '-', label='y')
    ax[0].plot(data['timestamp'], data['stateEstimate.qz'], '-', label='z')

    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('quaternions')
    ax[0].legend(loc=9, ncol=3, borderaxespad=0.)

    ax[1].plot(data['timestamp'], quaternions[:,0], '-', label='w')
    ax[1].plot(data['timestamp'], quaternions[:,1], '-', label='x')
#   ax[1].plot(data['timestamp'], quaternions[:,2], '-', label='y')
#   ax[1].plot(data['timestamp'], quaternions[:,3], '-', label='z')

    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('new_quaternions')
#   ax[1].legend(loc=9, ncol=3, borderaxespad=0.)
    plt.show()

    print(quaternions[2])
    print(np.array([data['stateEstimate.qw'][2],data['stateEstimate.qx'][2], data['stateEstimate.qy'][2], data['stateEstimate.qz'][2]]))