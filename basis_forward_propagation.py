import numpy as np 
import cfusdlog
import argparse
import numpy as np
from config.multirotor_config import MultirotorConfig
import rowan
from plot_data import compare_data, errors, trajectory_x_y_plane, trajectory
from sklearn import preprocessing

ms2s = MultirotorConfig.ms2s
g = MultirotorConfig.GRAVITATION
r2d = MultirotorConfig.rad2deg
d2r = MultirotorConfig.deg2rad
I = MultirotorConfig.INERTIA
m = MultirotorConfig.MASS
ms2g = MultirotorConfig.ms2g


def decode_data(path):
    data_usd = cfusdlog.decode(path)
    data = data_usd['fixedFrequency']
    return data


def thrust_torque(pwm_1, pwm_2, pwm_3, pwm_4, mv):
    g2N = MultirotorConfig.g2N
    f_1 = (11.09-39.08*pwm_1-9.53*mv +20.57*pwm_1**2 + 38.43*pwm_1*mv)*g2N
    f_2 = (11.09-39.08*pwm_2-9.53*mv +20.57*pwm_2**2 + 38.43*pwm_2*mv)*g2N
    f_3 = (11.09-39.08*pwm_3-9.53*mv +20.57*pwm_3**2 + 38.43*pwm_3*mv)*g2N
    f_4 = (11.09-39.08*pwm_4-9.53*mv +20.57*pwm_4**2 + 38.43*pwm_4*mv)*g2N
    arm = MultirotorConfig.ARM
    t2t =MultirotorConfig.t2t
    B0 = np.array([
			[1, 1, 1, 1],
			[-arm, -arm, arm, arm],
			[-arm, arm, arm, -arm],
			[-t2t, t2t, -t2t, t2t]
			])

    u = B0 @ np.array([f_1, f_2, f_3, f_4])
    return u 

def acceleration(u, z_w, z_b):
    g = MultirotorConfig.GRAVITATION
    acc = (-m*g*z_w+u[0]*z_b)/m 
    return acc + np.array([0,0,g])
    
def velocity(acc, vel, time, prev_time):
    dt = (time - prev_time)* ms2s
    v = vel + acc*dt
    return v
    
def position(vel, pos, time, prev_time):
    dt = (time - prev_time)*ms2s
    y = pos + vel*dt
    return y

def angular_acc(u, wbw):
    wbw_2 = np.linalg.pinv(I)@(np.cross(-wbw,I)@wbw+np.array([u[1], u[2], u[3]]))
    return wbw_2

def angular_velocity(acc_a, vel_a, time, prev_time):
    dt = (time - prev_time)*ms2s
    v = vel_a + acc_a*dt
    return v

def new_quaternion(v, q, time, prev_time):
    dt = (time - prev_time)*ms2s
    quaternion = rowan.calculus.integrate(q, v, dt)
    return quaternion

def propagate(data,name):

    pwm_1 = preprocessing.normalize(data['pwm.m1_pwm'][None])[0]
    pwm_2 = preprocessing.normalize(data['pwm.m2_pwm'][None])[0]
    pwm_3 = preprocessing.normalize(data['pwm.m3_pwm'][None])[0]
    pwm_4 =preprocessing.normalize(data['pwm.m4_pwm'][None])[0]
    mv = preprocessing.normalize(data['pm.vbatMV'][None])[0]

    z_w = np.array([0,0,1])
    prev_time = data['timestamp'][0]

    vel_a = []
    err_vel_a = []
    acc_a = []
    pos = []
    err_pos = []
    vel = []
    err_vel = []
    acc = []
    err_acc = []
    quaternions = []
    err_quaternions = []

    acc.append(np.array([data['acc.x'][0], data['acc.y'][0], data['acc.z'][0]])*g)
    vel_a.append(np.array([data['gyro.x'][0], data['gyro.y'][0], data['gyro.z'][0]])*d2r)
    vel.append(np.array([data['stateEstimate.vx'][0], data['stateEstimate.vy'][0], data['stateEstimate.vz'][0]]))
    pos.append(np.array([data['stateEstimate.x'][0], data['stateEstimate.y'][0], data['stateEstimate.z'][0]]))
    quaternions.append(np.array([data['stateEstimate.qw'][0],data['stateEstimate.qx'][0], data['stateEstimate.qy'][0], data['stateEstimate.qz'][0]]))

    for i in range(len(data['timestamp'])-1):
        R = rowan.to_matrix(np.array([data['stateEstimate.qw'][i],data['stateEstimate.qx'][i], data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]]))
        z_b= np.asarray(R@z_w) 
        u = thrust_torque(pwm_1[i], pwm_2[i], pwm_3[i], pwm_4[i], mv[i])

        time = data['timestamp'][i+1]

        ang_a = angular_acc(u, np.array([data['gyro.x'][i], data['gyro.y'][i], data['gyro.z'][i]])*d2r)
        acc_a.append(ang_a)

        a = acceleration(u, z_w, z_b)
        acc.append(a)
        err_a = np.array([data['acc.x'][i+1], data['acc.y'][i+1], data['acc.z'][i+1]]) -a*ms2g
        err_acc.append(err_a)

        v = velocity(acc[i], np.array([data['stateEstimate.vx'][i], data['stateEstimate.vy'][i], data['stateEstimate.vz'][i]]), time, prev_time)
        vel.append(v)
        err_v = np.array([data['stateEstimate.vx'][i+1], data['stateEstimate.vy'][i+1], data['stateEstimate.vz'][i+1]]) -v
        err_vel.append(err_v)

        y = position(vel[i],  np.array([data['stateEstimate.x'][i], data['stateEstimate.y'][i], data['stateEstimate.z'][i]]), time, prev_time)
        pos.append(y)
        err_y = np.array([data['stateEstimate.x'][i+1], data['stateEstimate.y'][i+1], data['stateEstimate.z'][i+1]]) -y
        err_pos.append(err_y)

        v_a = angular_velocity(acc_a[i],  np.array([data['gyro.x'][i], data['gyro.y'][i], data['gyro.z'][i]])*d2r  ,time, prev_time)
        vel_a.append(v_a)
        err_v_a = np.array([data['gyro.x'][i+1], data['gyro.y'][i+1], data['gyro.z'][i+1]])*d2r - v_a
        err_vel_a.append(err_v_a)

        quat = new_quaternion(vel_a[i], np.array([data['stateEstimate.qw'][i],data['stateEstimate.qx'][i], data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]]), time, prev_time)
        quaternions.append(quat)

        err_quat = rowan.multiply(np.array([data['stateEstimate.qw'][i+1],data['stateEstimate.qx'][i+1], data['stateEstimate.qy'][i+1], data['stateEstimate.qz'][i+1]]), rowan.inverse(quat))
        err_quaternions.append(err_quat)

        prev_time = time

    acc = np.array(acc)*ms2g
    vel = np.array(vel)
    vel_a = np.array(vel_a)*r2d
    pos = np.array(pos)
    quaternions = np.array(quaternions)

    err_acc = np.array(err_acc)
    err_vel = np.array(err_vel)
    err_pos = np.array(err_pos)
    err_vel_a = np.array(err_vel_a)
    err_quaternions = np.array(err_quaternions)


    compare_data(data, quaternions, acc, vel, vel_a, pos, name )
    errors(data, err_acc, err_vel, err_pos, err_vel_a, err_quaternions, name)
    trajectory_x_y_plane(data, name)
    trajectory(data, name)

if __name__ == "__main__":
    for i in ['00', '01', '02', '03', '04','05', '06', '10', '11']:
        path = f'hardware/data/jana{i}'
        data = decode_data(path)
        name = f'jana{i}'
        propagate(data, name)