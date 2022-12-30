import numpy as np 
import cfusdlog
import matplotlib.pyplot as plt
import numpy as np
from config.multirotor_config import MultirotorConfig
import rowan
from sklearn import preprocessing
from model import NeuralNetwork
from Basis_Forward_Propagation import thrust_torque
from plot_data import residual_plot
import torch

def angular_acceleration(a_vel, prev_time, time):
    t = (time - prev_time)* 0.001
    a_acc = a_vel/t
    return a_acc

def disturbance_forces(m, acc, R, f_u):
    g = np.array([0,0,-9.81])
    f_a = m*acc - m*g - R@f_u
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
    d = MultirotorConfig.DISTANCE_M_C
    m = MultirotorConfig.MASS
    f = []
    tau = []
    prev_time = start_time
    
    pwm_1 = preprocessing.normalize(data['pwm.m1_pwm'][None])[0]
    pwm_2 = preprocessing.normalize(data['pwm.m2_pwm'][None])[0]
    pwm_3 = preprocessing.normalize(data['pwm.m3_pwm'][None])[0]
    pwm_4 =preprocessing.normalize(data['pwm.m4_pwm'][None])[0]
    mv = preprocessing.normalize(data['pm.vbatMV'][None])[0]
    model = NeuralNetwork()
    model.double()
    model.load_state_dict(torch.load('model_1.pth'))
    pred = []

    for i in range(1,len(data['timestamp'])):
        time = data['timestamp'][i]
        a_vel = np.array([data['gyro.x'][i], data['gyro.y'][i], data['gyro.z'][i]])*0.017453
        acc = np.array([data['acc.x'][i], data['acc.y'][i], data['acc.z'][i]])*9.80665

        R = rowan.to_matrix(np.array([data['stateEstimate.qw'][i],data['stateEstimate.qx'][i], data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]]))
        u = thrust_torque(pwm_1[i], pwm_2[i], pwm_3[i], pwm_4[i], mv[i])
        a_acc = angular_acceleration(a_vel, prev_time, time)
        f_u = np.array([0,0, u[0]])
        f_a = disturbance_forces(m, acc, R, f_u)
        f.append(f_a)
        X = np.array(tuple(data.values()) ).T[i][1:]
        X = preprocessing.normalize(X[None])[0]
        X = torch.from_numpy(X) 
        pred.append(model(X).cpu().detach().numpy())
        tau_u = np.array([u[1], u[2], u[3]])
        tau_a = disturbance_torques(I, a_acc, a_vel, tau_u)
        tau.append(tau_a)


    f = np.array(f)
    tau = np.array(tau)
    pred = np.array(pred)

    fig, ax = plt.subplots(2)
    ax[0].plot(data['timestamp'][1:], f[:,0], '-', label='X')
    ax[0].plot(data['timestamp'][1:], f[:,1], '-', label='Y')
    ax[0].plot(data['timestamp'][1:], f[:,2], '-', label='Z')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('f_a')
    ax[0].set_title('f_a')
    ax[0].legend(loc=9, ncol=3, borderaxespad=0.)

    ax[1].plot(data['timestamp'][1:], pred[:,0], '-', label='X')
    ax[1].plot(data['timestamp'][1:], pred[:,1], '-', label='Y')
    ax[1].plot(data['timestamp'][1:], pred[:,2], '-', label='Z')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('f_p')
    ax[1].set_title('f predicted')
    ax[1].legend(loc=9, ncol=3, borderaxespad=0.)
    plt.show()


    #model.train_model(data, f)

    residual_plot(data, f, tau)

