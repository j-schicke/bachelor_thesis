import numpy as np 
import cfusdlog
import matplotlib.pyplot as plt
import numpy as np
from config.multirotor_config import MultirotorConfig
import rowan
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from model import NeuralNetwork

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

def angular_acceleration(a_vel, prev_time, time):
    t = (time - prev_time)*1000
    a_acc = a_vel/t
    return a_acc

def disturbance_forces(m, vel, R, f_u):
    g = np.array([0,0,9.81])
    f_a = m*vel - m*g - R@f_u
    return f_a


def disturbance_torques(I, a_acc, a_vel, tau_u):
    tau_a = I@a_acc - np.cross(I@a_vel, a_vel) - tau_u
    return tau_a

def train_loop(model, X, y ,loss_fn, optimizer):

        for i in range(len(y)):
        
            optimizer.zero_grad()
            pred = model(X[i]) 
            loss = loss_fn(pred, y[i] )

            loss.backward()
            optimizer.step()
          
        loss = loss.item()
        print(f'loss: {loss :> 5f}')

    

def train_model(data, f):
    X = np.array(tuple(data.values()) ).T[:-1]
    y = np.array(f)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    
    X_train = torch.from_numpy(X_train) 
    y_train = torch.from_numpy(y_train)


    model = NeuralNetwork()
    model.double()
    epos = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr =0.003)
    
    for t in range(epos):
        print(f"Epoch {t+1}\n-------------------------------")
        
        train_loop(model,X_train, y_train, loss_fn, optimizer)


    print("Done!")
    torch.save(model.state_dict(), 'model_1.pth')


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

    f_p = []
    for i in range(1,len(data['timestamp'])):
        time = data['timestamp'][i]
        a_vel = np.array([data['gyro.x'][i], data['gyro.y'][i], data['gyro.z'][i]])
        vel = np.array([data['stateEstimate.vx'][i], data['stateEstimate.vy'][i], data['stateEstimate.vz'][i]])

        R = rowan.to_matrix(np.array([data['stateEstimate.qw'][i],data['stateEstimate.qx'][i], data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]]))
        u = thrust_torque(pwm_1[i], pwm_2[i], pwm_3[i], pwm_4[i], mv[i])
        a_acc = angular_acceleration(a_vel, prev_time, time)
        f_u = np.array([0,0, u[0]])
        f_a = disturbance_forces(m, vel, R, f_u)
        f.append(f_a)

        tau_u = np.array([u[1], u[2], u[3]])
        tau_a = disturbance_torques(I, a_acc, a_vel, tau_u)
        tau.append(tau_a)
        X = np.array(tuple(data.values()) ).T[i]
        X = torch.from_numpy(X) 
        f_p.append(model(X).cpu().detach().numpy())
    #train_model(data, f)


    




    f = np.array(f)
    f_p = np.array(f_p)
    tau = np.array(tau)
    fig, ax = plt.subplots(2)
    ax[0].plot(data['timestamp'][1:], f[:,0], '-', label='X')
    ax[0].plot(data['timestamp'][1:], f[:,1], '-', label='Y')
    ax[0].plot(data['timestamp'][1:], f[:,2], '-', label='Z')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('f_a')
    ax[0].set_title('f_a')
    ax[0].legend(loc=9, ncol=3, borderaxespad=0.)

    ax[1].plot(data['timestamp'][1:], f_p[:,0], '-', label='X')
    ax[1].plot(data['timestamp'][1:], f_p[:,1], '-', label='Y')
    ax[1].plot(data['timestamp'][1:], f_p[:,2], '-', label='Z')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('f_p')
    ax[1].set_title('f predicted')
    ax[1].legend(loc=9, ncol=3, borderaxespad=0.)
    plt.show()


