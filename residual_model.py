import numpy as np
import rowan
from config.multirotor_config import MultirotorConfig
from basis_forward_propagation import decode_data
from model import NeuralNetwork
from plot_data import compare_predicted_f_a, compare_predicted_tau_a, error_pred_f, error_pred_tau
from residual_calculation import residual
import torch
import time


def model_predict(data, y, name):

    d2r = MultirotorConfig.deg2rad
    model.load_state_dict(torch.load('model_1.pth'))
    pred = []

    for i in range(1,len(data['timestamp'])):
        # R = rowan.to_matrix(np.array([data['stateEstimate.qw'][i],data['stateEstimate.qx'][i], data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]]))[:,:2]
        # R = R.reshape(6,1).flatten()
        X = np.array([data['stateEstimate.vx'][i], data['stateEstimate.vy'][i], data['stateEstimate.vz'][i], data['gyro.x'][i]*d2r, data['gyro.y'][i]*d2r,data['gyro.z'][i]*d2r])
        # X = np.append(X, R, axis = 0 ) 
        X = torch.from_numpy(X) 
        pred.append(model(X).cpu().detach().numpy())

    pred = np.array(pred)
    compare_predicted_f_a(data, y[:,:3], pred, name)
    compare_predicted_tau_a(data, y[:,3:], pred, name)
    error_pred_tau(data, y[:,3:], pred, name)
    error_pred_f(data, y[:,:3], pred, name)
    
if __name__ == '__main__':

    start = time.time()


    model = NeuralNetwork()
    model.double()
    model.train_model() 
    
    for i in ['00', '01', '02', '03', '04', '05', '06', '10','11']:
        y = np.array([])
        path = f"hardware/data/jana{i}"
        data = decode_data(path)
        name = f"jana{i}"
        f_a, tau_a, = residual(data, name)
        y = np.append(f_a, tau_a, axis=1)
        model_predict(data,y, name)


    end = time.time()
    print(end - start)