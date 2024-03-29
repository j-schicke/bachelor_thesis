import numpy as np
import rowan
from config.multirotor_config import MultirotorConfig
from basis_forward_propagation import decode_data
from model import NeuralNetwork
from residual_calculation import residual
import torch
from torch import nn
from time import perf_counter
from plot_data import plot_test_data_f, plot_test_data_tau, model_error_f, model_error_tau
    
if __name__ == '__main__':

    model = NeuralNetwork()
    model.double()
    timestamp, X_test, y_test = model.train_model() 
    
    pred_arr = []
    for i in range(len(y_test)):
        pred = model.forward(X_test[i])
        pred_arr.append(pred.cpu().detach().numpy())
    pred_arr = np.array(pred_arr)
    y_test = ( np.array(y_test))


    plot_test_data_f(y_test[:, :3], pred_arr, timestamp)
    plot_test_data_tau(y_test[:,3:], pred_arr, timestamp)
    model_error_f(y_test, pred_arr, timestamp)
    model_error_tau(y_test, pred_arr, timestamp)
