import numpy as np
import rowan
from config.multirotor_config import MultirotorConfig
from basis_forward_propagation import decode_data
from model import NeuralNetwork
from residual_calculation import residual
import torch
from torch import nn
from time import perf_counter
    
if __name__ == '__main__':
#     for i in ['00', '01', '02', '03', '04','05', '06', '10', '11']:
#         path = f'hardware/data/jana{i}'
#         data = decode_data(path)
#         name = f'jana{i}'
#         residual(data, name)

    start = perf_counter()

    model = NeuralNetwork(input_size= 12)
    model.double()
    model.train_model() 

    end = perf_counter()
    print(end - start)