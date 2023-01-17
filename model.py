import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np
from sklearn.model_selection import train_test_split
from plot_data import losses
from config.multirotor_config import MultirotorConfig

d2r = MultirotorConfig.deg2rad


class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 6, hidden_size = 6, output_size = 6):
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_relu = nn.Sequential(
            spectral_norm(nn.Linear(self.input_size, self.hidden_size), n_power_iterations = 2),
            nn.ReLU(),
            spectral_norm(nn.Linear(self.hidden_size, self.output_size),n_power_iterations = 2)

        )
    def forward(self, x):
        pred = self.linear_relu(x)
        return pred

    def train_loop(self, X, y ,loss_fn, optimizer):
        self.train()
        train_loss = []
        for i in range(len(y)):
            
            optimizer.zero_grad()
            pred = self.forward(X[i]) 
            loss = loss_fn(pred, y[i] )

            loss.backward()
            optimizer.step()
            loss = loss.item()
            train_loss.append(loss)
        avg_train = np.array(train_loss).sum()/len(y)

        print(f'avg. train loss: {avg_train :> 5f}')
        return avg_train

    def test_loop(self, X, y, loss_fn):
        self.eval()
        test_losses = []
        with torch.no_grad():
            for i in range(len(y)):
                pred = self.forward(X[i])
                test_loss = loss_fn(pred, y[i])
                test_losses.append(test_loss.item())
        avg_test = np.array(test_losses).sum()/len(y) 
        print(f'avg. test loss: {avg_test :> 5f}')
        return avg_test


    def train_model(self, data, y):
        X = np.array([data['stateEstimate.vx'], data['stateEstimate.vy'], data['stateEstimate.vz'], data['gyro.x']*d2r, data['gyro.y']*d2r,data['gyro.z']*d2r])
        X = X.T[1:]
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state= 1)

        X_train = torch.from_numpy(X_train) 
        X_test = torch.from_numpy(X_test)

        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        self.double()
        epos = 25
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr =0.001)
        train_losses = []

        test_losses = []
        
        for t in range(epos):
            print(f"Epoch {t+1}\n-------------------------------")
            
            train_losses.append(self.train_loop(X_train, y_train, loss_fn, optimizer))
            test_losses.append(self.test_loop(X_test, y_test, loss_fn))


        print("Done!")
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        losses(train_losses, test_losses)

        torch.save(self.state_dict(), 'model_1.pth')