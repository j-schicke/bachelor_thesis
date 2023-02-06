import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np
from sklearn.model_selection import train_test_split
from plot_data import losses, plot_test_data_f, plot_test_data_tau
from basis_forward_propagation import decode_data
from config.multirotor_config import MultirotorConfig
from residual_calculation import residual
import rowan
from sklearn.utils import shuffle

d2r = MultirotorConfig.deg2rad


class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 6, hidden_size = 6, output_size = 6, spectral = False):
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if spectral:
            self.linear_relu = nn.Sequential(
                spectral_norm(nn.Linear(self.input_size, self.hidden_size), n_power_iterations = 2),
                nn.ReLU(),
                spectral_norm(nn.Linear(self.hidden_size, self.output_size),n_power_iterations = 2)
            )
        else:
            self.linear_relu = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size)
            )


    def forward(self, x):
        pred = self.linear_relu(x)
        return pred

    def train_loop(self, X, y ,loss_fn, optimizer):
        self.train()
        train_loss = []
        for i in range(len(y)):
            
            optimizer.zero_grad()
            pred = self.forward(X[:,i]) 
            #loss_1 = loss_fn(pred[:3], y[:3,i])
            #loss_2 = loss_fn(pred[3:], y[3:,i])
            loss = loss_fn(pred, y[:,i] )
            #loss = loss_1 + 10*loss_2

            loss.backward()
            optimizer.step()
            loss = loss.item()
            train_loss.append(loss)
        avg_train = np.array(train_loss).sum()/len(y)

        print(f'avg. train loss: {avg_train :> 5f}')
        return avg_train

    def test_loop(self, X, y, loss_fn, num, epoche):
        self.eval()
        test_losses = []
        pred_arr = []

        with torch.no_grad():

            for i in range(len(y)):

                pred = self.forward(X[i])
                pred_arr.append(pred.cpu().detach().numpy())
                test_loss = loss_fn(pred, y[i])
                #test_loss_1 = loss_fn(pred[:3], y[i,:3 ])
                #test_loss_2 = loss_fn(pred[3:], y[i, 3:])
                #test_loss = test_loss_1 + 10*test_loss_2

                test_losses.append(test_loss.item())
                
        avg_test = np.array(test_losses).sum()/len(y) 
        print(f'avg. test loss: {avg_test :> 5f}')
        if num == epoche-1:
            pred_arr = np.array(pred_arr)
            y = np.array(y)
            #plot_test_data_f(y[:, :3], pred_arr)
            plot_test_data_tau(y, pred_arr)

        return avg_test


    def train_model(self):
        X = np.array([])
        y = np.array([])

        for i in ['00', '01', '02', '03', '04', '05', '06', '10','11']:


            data = decode_data(f"hardware/data/jana{i}")

            k = np.array([data['stateEstimate.vx'][1:], data['stateEstimate.vy'][1:], data['stateEstimate.vz'][1:], data['gyro.x'][1:]*d2r, data['gyro.y'][1:]*d2r,data['gyro.z'][1:]*d2r])

            r = np.array([])

            for j in range(1,len(data['timestamp'])):

                R = rowan.to_matrix(np.array([data['stateEstimate.qw'][j],data['stateEstimate.qx'][j], data['stateEstimate.qy'][j], data['stateEstimate.qz'][j]]))[:,:2]
                R = R.reshape(1, 6)
                if len(r) == 0:
                    r = R
                else:
                    r = np.append(r, R, axis=0)
            k = np.append(k, r.T, axis = 0)

            if len(X) == 0:
                X = k.T
            else:
                X = np.append(X, k.T, axis=0)

            name = f"jana{i}"
            f_a, tau_a, = residual(data, name)
            #tmp = np.append(f_a, tau_a, axis=1)
            tmp = tau_a
            if len(y) == 0:
                y = tmp
            else: 
                y = np.append(y, tmp, axis=0)
        X, y = shuffle(X, y, random_state=3)

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state= 1)

        X_train = torch.from_numpy(X_train.T) 
        X_test = torch.from_numpy(X_test)

        y_train = torch.from_numpy(y_train.T)
        y_test = torch.from_numpy(y_test)


        self.double()
        epoche = 100
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.003)
        loss_fn = nn.MSELoss()
        train_losses = []

        test_losses = []
        
        for t in range(epoche):
            print(f"Epoch {t+1}\n-------------------------------")
            
            train_losses.append(self.train_loop(X_train, y_train, loss_fn, optimizer))
            test_losses.append(self.test_loop(X_test, y_test, loss_fn, t, epoche))

            print(f"\n-------------------------------")



        print("Done!")
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        losses(train_losses, test_losses)

        torch.save(self.state_dict(), 'model_1.pth')