import torch
from torch import nn
import  rowan
import numpy as np
from plot_data import losses
from basis_forward_propagation import decode_data
from config.multirotor_config import MultirotorConfig
from residual_calculation import residual
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from time import perf_counter


d2r = MultirotorConfig.deg2rad
g = MultirotorConfig.GRAVITATION


class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 12, hidden_size = 15,  output_size = 6):
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_relu = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )
    def forward(self, x):
        pred = self.linear_relu(x)
        return pred

    def train_loop(self, dataloader ,loss_fn, optimizer):
        self.train()
        d = 0
        train_loss = []
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            pred = self.forward(X)
            loss = loss_fn(pred, y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                d+=1
                loss = loss.item()
                train_loss.append(loss)

        avg_train = np.array(train_loss).sum()/d

        print(f'avg. train loss: {avg_train :> 5f}')
        return avg_train

    def test_loop(self, dataloader, loss_fn):
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self.forward(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches

        print(f"Avg. test loss: {test_loss:>8f} \n")
        return test_loss 

    def train_model(self):

        train_dataloader, test_dataloader, test_timestamp, X_test, y_test = self.train_test_data()
        
        self.double()
        epoche = 30
        optimizer = torch.optim.Adam(self.parameters(), lr =0.0035)

        loss_fn = nn.MSELoss()
        train_losses = []
        test_losses = []

        for t in range(epoche):
            print(f"Epoch {t+1}\n-------------------------------")
                
            train_losses.append(self.train_loop(train_dataloader, loss_fn, optimizer))
            test_losses.append(self.test_loop(test_dataloader, loss_fn))
        
            print(f"\n-------------------------------")

        print("Done!")
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        losses(train_losses, test_losses)

        torch.save(self.state_dict(), 'model_1.pth')
        return test_timestamp, X_test, y_test
    
    def train_test_data(self):
        minmax_scaler = MinMaxScaler(feature_range=(-1,1))

        X = np.array([])
        y = np.array([])

        for i in ['00', '01', '02', '03', '04','05', '06', '10', '11','20', '23', '24', '25', '27', '28', '29', '30', '32', '33']:


            data = decode_data(f"hardware/data/jana{i}")
            r = np.array([])
            for j in range(1,len(data['timestamp'])):
                R = rowan.to_matrix(np.array([data['stateEstimate.qw'][j],data['stateEstimate.qx'][j], data['stateEstimate.qy'][j], data['stateEstimate.qz'][j]]))[:,:2]
                R = R.reshape(1, 6)
                if len(r) == 0:
                    r = R
                else:
                    r = np.append(r, R, axis=0)

            k = np.array([data['acc.x'][1:]*g, data['acc.y'][1:]*g, data['acc.z'][1:]*g, data['gyro.x'][1:]*d2r, data['gyro.y'][1:]*d2r,data['gyro.z'][1:]*d2r]).T

            k = np.append(k, r, axis=1)
            if i == '02':
                X_test = k
                name = f"jana{i}"
                f_a, tau_a, = residual(data, name)
                tmp = np.append(f_a, tau_a, axis=1)
                y_test = tmp
                test_timestamp = data['timestamp'][1:]
            
            else:
                if len(X) == 0:
                    X = k
                else:
                    X = np.append(X, k, axis=0)

                name = f"jana{i}"
                f_a, tau_a, = residual(data, name)
                tmp = np.append(f_a, tau_a, axis=1)
                if len(y) == 0:
                    y = tmp
                else: 
                    y = np.append(y, tmp, axis=0)


        X, y_train = shuffle(X, y)
        

        y_full = np.append(y_train, y_test, axis = 0)
        y_scaled = minmax_scaler.fit_transform(y_full)
        y_train = y_scaled[:len(y_train),:]
        y_test = y_scaled[len(y_train):,:]

        start = perf_counter()

        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(np.array(y_test))


        X_train = torch.from_numpy(np.array(X))
        y_train = torch.from_numpy(np.array(y_train))

        train_dataset = TensorDataset(X_train,y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=64)
        test_dataloader = DataLoader(test_dataset, batch_size=64)
        return train_dataloader, test_dataloader, test_timestamp, X_test, y_test
