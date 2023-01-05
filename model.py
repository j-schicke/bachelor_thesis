import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from plot_data import losses



class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 25, hidden_size = [19, 13], output_size = 6):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_relu = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], self.output_size)

        )
    def forward(self, x):
        f_a = self.linear_relu(x )
        return f_a

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
        del data['timestamp']
        X = np.array(tuple(data.values()) ).T[:-1]
        y = np.array(y)
        #X = preprocessing.normalize(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state= 13)

        X_train = preprocessing.normalize(X_train)
        X_train = torch.from_numpy(X_train) 
        
        X_test = preprocessing.normalize(X_test)
        X_test = torch.from_numpy(X_test)

        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
        self.double()
        epos = 25
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr =0.03)
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


    
    
