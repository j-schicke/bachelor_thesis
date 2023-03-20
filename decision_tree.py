import numpy as np
import xgboost as xg
from basis_forward_propagation import decode_data
from residual_calculation import residual
from sklearn.utils import shuffle
from config.multirotor_config import MultirotorConfig
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from plot_data import plot_test_pred_f, plot_test_pred_tau, tree_losses, tree_error_f, tree_error_tau
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from time import perf_counter
import torch
import rowan


d2r = MultirotorConfig.deg2rad
g = MultirotorConfig.GRAVITATION

def test_train_data():
    X_train = np.array([])
    y_train = np.array([])
    minmax_scaler = MinMaxScaler(feature_range=(-1,1))
    for i in ['00', '01', '02', '03', '04','05', '06', '10', '11', '20', '23', '24', '25', '27', '28', '29', '30', '32', '33']:
        if i == '02':
            data = decode_data(f"hardware/data/jana{i}")
            X_test = np.array([data['stateEstimate.vx'][1:], data['stateEstimate.vy'][1:], data['stateEstimate.vz'][1:], data['gyro.x'][1:]*d2r, data['gyro.y'][1:]*d2r,data['gyro.z'][1:]*d2r]).T

            name = f"jana{i}"
            f_a,tau_a = residual(data, name)
            y_test = np.append(f_a, tau_a, axis=1)
            test_timestamp = data['timestamp'][1:]

        else:
            data = decode_data(f"hardware/data/jana{i}")
            k = np.array([data['stateEstimate.vx'][1:], data['stateEstimate.vy'][1:], data['stateEstimate.vz'][1:], data['gyro.x'][1:]*d2r, data['gyro.y'][1:]*d2r,data['gyro.z'][1:]*d2r]).T

            if len(X_train) == 0:
                X_train = k
            else:
                X_train = np.append(X_train, k, axis=0)

            name = f"jana{i}"
            f_a, tau_a = residual(data, name)
            tmp = np.append(f_a, tau_a, axis=1)
            if len(y_train) == 0:
                y_train = tmp
            else: 
                y_train = np.append(y_train, tmp, axis=0)

    X_train = pd.DataFrame(X_train, columns = ['Vel X','Vel Y','Vel Z', 'Gyr X', 'Gyr Y', 'Gyr Z'])
    X_test = pd.DataFrame(X_test, columns=['Vel X','Vel Y','Vel Z', 'Gyr X', 'Gyr Y', 'Gyr Z'])

    y_full = np.append(y_train[:,3:], y_test[:,3:], axis=0)
    y_scaled = minmax_scaler.fit_transform(y_full)
    y_train = np.append(y_train[:, :3], y_scaled[:len(y_train),:], axis = 1)
    y_test = np.append(y_test[:,:3],y_scaled[len(y_train):,:], axis = 1)

    y_train = torch.from_numpy(y_train) 
    y_test = torch.from_numpy(y_test)

    X_train, y_train = shuffle(X_train, y_train, random_state=3)

    return X_train, y_train, X_test, y_test, test_timestamp

def train_tree():

    X_train, y_train, X_test, y_test, test_timestamp = test_train_data()

    eval_set = [(X_train, y_train), (X_test, y_test)]


    model = xg.XGBRegressor(n_estimators=100, objective = 'reg:squarederror')
    model.fit(X_train, y_train, eval_set = eval_set)

    results = model.evals_result()
    tree_losses(results)

    xg.plot_importance(model)
    plt.savefig('pdf/Decision Tree/features.png')

    model.save_model('tree.json')
    xg.plot_tree(model)
    plt.show()
    test_tree(X_test,y_test, model, test_timestamp)

def test_tree(X, y, model, test_timestamp):
    pred = model.predict(X.values)


    y = np.array(y)
    pred = np.array(pred)
    plot_test_pred_f(y[:, :3], pred, test_timestamp)
    plot_test_pred_tau(y[:, 3:], pred, test_timestamp)
    tree_error_f(y[:, :3], pred, test_timestamp)
    tree_error_tau(y[:, 3:], pred, test_timestamp)


if __name__ == "__main__":
    train_tree()
