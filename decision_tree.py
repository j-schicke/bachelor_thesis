import numpy as np
import xgboost as xg
from sklearn.model_selection import train_test_split
from basis_forward_propagation import decode_data
from residual_calculation import residual
from sklearn.utils import shuffle
import rowan
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config.multirotor_config import MultirotorConfig
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from plot_data import plot_test_pred_f, plot_test_pred_tau, tree_losses
import pandas as pd 


d2r = MultirotorConfig.deg2rad

def test_tree(X, y, model):
    pred = model.predict(X.values)


    y = np.array(y)
    pred = np.array(pred)
    plot_test_pred_f(y[:, :3], pred)
    plot_test_pred_tau(y[:, 3:], pred)




def train_tree():
    X = np.array([])
    y = np.array([])
    for i in ['00', '01', '02', '03', '04', '05', '06', '10','11']:

            data = decode_data(f"hardware/data/jana{i}")
            r = np.array([])
            for j in range(1,len(data['timestamp'])):
                R = rowan.to_matrix(np.array([data['stateEstimate.qw'][j],data['stateEstimate.qx'][j], data['stateEstimate.qy'][j], data['stateEstimate.qz'][j]]))[:,:2]
                R = R.reshape(1, 6)
                if len(r) == 0:
                    r = R
                else:
                    r = np.append(r, R, axis=0)


            k = np.array([data['stateEstimate.vx'][1:], data['stateEstimate.vy'][1:], data['stateEstimate.vz'][1:], data['gyro.x'][1:]*d2r, data['gyro.y'][1:]*d2r,data['gyro.z'][1:]*d2r])
            k = np.append(k, r.T, axis = 0)
            if len(X) == 0:
                X = k.T
            else:
                X = np.append(X, k.T, axis=0)

            name = f"jana{i}"
            f_a, tau_a, = residual(data, name)
            tmp = np.append(f_a, tau_a, axis=1)
            if len(y) == 0:
                y = tmp
            else: 
                y = np.append(y, tmp, axis=0)
    
    X, y = shuffle(X, y, random_state=3)

    df = pd.DataFrame(X, columns = ['Vel X','Vel Y','Vel Z', 'Gyr X', 'Gyr Y', 'Gyr Z', 'rot 00', 'rot 01', 'rot 02', 'rot10', 'rot11', 'rot12'])

    X_train, X_test, y_train, y_test = train_test_split(df, y,train_size=0.7, random_state = 3)

    eval_set = [(X_train, y_train), (X_test, y_test)]

    model = xg.XGBRegressor(n_estimators=1000, learning_rate = 0.0001, objective = 'reg:squarederror')
    model.fit(X_train, y_train, eval_set = eval_set)

    results = model.evals_result()
    tree_losses(results)

    # xg.plot_importance(model)
    # plt.savefig('pdf/Decision Tree/features.pdf')

    model.save_model('tree.json')
    test_tree(X_test,y_test, model)

train_tree()