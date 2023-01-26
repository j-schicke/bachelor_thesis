import numpy as np
import xgboost as xg
from sklearn.model_selection import train_test_split
from basis_forward_propagation import decode_data
from residual_calculation import residual
from sklearn.utils import shuffle
import rowan
from sklearn.metrics import mean_squared_error as MSE
from config.multirotor_config import MultirotorConfig

d2r = MultirotorConfig.deg2rad

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

    # # Splitting
    train_X, test_X, train_y, test_y = train_test_split(X, y,train_size=0.7, random_state = 3)

    # Instantiation
    xgb_r = xg.XGBRegressor(objective ='reg:squarederror', n_estimators = 25 , seed = 3)

    # Fitting the model
    xgb_r.fit(train_X, train_y)

    # Predict the model
    pred = xgb_r.predict(test_X)



    rmse = MSE(test_y, pred)
    print("RMSE : % f" %(rmse))


    # xgb_r.save_model('tree.json')

train_tree()