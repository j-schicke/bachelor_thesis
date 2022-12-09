import numpy as np

class MultirotorConfig:
    MASS = 0.028 #in kg
    INERTIA = np.array([[16.571710, 0.830306, 0.718277], [0.830806, 16.655602, 1.800197], [0.718277, 1.800197, 29.261652]])*10**(-6)
    THRUST_C = -9.1785*10**(-7)    
    TORQUE_C = -10.311*10**(-7)
    DISTANCE_M_C  = 0.092 #in meters