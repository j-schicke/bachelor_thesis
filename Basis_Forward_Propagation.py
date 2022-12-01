import numpy as np 
import cfusdlog
import matplotlib.pyplot as plt
import argparse
import numpy as np
import mplcursors
import functools
import math

def showAnnotation(data, sel):
    idx = sel.target.index
    sel.annotation.set_text(
        "\n".join(['{}: {}'.format(key, data[key][idx]) for key in data.keys()]))


def euler_from_quaternion(x, y, z, w):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z


def plot_data(data):
    start_time = data['timestamp'][0]
    # new figure
    del data['pwm.m1_pwm']
    del data['pwm.m2_pwm']
    del data['pwm.m3_pwm']
    del data['pwm.m4_pwm']
    del data['pm.vbatMV']


    fig, ax = plt.subplots()
    t = (data['timestamp'] - start_time) / 1000
    ax.scatter(t, t*0)
    ax.set_title('fixedFrequency')
    print(data.keys())
    crs = mplcursors.cursor(hover=True)
    crs.connect("add", functools.partial(showAnnotation, data))
    ax.set_xlabel('Time [s]')

    plt.show()


def newton_euler(w, ct, cq, d):
    u = np.array([[ct, ct, ct, ct], [0, d*ct, 0, -d*ct], [-d*ct, 0, d*ct, 0], [cq, cq, cq, cq]]) @ w**2
    return u

def newton(m, u, z_w, z_b):
    g = 9.8
    m_r = -m*g*z_w+u[0]*z_b
    return m_r

def euler(u, wbw, I):
    wbw_2 = np.linalg.inv(I)@(np.cross(-wbw,I)@wbw+np.array([u[1], u[2], u[3]]))
    return wbw_2

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("file_usd")
    #args = parser.parse_args()
    #data_usd = cfusdlog.decode(args.file_usd)
    data_usd = cfusdlog.decode("log01")
    data = data_usd['fixedFrequency']
    I = np.array([[16.571710, 0.830306, 0.718277], [0.830806, 16.655602, 1.800197], [0.718277, 1.800197, 29.261652]])
    ct = -9.1785*10**(-7)
    cq = -10.311*10**(-7)
    d = 0.092
    m = 28
    for i in range(len(data)):
        x_w = np.array([1,0,0])
        y_w = np.array([0,1,0])
        z_w = np.array([0,0,1])
        r, p, y = euler_from_quaternion(data['stateEstimate.qx'][i], data['stateEstimate.qy'][i], data['stateEstimate.qz'][i], data['stateEstimate.qw'][i])
        R = np.array([[math.cos(y)*math.cos(p)-math.sin(r)*math.sin(y)*math.sin(p),-math.cos(r)*math.sin(y),math.cos(y)*math.sin(p)+math.cos(p)*math.sin(r)*math.sin(y)],[math.cos(p)*math.sin(y)+math.cos(y)*math.sin(r)*math.sin(p), math.sin(r)*math.cos(y), math.cos(p)*math.sin(y)-math.cos(y)*math.cos(p)*math.sin(r)],[-math.cos(r)*math.sin(p),math.sin(r), math.cos(r)*math.cos(p)]])
        
        x_b= np.asarray(R@x_w)
        y_b= np.asarray(R@y_w)
        z_b= np.asarray(R@z_w)

        w = np.array([data['rpm.m1'][i], data['rpm.m2'][i], data['rpm.m3'][i], data['rpm.m4'][i]])
        u = newton_euler(w, ct, cq, d)
        wbw = np.array([data['gyro.x'][i], data['gyro.y'][i], data['gyro.z'][i]])
        acc_a = euler(u, wbw, I)
        #acc_cm = newton(m, u, z_w, data['stateEstimate.z'][i])
        acc_cm = newton(m,u, z_w, z_b)
      

    plot_data(data)
    

