# -*- coding: utf-8 -*-
"""
plotting a generic USD log
"""
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import functools
from mpl_toolkits.mplot3d import Axes3D



def compare_gyro(data, vel_a, name):
    t = (data['timestamp'] - data['timestamp'][0]) / 1000

    fig,ax = plt.subplots(3)

    ax[0].plot(t, data['gyro.x'], '-', label='data')
    ax[0].plot(t, vel_a[:,0], '-', label='propagated')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('Gyroscope [°/s]')
    ax[0].set_title('angular velocity X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)



    ax[1].plot(t, data['gyro.y'], '-', label='data')
    ax[1].plot(t, vel_a[:,1], '-', label='propagated')
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('Gyroscope [°/s]')
    ax[1].set_title('angular velocity Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[2].plot(t, data['gyro.z'], '-', label='data')
    ax[2].plot(t, vel_a[:,2], '-', label='propagated')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('Gyroscope [°/s]')
    ax[2].set_title('angular velocity Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    
    plt.tight_layout()

    plt.savefig(f'pdf/Data/{name}/angular_velocity.png', bbox_inches='tight')  

def compare_position(data, pos, name):
    t = (data['timestamp'] - data['timestamp'][0]) / 1000


    fig,ax = plt.subplots(3)

    ax[0].plot(t, data['stateEstimate.x'], '-', label='data')
    ax[0].plot(t, pos[:,0], '-', label='propagated')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('position')
    ax[0].set_title('position X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[1].plot(t, data['stateEstimate.y'], '-', label='data')
    ax[1].plot(t, pos[:,1], '-', label='propagated')
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('position')
    ax[1].set_title('position Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[2].plot(t, data['stateEstimate.z'], '-', label='data')
    ax[2].plot(t, pos[:,2], '-', label='propagated')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('position')
    ax[2].set_title('position Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()


    plt.savefig(f'pdf/Data/{name}/position.png', bbox_inches='tight')  


def trajectory_x_y_plane(data, name):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xline = data['stateEstimate.x']
    yline = data['stateEstimate.y']
    zline = data['stateEstimate.z']

    ax.plot3D(xline, yline, zline, 'gray')
    ax.view_init(elev=90)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.tight_layout()

    plt.savefig(f'pdf/Data/{name}/trajectory_x_y_plane.png', bbox_inches='tight')

def trajectory(data, name):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xline = data['stateEstimate.x']
    yline = data['stateEstimate.y']
    zline = data['stateEstimate.z']

    ax.plot3D(xline, yline, zline, 'gray')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.tight_layout()

    plt.savefig(f'pdf/Data/{name}/trajectory.png', bbox_inches='tight')


def compare_velocity(data, vel, name):
    t = (data['timestamp'] - data['timestamp'][0]) / 1000


    fig,ax = plt.subplots(3)

    ax[0].plot(t, data['stateEstimate.vx'], '-', label='data')
    ax[0].plot(t, vel[:,0], '-', label='propagate')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('velocity [m/s]')
    ax[0].set_title('velocity X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[1].plot(t, data['stateEstimate.vy'], '-', label='data')
    ax[1].plot(t, vel[:,1], '-', label='propagate')
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('velocity [m/s]')
    ax[1].set_title('velocity Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[2].plot(t, data['stateEstimate.vz'], '-', label='data')
    ax[2].plot(t, vel[:,2], '-', label='propagate')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('velocity [m/s]')
    ax[2].set_title('velocity Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Data/{name}/velocity.png', bbox_inches='tight')  


def compare_quaternions(data, quaternions, name):
    t = (data['timestamp'] - data['timestamp'][0]) / 1000

    fig, ax = plt.subplots(4)

    ax[0].plot(t, data['stateEstimate.qw'], '-', label='data')
    ax[0].plot(t, quaternions[:,0], '-', label='propagate')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('quaternions')
    ax[0].set_title('quaternions W')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[1].plot(t, data['stateEstimate.qx'], '-', label='data')
    ax[1].plot(t, quaternions[:,1], '-', label='propagate')
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('quaternions')
    ax[1].set_title('quaternions X')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[2].plot(t, data['stateEstimate.qy'], '-', label='data')
    ax[2].plot(t, quaternions[:,2], '-', label='propagate')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('quaternions')
    ax[2].set_title('quaternions Y')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[3].plot(t, data['stateEstimate.qz'], '-', label='data')
    ax[3].plot(t, quaternions[:,3], '-', label='propagate')
    ax[3].set_xlabel('timestamp [s]')
    ax[3].set_ylabel('quaternions')
    ax[3].set_title('quaternions Z')
    ax[3].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()


    plt.savefig(f'pdf/Data/{name}/quaternions.png', bbox_inches='tight')  
def compare_acceleration(data, acc, name):
    t = (data['timestamp'] - data['timestamp'][0]) / 1000

    fig,ax = plt.subplots(3)

    ax[0].plot(t, data['acc.x'], '-', label='data')
    ax[0].plot(t, acc[:,0], '-', label='propagate')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('acceleration [g]')
    ax[0].set_title('acceleration X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[1].plot(t, data['acc.y'], '-', label='data')
    ax[1].plot(t, acc[:,1], '-', label='propagate')
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('acceleration [g]')
    ax[1].set_title('acceleration Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[2].plot(t, data['acc.z'], '-', label='data')
    ax[2].plot(t, acc[:,2], '-', label='propagate')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('acceleration [g]')
    ax[2].set_title('acceleration Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()


    plt.savefig(f'pdf/Data/{name}/acceleration.png', bbox_inches='tight')


def error_acceleration(data, err_acc, name):
    fig, ax = plt.subplots()
    t = (data['timestamp'][1:] - data['timestamp'][1]) / 1000

    ax.plot(t, err_acc[:,0], '-', label='X')
    ax.plot(t, err_acc[:,1], '-', label='Y')
    ax.plot(t, err_acc[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [s]')
    ax.set_ylabel('acceleration [g]')
    ax.set_title('error acceleration')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Data/{name}/error/error_acceleration.png', bbox_inches='tight')

def error_velocity(data, err_vel, name):
    t = (data['timestamp'][1:] - data['timestamp'][1]) / 1000

    fig, ax = plt.subplots()
    ax.plot(t, err_vel[:,0], '-', label='X')
    ax.plot(t, err_vel[:,1], '-', label='Y')
    ax.plot(t, err_vel[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [s]')
    ax.set_ylabel('velocity [m/s]')
    ax.set_title('error velocity')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.savefig(f'pdf/Data/{name}/error/error_velocity.png', bbox_inches='tight')

    plt.tight_layout()

def error_angular_velocity(data, err_vel_a, name):
    fig, ax = plt. subplots()
    t = (data['timestamp'][1:] - data['timestamp'][1]) / 1000

    ax.plot(t,err_vel_a[:,0], '-', label='X')
    ax.plot(t, err_vel_a[:,1], '-', label='Y')
    ax.plot(t, err_vel_a[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [s]')
    ax.set_ylabel('angular_velocity [°/s]')
    ax.set_title('error angular velocity')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Data/{name}/error/error_angular_velocity.png', bbox_inches='tight')

def error_quaternions(data, err_quat, name):
    t = (data['timestamp'][1:] - data['timestamp'][1]) / 1000

    fig, ax = plt.subplots()

    ax.plot(t, err_quat[:,0], '-', label='W')
    ax.plot(t, err_quat[:,1], '-', label='X')
    ax.plot(t, err_quat[:,2], '-', label='Y')
    ax.plot(t,err_quat[:,3], '-', label='Z')  
    ax.set_xlabel('timestamp [s]')
    ax.set_ylabel('quaternions')
    ax.set_title('error quaternion')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Data/{name}/error/error_quaternions.png', bbox_inches='tight')

def error_position(data, err_pos, name):
    t = (data['timestamp'][1:] - data['timestamp'][1]) / 1000

    fig, ax = plt.subplots()
    ax.plot(t, err_pos[:,0], '-', label='X')
    ax.plot(t, err_pos[:,1], '-', label='Y')
    ax.plot(t, err_pos[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [s]')
    ax.set_ylabel('position')
    ax.set_title('error position')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()
    
    plt.savefig(f'pdf/Data/{name}/error/error_position.png', bbox_inches='tight')


def f_a_plot(data, f, name):
    t = (data['timestamp'][1:] - data['timestamp'][1]) / 1000

    fig, ax = plt.subplots(3)
    ax[0].plot(t, f[:,0], '-')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('N')
    ax[0].set_title('f_a X')


    ax[1].plot(t, f[:,1], '-', label='Y')
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('N')
    ax[1].set_title('f_a Y')

    
    ax[2].plot(t, f[:,2], '-', label='Z')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('N')
    ax[2].set_title('f_a Z')

    plt.tight_layout()


    plt.savefig(f'pdf/Data/{name}/f_a.png',bbox_inches='tight')

def tau_a_plot(data, tau, name):
    t = (data['timestamp'][1:] - data['timestamp'][1]) / 1000

    fig, ax = plt.subplots(3)
    ax[0].plot(t, tau[:,0], '-')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('rad/s²')
    ax[0].set_title('tau_a X')


    ax[1].plot(t, tau[:,1], '-', label='Y')
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('rad/s²')
    ax[1].set_title('tau_a Y')

    
    ax[2].plot(t, tau[:,2], '-', label='Z')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('rad/s²')
    ax[2].set_title('tau_a Z')

    plt.tight_layout()

    plt.savefig(f'pdf/Data/{name}/tau_a.png', bbox_inches='tight')

def errors(data, err_acc, err_vel, err_pos, err_vel_a, err_quaternions, name):

    error_acceleration(data, err_acc, name)
    error_velocity(data, err_vel, name)
    error_angular_velocity(data, err_vel_a, name)
    error_quaternions(data, err_quaternions, name)
    error_position(data, err_pos, name)

def compare_data(data, quaternions, acc, vel, vel_a, pos, name):

    compare_quaternions(data, quaternions, name)
    compare_acceleration(data,acc, name)
    compare_velocity(data, vel, name)
    compare_gyro(data,vel_a, name)
    compare_position(data, pos, name)



def plot_test_data_f(f, pred, test_timestamp):
    t = (test_timestamp - test_timestamp[0]) / 1000

    fig, ax = plt.subplots(3)
    ax[0].plot(t, f[:, 0], '-', label='calculated', alpha=0.7)
    ax[0].plot(t, pred[:, 0], '-', label='predicted', alpha=0.7)
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('N')
    ax[0].set_title('Disturbance Forces X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[1].plot(t, f[:, 1], '-', label='calculated', alpha=0.7)
    ax[1].plot(t, pred[:, 1], '-', label='predicted', alpha=0.7)
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('N')
    ax[1].set_title('Disturbance Forces Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[2].plot(t, f[:,2], '-', label='calculated', alpha=0.7)
    ax[2].plot(t, pred[:, 2], '-', label='predicted', alpha=0.7)
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('N')
    ax[2].set_title('Disturbance Forces Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Supervised learning/predicted f.png', bbox_inches='tight')

def plot_test_data_tau(tau, pred, test_timestamp):
    t = (test_timestamp - test_timestamp[0]) / 1000

    tau = np.array(tau)
    fig, ax = plt.subplots(3)
    ax[0].plot(t, tau[:,0], '-', label='calculated', alpha=0.7)
    ax[0].plot(t, pred[:,3], '-', label='predicted', alpha=0.7)
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('rad/s²')
    ax[0].set_title('Disturbance Torques X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    
    ax[1].plot(t, tau[:,1],'-', label='calculated', alpha=0.7)
    ax[1].plot(t, pred[:, 4], '-', label='predicted', alpha=0.7)
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('rad/s²')
    ax[1].set_title('Disturbance Torques Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[2].plot(t, tau[:,2], '-', label='calculated', alpha=0.7)
    ax[2].plot(t, pred[:,5], '-', label='predicted', alpha=0.7)
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('rad/s²')
    ax[2].set_title('Disturbance Torques Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Supervised learning/predicted tau.png', bbox_inches='tight')


def losses(train_losses, test_losses):

    fig, ax = plt.subplots()
    ax.plot(range(len(train_losses)), train_losses, label = 'training')
    ax.plot(range(len(test_losses)), test_losses, label = 'test')
    ax.set_xlabel('batch')
    ax.set_ylabel('loss')
    ax.set_title('losses')

    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Supervised learning/losses.png', bbox_inches='tight')

def model_error_f(f, pred, test_timestamp, scaler):
    t = (test_timestamp - test_timestamp[0]) / 1000

    fig, ax = plt.subplots(3)
    error_f = np.sqrt((f-pred)**2)[:,:3]
    print(f"f error rows: {np.mean(error_f, axis = 0 )}")
    print(f"overall error f: {np.mean(error_f)}")


    ax[0].plot(t, error_f[:, 0], '-')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('error')
    ax[0].set_title('Disturbance Force X Error')

    ax[1].plot(t, error_f[:, 1], '-') 
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('error')
    ax[1].set_title('Disturbance Force X Error')
   
    ax[2].plot(t, error_f[:, 2], '-')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('error')
    ax[2].set_title('Disturbance Force X Error')

    plt.tight_layout()

    plt.savefig(f'pdf/Supervised learning/predicted f absolut error.png', bbox_inches='tight')


def model_error_tau(tau, pred, test_timestamp, scaler):
    t = (test_timestamp - test_timestamp[0]) / 1000
    fig, ax = plt.subplots(3)
    error_tau = np.sqrt((tau-pred)**2)[:,3:]
    print(f"tau error rows: {np.mean(error_tau, axis = 0 )}")
    print(f"overall error tau: {np.mean(error_tau)}")



    ax[0].plot(t, error_tau[:, 0], '-')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('error')
    ax[0].set_title('Residual Torque X Error')

    ax[1].plot(t, error_tau[:, 1], '-') 
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('error')
    ax[1].set_title('Residual Torque Y Error')

    ax[2].plot(t, error_tau[:, 2], '-')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('error')
    ax[2].set_title('Residual Torque Z Error')

    plt.tight_layout()

    plt.savefig(f'pdf/Supervised learning/predicted tau absolut error.png', bbox_inches='tight')

def plot_test_pred_f(f, pred, test_timestamp):
    t = (test_timestamp - test_timestamp[0]) / 1000

    fig, ax = plt.subplots(3)
    ax[0].plot(t, f[:, 0], '-', label='calculated', alpha=0.7)
    ax[0].plot(t, pred[:, 0], '-', label='predicted', alpha=0.7)
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('N')
    ax[0].set_title('Disturbance Forces X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[1].plot(t, f[:, 1], '-', label='calculated', alpha=0.7)
    ax[1].plot(t, pred[:, 1], '-', label='predicted', alpha=0.7)
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('N')
    ax[1].set_title('Disturbance Forces Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[2].plot(t, f[:,2], '-', label='calculated', alpha=0.7)
    ax[2].plot(t, pred[:, 2], '-', label='predicted', alpha=0.7)
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('N')
    ax[2].set_title('Disturbance Forces Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Decision Tree/predicted f.png', bbox_inches='tight')

def plot_test_pred_tau(tau, pred, test_timestamp ):
    t = (test_timestamp - test_timestamp[0]) / 1000
    tau = np.array(tau)
    fig, ax = plt.subplots(3)
    ax[0].plot(t, tau[:,0], '-', label='calculated', alpha=0.7)
    ax[0].plot(t, pred[:,3], '-', label='predicted', alpha=0.7)
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('rad/s²')
    ax[0].set_title('Disturbance Torques X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    
    ax[1].plot(t, tau[:,1],'-', label='calculated', alpha=0.7)
    ax[1].plot(t, pred[:, 4], '-', label='predicted', alpha=0.7)
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('rad/s²')
    ax[1].set_title('Disturbance Torques Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[2].plot(t, tau[:,2], '-', label='calculated', alpha=0.7)
    ax[2].plot(t, pred[:,5], '-', label='predicted', alpha=0.7)
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('rad/s²')
    ax[2].set_title('Disturbance Torques Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Decision Tree/predicted tau.png', bbox_inches='tight')

def tree_losses(results):

    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    plt.plot(x_axis, results['validation_0']['rmse'], label='train')
    plt.plot(x_axis, results['validation_1']['rmse'], label='test')
    plt.title('Decision Tree Loss')

    plt.legend()
    plt.savefig('pdf/Decision Tree/Decision Tree Loss.png')

def tree_error_f(f, pred, test_timestamp):
    t = (test_timestamp - test_timestamp[0]) / 1000

    fig, ax = plt.subplots(3)
    error_f = np.sqrt((f-pred[:,:3])**2)
    print(f"f error rows: {np.mean(error_f, axis = 0 )}")
    print(f"overall error f: {np.mean(error_f)}")


    ax[0].plot(t, error_f[:, 0], '-')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('error')
    ax[0].set_title('Disturbance Force X Error')

    ax[1].plot(t, error_f[:, 1], '-') 
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('error')
    ax[1].set_title('Disturbance Force X Error')
   
    ax[2].plot(t, error_f[:, 2], '-')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('error')
    ax[2].set_title('Disturbance Force X Error')

    plt.tight_layout()

    plt.savefig(f'pdf/Decision Tree/predicted f absolut error.png', bbox_inches='tight')


def tree_error_tau(tau, pred, test_timestamp):
    t = (test_timestamp - test_timestamp[0]) / 1000

    fig, ax = plt.subplots(3)
    error_tau = np.sqrt((tau-pred[:,3:])**2)
    print(f"tau error rows: {np.mean(error_tau, axis = 0 )}")
    print(f"overall error tau: {np.mean(error_tau)}")

    ax[0].plot(t, error_tau[:, 0], '-')
    ax[0].set_xlabel('timestamp [s]')
    ax[0].set_ylabel('error')
    ax[0].set_title('Residual Torque X Error')

    ax[1].plot(t, error_tau[:, 1], '-') 
    ax[1].set_xlabel('timestamp [s]')
    ax[1].set_ylabel('error')
    ax[1].set_title('Residual Torque Y Error')

    ax[2].plot(t, error_tau[:, 2], '-')
    ax[2].set_xlabel('timestamp [s]')
    ax[2].set_ylabel('error')
    ax[2].set_title('Residual Torque Z Error')

    plt.tight_layout()

    plt.savefig(f'pdf/Decision Tree/predicted tau absolut error.png', bbox_inches='tight') 



