# -*- coding: utf-8 -*-
"""
plotting a generic USD log
"""
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import functools
from mpl_toolkits.mplot3d import Axes3D



def showAnnotation(data, sel):
    idx = sel.target.index
    sel.annotation.set_text(
        "\n".join(['{}: {}'.format(key, data[key][idx]) for key in data.keys()]))

def plot_all(data):
    fig, ax = plt.subplots()
    t = (data['timestamp'] - data['timestamp'][0]) / 1000
    ax.scatter(t, t*0)
    ax.set_title('fixedFrequency')
    print(data.keys())
    crs = mplcursors.cursor(hover=True)
    crs.connect("add", functools.partial(showAnnotation, data))
    ax.set_xlabel('Time [s]')

    plt.show()

def compare_gyro(data, vel_a, name):

    fig,ax = plt.subplots(3)

    ax[0].plot(data['timestamp'], data['gyro.x'], '-', label='data')
    ax[0].plot(data['timestamp'], vel_a[:,0], '-', label='propagated')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('Gyroscope [°/s]')
    ax[0].set_title('angular velocity X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)



    ax[1].plot(data['timestamp'], data['gyro.y'], '-', label='data')
    ax[1].plot(data['timestamp'], vel_a[:,1], '-', label='propagated')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('Gyroscope [°/s]')
    ax[1].set_title('angular velocity Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[2].plot(data['timestamp'], data['gyro.z'], '-', label='data')
    ax[2].plot(data['timestamp'], vel_a[:,2], '-', label='propagated')
    ax[2].set_xlabel('timestamp [ms]')
    ax[2].set_ylabel('Gyroscope [°/s]')
    ax[2].set_title('angular velocity Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    
    plt.tight_layout()

    plt.savefig(f'pdf/{name}/angular_velocity.png', bbox_inches='tight')  

def compare_position(data, pos, name):

    fig,ax = plt.subplots(3)

    ax[0].plot(data['timestamp'], data['stateEstimate.x'], '-', label='data')
    ax[0].plot(data['timestamp'], pos[:,0], '-', label='propagated')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('position')
    ax[0].set_title('position X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[1].plot(data['timestamp'], data['stateEstimate.y'], '-', label='data')
    ax[1].plot(data['timestamp'], pos[:,1], '-', label='propagated')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('position')
    ax[1].set_title('position Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[2].plot(data['timestamp'], data['stateEstimate.z'], '-', label='data')
    ax[2].plot(data['timestamp'], pos[:,2], '-', label='propagated')
    ax[2].set_xlabel('timestamp [ms]')
    ax[2].set_ylabel('position')
    ax[2].set_title('position Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()


    plt.savefig(f'pdf/{name}/position.png', bbox_inches='tight')  


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

    plt.savefig(f'pdf/{name}/trajectory_x_y_plane.png', bbox_inches='tight')

def trajectory(data, name):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xline = data['stateEstimate.x']
    yline = data['stateEstimate.y']
    zline = data['stateEstimate.z']

    ax.plot3D(xline, yline, zline, 'gray')
#    ax.view_init(-60, 60)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.tight_layout()

    plt.savefig(f'pdf/{name}/trajectory.png', bbox_inches='tight')


def compare_velocity(data, vel, name):

    fig,ax = plt.subplots(3)

    ax[0].plot(data['timestamp'], data['stateEstimate.vx'], '-', label='data')
    ax[0].plot(data['timestamp'], vel[:,0], '-', label='propagate')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('velocity [m/s]')
    ax[0].set_title('velocity X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[1].plot(data['timestamp'], data['stateEstimate.vy'], '-', label='data')
    ax[1].plot(data['timestamp'], vel[:,1], '-', label='propagate')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('velocity [m/s]')
    ax[1].set_title('velocity Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[2].plot(data['timestamp'], data['stateEstimate.vz'], '-', label='data')
    ax[2].plot(data['timestamp'], vel[:,2], '-', label='propagate')
    ax[2].set_xlabel('timestamp [ms]')
    ax[2].set_ylabel('velocity [m/s]')
    ax[2].set_title('velocity Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/{name}/velocity.png', bbox_inches='tight')  


def rpm(data):
    fig, ax = plt.subplots()
    plt.plot(data['timestamp'], data['rpm.m1'], '-', label='1')
    plt.plot(data['timestamp'], data['rpm.m2'], '-', label='2')
    plt.plot(data['timestamp'], data['rpm.m3'], '-', label='3')
    plt.plot(data['timestamp'], data['rpm.m4'], '-', label='4')

    plt.xlabel('timestamp [ms]')
    plt.ylabel('rotor speed')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.show()

def compare_quaternions(data, quaternions, name):

    fig, ax = plt.subplots(4)

    ax[0].plot(data['timestamp'], data['stateEstimate.qw'], '-', label='data')
    ax[0].plot(data['timestamp'], quaternions[:,0], '-', label='propagate')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('quaternions')
    ax[0].set_title('quaternions W')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[1].plot(data['timestamp'], data['stateEstimate.qx'], '-', label='data')
    ax[1].plot(data['timestamp'], quaternions[:,1], '-', label='propagate')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('quaternions')
    ax[1].set_title('quaternions X')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[2].plot(data['timestamp'], data['stateEstimate.qy'], '-', label='data')
    ax[2].plot(data['timestamp'], quaternions[:,2], '-', label='propagate')
    ax[2].set_xlabel('timestamp [ms]')
    ax[2].set_ylabel('quaternions')
    ax[2].set_title('quaternions Y')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[3].plot(data['timestamp'], data['stateEstimate.qz'], '-', label='data')
    ax[3].plot(data['timestamp'], quaternions[:,3], '-', label='propagate')
    ax[3].set_xlabel('timestamp [ms]')
    ax[3].set_ylabel('quaternions')
    ax[3].set_title('quaternions Z')
    ax[3].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()


    plt.savefig(f'pdf/{name}/quaternions.png', bbox_inches='tight')  



def compare_acceleration(data, acc, name):

    fig,ax = plt.subplots(3)

    ax[0].plot(data['timestamp'], data['acc.x'], '-', label='data')
    ax[0].plot(data['timestamp'], acc[:,0], '-', label='propagate')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('acceleration [g]')
    ax[0].set_title('acceleration X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[1].plot(data['timestamp'], data['acc.y'], '-', label='data')
    ax[1].plot(data['timestamp'], acc[:,1], '-', label='propagate')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('acceleration [g]')
    ax[1].set_title('acceleration Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[2].plot(data['timestamp'], data['acc.z'], '-', label='data')
    ax[2].plot(data['timestamp'], acc[:,2], '-', label='propagate')
    ax[2].set_xlabel('timestamp [ms]')
    ax[2].set_ylabel('acceleration [g]')
    ax[2].set_title('acceleration Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()


    plt.savefig(f'pdf/{name}/acceleration.png', bbox_inches='tight')


def error_acceleration(data, err_acc, name):
    fig, ax = plt.subplots()
    ax.plot(data['timestamp'][1:], err_acc[:,0], '-', label='X')
    ax.plot(data['timestamp'][1:], err_acc[:,1], '-', label='Y')
    ax.plot(data['timestamp'][1:], err_acc[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [ms]')
    ax.set_ylabel('acceleration [g]')
    ax.set_title('error acceleration')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/{name}/error/error_acceleration.png', bbox_inches='tight')

def error_velocity(data, err_vel, name):
    fig, ax = plt.subplots()
    ax.plot(data['timestamp'][1:], err_vel[:,0], '-', label='X')
    ax.plot(data['timestamp'][1:], err_vel[:,1], '-', label='Y')
    ax.plot(data['timestamp'][1:], err_vel[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [ms]')
    ax.set_ylabel('velocity [m/s]')
    ax.set_title('error velocity')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.savefig(f'pdf/{name}/error/error_velocity.png', bbox_inches='tight')

    plt.tight_layout()

def error_angular_velocity(data, err_vel_a, name):
    fig, ax = plt. subplots()

    ax.plot(data['timestamp'][1:],err_vel_a[:,0], '-', label='X')
    ax.plot(data['timestamp'][1:], err_vel_a[:,1], '-', label='Y')
    ax.plot(data['timestamp'][1:], err_vel_a[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [ms]')
    ax.set_ylabel('angular_velocity [°/s]')
    ax.set_title('error angular velocity')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/{name}/error/error_angular_velocity.png', bbox_inches='tight')

def error_quaternions(data, err_quat, name):

    fig, ax = plt.subplots()

    ax.plot(data['timestamp'][1:], err_quat[:,0], '-', label='W')
    ax.plot(data['timestamp'][1:], err_quat[:,1], '-', label='X')
    ax.plot(data['timestamp'][1:], err_quat[:,2], '-', label='Y')
    ax.plot(data['timestamp'][1:], err_quat[:,3], '-', label='Z')  
    ax.set_xlabel('timestamp [ms]')
    ax.set_ylabel('quaternions')
    ax.set_title('error quaternion')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/{name}/error/error_quaternions.png', bbox_inches='tight')

def error_position(data, err_pos, name):
    fig, ax = plt.subplots()
    ax.plot(data['timestamp'][1:], err_pos[:,0], '-', label='X')
    ax.plot(data['timestamp'][1:], err_pos[:,1], '-', label='Y')
    ax.plot(data['timestamp'][1:], err_pos[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [ms]')
    ax.set_ylabel('position')
    ax.set_title('error position')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()
    
    plt.savefig(f'pdf/{name}/error/error_position.png', bbox_inches='tight')


def f_a_plot(data, f, name):

    # fig, ax = plt.subplots()
    # ax.plot(data['timestamp'][1:], f[:,0], '-', label='X')
    # ax.plot(data['timestamp'][1:], f[:,1], '-', label='Y')
    # ax.plot(data['timestamp'][1:], f[:,2], '-', label='Z')
    # ax.set_xlabel('timestamp [ms]')
    # ax.set_ylabel('N')
    # ax.set_title('f_a')

    fig, ax = plt.subplots(3)
    ax[0].plot(data['timestamp'][1:], f[:,0], '-')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('N')
    ax[0].set_title('f_a X')


    ax[1].plot(data['timestamp'][1:], f[:,1], '-', label='Y')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('N')
    ax[1].set_title('f_a Y')

    
    ax[2].plot(data['timestamp'][1:], f[:,2], '-', label='Z')
    ax[2].set_xlabel('timestamp [ms]')
    ax[2].set_ylabel('N')
    ax[2].set_title('f_a Z')

    plt.tight_layout()


    plt.savefig(f'pdf/{name}/f_a.png',bbox_inches='tight')

def tau_a_plot(data, tau, name):
    #fig, ax = plt.subplots()

    # ax.plot(data['timestamp'][1:], tau[:,0], '-', label='X')
    # ax.plot(data['timestamp'][1:], tau[:,1], '-', label='Y')
    # ax.plot(data['timestamp'][1:], tau[:,2], '-', label='Z')
    # ax.set_xlabel('timestamp [ms]')
    # ax.set_ylabel('rad/s²')
    # ax.set_title('tau_a')

    fig, ax = plt.subplots(3)
    ax[0].plot(data['timestamp'][1:], tau[:,0], '-')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('rad/s²')
    ax[0].set_title('tau_a X')


    ax[1].plot(data['timestamp'][1:], tau[:,1], '-', label='Y')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('rad/s²')
    ax[1].set_title('tau_a Y')

    
    ax[2].plot(data['timestamp'][1:], tau[:,2], '-', label='Z')
    ax[2].set_xlabel('timestamp [ms]')
    ax[2].set_ylabel('rad/s²')
    ax[2].set_title('tau_a Z')

    plt.tight_layout()

    plt.savefig(f'pdf/{name}/tau_a.png', bbox_inches='tight')

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

    fig, ax = plt.subplots(3)
    ax[0].plot(test_timestamp, f[:, 0], '-', label='calculated', alpha=0.7)
    ax[0].plot(test_timestamp, pred[:, 0], '-', label='predicted', alpha=0.7)
    ax[0].set_xlabel('test data')
    ax[0].set_ylabel('N')
    ax[0].set_title('Disturbance Forces X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[1].plot(test_timestamp, f[:, 1], '-', label='calculated', alpha=0.7)
    ax[1].plot(test_timestamp, pred[:, 1], '-', label='predicted', alpha=0.7)
    ax[1].set_xlabel('test data')
    ax[1].set_ylabel('N')
    ax[1].set_title('Disturbance Forces Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[2].plot(test_timestamp, f[:,2], '-', label='calculated', alpha=0.7)
    ax[2].plot(test_timestamp, pred[:, 2], '-', label='predicted', alpha=0.7)
    ax[2].set_xlabel('test data')
    ax[2].set_ylabel('N')
    ax[2].set_title('Disturbance Forces Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Supervised learning/predictited f.png', bbox_inches='tight')

def plot_test_data_tau(tau, pred, test_timestamp):
    tau = np.array(tau)
    fig, ax = plt.subplots(3)
    ax[0].plot(test_timestamp, tau[:,0], '-', label='calculated', alpha=0.7)
    ax[0].plot(test_timestamp, pred[:,3], '-', label='predicted', alpha=0.7)
    ax[0].set_xlabel('test data')
    ax[0].set_ylabel('rad/s²')
    ax[0].set_title('Disturbance Torques X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    
    ax[1].plot(test_timestamp, tau[:,1],'-', label='calculated', alpha=0.7)
    ax[1].plot(test_timestamp, pred[:, 4], '-', label='predicted', alpha=0.7)
    ax[1].set_xlabel('test data')
    ax[1].set_ylabel('rad/s²')
    ax[1].set_title('Disturbance Torques Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[2].plot(test_timestamp, tau[:,2], '-', label='calculated', alpha=0.7)
    ax[2].plot(test_timestamp, pred[:,5], '-', label='predicted', alpha=0.7)
    ax[2].set_xlabel('test data')
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



def plot_test_pred_f(f, pred, test_timestamp):

    fig, ax = plt.subplots(3)
    ax[0].plot(test_timestamp, f[:, 0], '-', label='calculated', alpha=0.7)
    ax[0].plot(test_timestamp, pred[:, 0], '-', label='predicted', alpha=0.7)
    ax[0].set_xlabel('timestamp')
    ax[0].set_ylabel('N')
    ax[0].set_title('Disturbance Forces X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[1].plot(test_timestamp, f[:, 1], '-', label='calculated', alpha=0.7)
    ax[1].plot(test_timestamp, pred[:, 1], '-', label='predicted', alpha=0.7)
    ax[1].set_xlabel('timestamp')
    ax[1].set_ylabel('N')
    ax[1].set_title('Disturbance Forces Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    ax[2].plot(test_timestamp, f[:,2], '-', label='calculated', alpha=0.7)
    ax[2].plot(test_timestamp, pred[:, 2], '-', label='predicted', alpha=0.7)
    ax[2].set_xlabel('timestamp')
    ax[2].set_ylabel('N')
    ax[2].set_title('Disturbance Forces Z')
    ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Decision Tree/predictited f.png', bbox_inches='tight')

def plot_test_pred_tau(tau, pred, test_timestamp):
    t = test_timestamp
    tau = np.array(tau)
    fig, ax = plt.subplots(3)
    ax[0].plot(t, tau[:,0], '-', label='calculated', alpha=0.7)
    ax[0].plot(t, pred[:,3], '-', label='predicted', alpha=0.7)
    ax[0].set_xlabel('timestamp')
    ax[0].set_ylabel('rad/s²')
    ax[0].set_title('Disturbance Torques X')
    ax[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    
    ax[1].plot(t, tau[:,1],'-', label='calculated', alpha=0.7)
    ax[1].plot(t, pred[:, 4], '-', label='predicted', alpha=0.7)
    ax[1].set_xlabel('timestamp')
    ax[1].set_ylabel('rad/s²')
    ax[1].set_title('Disturbance Torques Y')
    ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax[2].plot(t, tau[:,2], '-', label='calculated', alpha=0.7)
    ax[2].plot(t, pred[:,5], '-', label='predicted', alpha=0.7)
    ax[2].set_xlabel('timestamp')
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
    fig, ax = plt.subplots(3)
    error_f = f-pred[:,:3]

    ax[0].plot(test_timestamp, error_f[:, 0], '-')
    ax[1].plot(test_timestamp, error_f[:, 1], '-')    
    ax[2].plot(test_timestamp, error_f[:, 2], '-')
    ax.set_xlabel('timestamp')
    ax.set_ylabel('error')
    ax.set_title('Disturbance Forces Error')

    # ax[1].plot(test_timestamp, f[:, 1], '-', label='calculated', alpha=0.7)
    # ax[1].plot(test_timestamp, pred[:, 1], '-', label='predicted', alpha=0.7)
    # ax[1].set_xlabel('time')
    # ax[1].set_ylabel('N')
    # ax[1].set_title('Disturbance Forces Y')
    # ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    # ax[2].plot(test_timestamp, f[:,2], '-', label='calculated', alpha=0.7)
    # ax[2].plot(test_timestamp, pred[:, 2], '-', label='predicted', alpha=0.7)
    # ax[2].set_xlabel('time')
    # ax[2].set_ylabel('N')
    # ax[2].set_title('Disturbance Forces Z')
    # ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Decision Tree/predictited f absolut error.png', bbox_inches='tight')


def tree_error_tau(tau, pred, test_timestamp):
    fig, ax = plt.subplots(3)
    error_tau = tau-pred[:,3:]

    ax[0].plot(test_timestamp, error_tau[:, 0], '-')
    ax[1].plot(test_timestamp, error_tau[:, 1], '-')    
    ax[2].plot(test_timestamp, error_tau[:, 2], '-')
    ax.set_xlabel('timestamp')
    ax.set_ylabel('error')
    ax.set_title('Disturbance Torques Error')

    # ax[1].plot(test_timestamp, f[:, 1], '-', label='calculated', alpha=0.7)
    # ax[1].plot(test_timestamp, pred[:, 1], '-', label='predicted', alpha=0.7)
    # ax[1].set_xlabel('time')
    # ax[1].set_ylabel('N')
    # ax[1].set_title('Disturbance Forces Y')
    # ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


    # ax[2].plot(test_timestamp, f[:,2], '-', label='calculated', alpha=0.7)
    # ax[2].plot(test_timestamp, pred[:, 2], '-', label='predicted', alpha=0.7)
    # ax[2].set_xlabel('time')
    # ax[2].set_ylabel('N')
    # ax[2].set_title('Disturbance Forces Z')
    # ax[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f'pdf/Decision Tree/predictited tau absolut error.png', bbox_inches='tight')


