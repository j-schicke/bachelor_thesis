# -*- coding: utf-8 -*-
"""
plotting a generic USD log
"""
import cfusdlog
import matplotlib.pyplot as plt
import argparse
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
    t = (data['timestamp'] - start_time) / 1000
    ax.scatter(t, t*0)
    ax.set_title('fixedFrequency')
    print(data.keys())
    crs = mplcursors.cursor(hover=True)
    crs.connect("add", functools.partial(showAnnotation, data))
    ax.set_xlabel('Time [s]')

    plt.show()

def compare_gyro(data, vel_a):

    fig,ax = plt.subplots(2)

    ax[0].plot(data['timestamp'], data['gyro.x'], '-', label='X')
    ax[0].plot(data['timestamp'], data['gyro.y'], '-', label='Y')
    ax[0].plot(data['timestamp'], data['gyro.z'], '-', label='Z')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('Gyroscope [°/s]')
    ax[0].set_title('data angular_velocity')
    ax[0].legend(loc=9, ncol=3, borderaxespad=0.)

    ax[1].plot(data['timestamp'], vel_a[:,0], '-', label='X')
    ax[1].plot(data['timestamp'], vel_a[:,1], '-', label='Y')
    ax[1].plot(data['timestamp'], vel_a[:,2], '-', label='Z')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('Gyroscope [°/s]')
    ax[1].set_title('output angular_velocity')
    ax[1].legend(loc=9, ncol=3, borderaxespad=0.)

    plt.savefig('pdf/angular_velocity.pdf')  


def trajectory(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    zline = data['stateEstimate.z']
    xline = data['stateEstimate.x']
    yline = data['stateEstimate.y']
    ax.plot3D(xline, yline, zline, 'gray')
    t = (data['timestamp'] - start_time) / 1000

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    fig.show()

def compare_velocity(data, vel):

    fig,ax = plt.subplots(2)

    ax[0].plot(data['timestamp'], data['stateEstimate.vx'], '-', label='X')
    ax[0].plot(data['timestamp'], data['stateEstimate.vy'], '-', label='Y')
    ax[0].plot(data['timestamp'], data['stateEstimate.vz'], '-', label='Z')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('velocity [m/s]')
    ax[0].set_title('data velocity')
    ax[0].legend(loc=9, ncol=3, borderaxespad=0.)

    ax[1].plot(data['timestamp'], vel[:,0], '-', label='X')
    ax[1].plot(data['timestamp'], vel[:,1], '-', label='Y')
    ax[1].plot(data['timestamp'], vel[:,2], '-', label='Z')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('velocity [m/s]')
    ax[1].set_title('output velocity')
    ax[1].legend(loc=9, ncol=3, borderaxespad=0.)

    plt.savefig('pdf/velocity.pdf')  


def rpm(data):
    fig, ax = plt.subplots()
    plt.plot(data['timestamp'], data['rpm.m1'], '-', label='1')
    plt.plot(data['timestamp'], data['rpm.m2'], '-', label='2')
    plt.plot(data['timestamp'], data['rpm.m3'], '-', label='3')
    plt.plot(data['timestamp'], data['rpm.m4'], '-', label='4')

    plt.xlabel('timestamp [ms]')
    plt.ylabel('rotor speed')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)

    plt.show()

def compare_quaternions(data, quaternions):

    fig, ax = plt.subplots(2)

    ax[0].plot(data['timestamp'], data['stateEstimate.qw'], '-', label='W')
    ax[0].plot(data['timestamp'], data['stateEstimate.qx'], '-', label='X')
    ax[0].plot(data['timestamp'], data['stateEstimate.qy'], '-', label='Y')
    ax[0].plot(data['timestamp'], data['stateEstimate.qz'], '-', label='Z')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('quaternions')
    ax[0].set_title('data quaternions')
    ax[0].legend(loc=9, ncol=3, borderaxespad=0.)

    ax[1].plot(data['timestamp'], quaternions[:,0], '-', label='W')
    ax[1].plot(data['timestamp'], quaternions[:,1], '-', label='X')
    ax[1].plot(data['timestamp'], quaternions[:,2], '-', label='Y')
    ax[1].plot(data['timestamp'], quaternions[:,3], '-', label='Z')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('quaternions')
    ax[1].set_title('output quaternions')
    ax[1].legend(loc=9, ncol=3, borderaxespad=0.)

    plt.savefig('pdf/quaternions.pdf')  



def compare_acceleration(data, acc):

    fig,ax = plt.subplots(2)
    ax[0].plot(data['timestamp'], data['acc.x'], '-', label='X')
    ax[0].plot(data['timestamp'], data['acc.y'], '-', label='Y')
    ax[0].plot(data['timestamp'], data['acc.z'], '-', label='Z')
    ax[0].set_xlabel('timestamp [ms]')
    ax[0].set_ylabel('acceleration [g]')
    ax[0].set_title('data acceleration')
    ax[0].legend(loc=9, ncol=3, borderaxespad=0.)

    ax[1].plot(data['timestamp'], acc[:,0], '-', label='X')
    ax[1].plot(data['timestamp'], acc[:,1], '-', label='Y')
    ax[1].plot(data['timestamp'], acc[:,2], '-', label='Z')
    ax[1].set_xlabel('timestamp [ms]')
    ax[1].set_ylabel('acceleration [g]')
    ax[1].set_title('output acceleration')
    ax[1].legend(loc=9, ncol=3, borderaxespad=0.)

    plt.savefig('pdf/acceleration.pdf')


def error_acceleration(data, err_acc):
    fig, ax = plt.subplots()
    ax.plot(data['timestamp'], err_acc[:,0], '-', label='X')
    ax.plot(data['timestamp'], err_acc[:,1], '-', label='Y')
    ax.plot(data['timestamp'], err_acc[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [ms]')
    ax.set_ylabel('acceleration [m/s²]')
    ax.set_title('error acceleration')
    ax.legend(loc=9, ncol=3, borderaxespad=0.)

    plt.savefig('pdf/error/error_acceleration.pdf')

def error_velocity(data, err_vel):
    fig, ax = plt.subplots()
    ax.plot(data['timestamp'], err_vel[:,0], '-', label='X')
    ax.plot(data['timestamp'], err_vel[:,1], '-', label='Y')
    ax.plot(data['timestamp'], err_vel[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [ms]')
    ax.set_ylabel('velocity [m/s]')
    ax.set_title('error velocity')
    ax.legend(loc=9, ncol=3, borderaxespad=0.)
    plt.savefig('pdf/error/error_velocity.pdf')

def error_angular_velocity(data, err_vel_a):
    fig, ax = plt. subplots()

    ax.plot(data['timestamp'],err_vel_a[:,0], '-', label='X')
    ax.plot(data['timestamp'], err_vel_a[:,1], '-', label='Y')
    ax.plot(data['timestamp'], err_vel_a[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [ms]')
    ax.set_ylabel('angular_velocity [°/s]')
    ax.set_title('error angular velocity')
    ax.legend(loc=9, ncol=3, borderaxespad=0.)

    plt.savefig('pdf/error/error_angular_velocity.pdf')

def error_quaternions(data, err_quat):

    fig, ax = plt.subplots()

    ax.plot(data['timestamp'], err_quat[:,0], '-', label='W')
    ax.plot(data['timestamp'], err_quat[:,1], '-', label='X')
    ax.plot(data['timestamp'], err_quat[:,2], '-', label='Y')
    ax.plot(data['timestamp'], err_quat[:,3], '-', label='Z')  
    ax.set_xlabel('timestamp [ms]')
    ax.set_ylabel('quaternions')
    ax.set_title('error quaternion')
    ax.legend(loc=9, ncol=3, borderaxespad=0.)

    plt.savefig('pdf/error/error_quaternions.pdf')

  
def error_position(data, err_pos):
    fig, ax = plt.subplots()
    ax.plot(data['timestamp'], err_pos[:,0], '-', label='X')
    ax.plot(data['timestamp'], err_pos[:,1], '-', label='Y')
    ax.plot(data['timestamp'], err_pos[:,2], '-', label='Z')
    ax.set_xlabel('timestamp [ms]')
    ax.set_ylabel('position')
    ax.set_title('error position')
    ax.legend(loc=9, ncol=3, borderaxespad=0.)
    
    plt.savefig('pdf/error/error_position.pdf')




def errors(data, acc, vel, vel_a, quaternions, pos):
    err_acc = acc - np.array([data['acc.x'], data['acc.y'], data['acc.z']]).T
    err_vel = vel - np.array([data['stateEstimate.vx'], data['stateEstimate.vy'], data['stateEstimate.vz']]).T
    err_vel_a = vel_a - np.array([data['gyro.x'], data['gyro.y'], data['gyro.z']]).T
    err_quat = quaternions - np.array([data['stateEstimate.qw'],data['stateEstimate.qx'], data['stateEstimate.qy'], data['stateEstimate.qz']]).T
    err_pos = pos - np.array([data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z']]).T

    error_acceleration(data, err_acc)
    error_velocity(data, err_vel)
    error_angular_velocity(data, err_vel_a)
    error_quaternions(data, err_quat)
    error_position(data, err_pos)

def compare_data(data, quaternions, acc, vel, vel_a):

    compare_quaternions(data, quaternions)
    compare_acceleration(data,acc)
    compare_velocity(data, vel)
    compare_gyro(data,vel_a)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("file_usd")
    #args = parser.parse_args()
    #data_usd = cfusdlog.decode(args.file_usd)

    data_usd = cfusdlog.decode("log01")
    data = data_usd['fixedFrequency']
    # decode binary log data


    # find start time
    data = data_usd['fixedFrequency']
    start_time = data['timestamp'][0]
    # new figure
