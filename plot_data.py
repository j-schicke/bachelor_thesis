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

def gyro(data):
    fig, ax = plt.subplots()
    plt.plot(data['timestamp'], data['gyro.x'], '-', label='X')
    plt.plot(data['timestamp'], data['gyro.y'], '-', label='Y')
    plt.plot(data['timestamp'], data['gyro.z'], '-', label='Z')
    plt.xlabel('timestamp [ms]')
    plt.ylabel('Gyroscope [°/s]')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)

    plt.show()

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

    plt.show()

def velocity(data):
    fig, ax = plt.subplots()
    plt.plot(data['timestamp'], data['stateEstimate.vx'], '-', label='X')
    plt.plot(data['timestamp'], data['stateEstimate.vy'], '-', label='Y')
    plt.plot(data['timestamp'], data['stateEstimate.vz'], '-', label='Z')
    plt.xlabel('timestamp [ms]')
    plt.ylabel('Velocity [m/s]')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)

    plt.show()

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

def quaternions(data):
    fig, ax = plt.subplots()
    plt.plot(data['timestamp'], data['stateEstimate.qx'], '-', label='x')
    plt.plot(data['timestamp'], data['stateEstimate.qy'], '-', label='y')
    plt.plot(data['timestamp'], data['stateEstimate.qz'], '-', label='z')
    plt.plot(data['timestamp'], data['stateEstimate.qw'], '-', label='w')

    plt.xlabel('timestamp [ms]')
    plt.ylabel('quaternions')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)

    plt.show()

def acceleration(data):

    fig, ax = plt.subplots()
    plt.plot(data['timestamp'], data['acc.x'], '-', label='X')
    plt.plot(data['timestamp'], data['acc.y'], '-', label='Y')
    plt.plot(data['timestamp'], data['acc.z'], '-', label='Z')
    plt.xlabel('timestamp [ms]')
    plt.ylabel('acceleration [m/s²]')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)

    plt.show()



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
    del data['pwm.m1_pwm']
    del data['pwm.m2_pwm']
    del data['pwm.m3_pwm']
    del data['pwm.m4_pwm']
    del data['pm.vbatMV']

    acceleration(data)
