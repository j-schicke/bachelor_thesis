# bachelor_thesis

## Requirements

Python version 3.8.10

### Packages:

* rowan 1.3.0
* numpy 1.19.0
* sklearn 1.2.0
* matplotlib 3.6.2
* mplcursors 0.5.2
* torch 1.13.1

## Code

### Basic Forward Propagation
The basic forward propagation of the quadrotor model propagate the acceleration, velocity, position, quaternions, angular velocity and angular acceleration. The input is a data file of the flight of a quadrotor. The data needs to include the timestamps, the position, the velocity, the acceleration, the anggular velocity, the quaternions, the PWM motor signals and the battery voltage. 

### Residual model
The residual model calculate the f_a and tau_a of the input data. Furthermore a model is trained with the data of a flight of the quadrotor with the f_a and tau_a as predictions. The input data needs to have the timestamps, the angular velocity, the acceleration, the quaternions, the PWM motor signals and the battery voltage.

### Model
The model file build a Neural Network. The model has input layer has the size 25, two hidden layers and the output layer's size is 6. The model is a linear regression andd the linear activation function is Relu. The model is trained in 25 Epoches, with 70% of the input data.

### Plot data
The file has all functions to plot the output. There are functions to compare either all or one output data from the propagation of the model. It plots the data of the output of the propagation, the trajectory of the flight, loss of the trained model and the calculated f_a and tau_a. It also has two functions to compare the predicted f_a and tau_a to the calculated f_a and tau_a.

### Config file
In the Config file are the values for the mass, inertia matrix, arms length and the arm. Furthermore in this file are the values to convert the values to another unit.

## Plots

The folder pdf includes the plot of the loss of the training and the validation of the linear regression and an subfolder for each flight of the quadrotor. In each flight folder are a 3d plot of the trajectory, the calculated f_a and tau_a and five plots that compare the acceleration, the velocity, the angular velocity, the position and the quaternions. Additionally there is a subfolder with the name error, in there are the five plots that shows the error of the acceleration, the velocity, the angular velocity, the position and the quaternions.