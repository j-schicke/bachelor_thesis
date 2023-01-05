Meetings:
06.01-12.01:

progress:
- add 1 g-unit to data acceleration
- wrote Read Me file
- plot avg. loss, compare pred and calculated f_a, tau_a
- add tau_a to the model prediction




30.12-05.01:

progress:
- plot test and trainings loss <br  />
- fixed acceleration <br  />
- worked on model for f_a  <br  />
- plot position, tau_a, f_a <br  />
- 

discussion:
- should I use a fewer attributes for the model <br  />
- quaternion error as distance between two quaternions or error = q2*q1⁻¹  <br  />
- acceleration and f_a both 3rd value in the array is different than it should be  <br  />



22.12-29.12:

progress:
-fixed issues for the model except for acceleration<br  />
- first version for a model for f_a

discussion:
- is the acceleration in g-unit? <br  />
- the acceleration output in g-unit is still 10x higher than the acceleration data <br  />



16.12-21.12:

progress;
plot the conparison of the data and model and the error of the model <br  />
calculate f_a and tau_a of the residual model

discussion:

the error of the model is really high (both angular and linear) -> error in the code of newton euler equation <br  />


09.12-15.12:

progress:

plot the data <br /> 
propagate the data <br /> 

next week plans:

compute model error<br /> 

discussion:

the output of the model is very different to the original data (accelerati9on original between 0-1 and output -3000 <br /> 


01.12-08.12: 


progress:

read [2] <br /> 
decode and plot the data <br /> 
implemting basic forward propagation <br /> 


next week plans:

use the forward propagation model with data <br /> 
calculate model error <br /> 


21.11-30.11:
 

progress:

read Multirotor Aerial Vehicle and Minimum Snap  <br /> 
read Trajectory Generation and Control for Quadrotors  <br /> 
read Dynamics Modelling and Linear Control of Quadcopter <br />
beginng implementing basic forward propagation <br /> 


next week plans:

check out the data <br /> 
implementing basic forward propagation <br /> 

discussion:

What is the output data of the forward propagation <br />