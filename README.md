# Introduction
This is the python implementation of a model predictive path generation method. Only numpy is used in the implementaion. It can be easily incorporated to other application. 

# QuickStart
```python

from Polyminal_TrajectoryGenerator_OneShotMethod import TrajectoryGenerator

PathGenerator = TrajectoryGenerator()

## coordinate
# Y    
# ^   /
# |  /
# | / <theta>
# o -- -- -- >X

x_0 = 0.0 # initial x position
y_0 = 0.0 # initial y position
theta_0 = 30.0 *np.pi/180  # initial heading angle of the vehicle 
kappa_0 = 30.0 *np.pi/180  # initial steering angle  
initial_state = [x_0, y_0, theta_0, kappa_0] 
    
x_f = 8.0 # final x position
y_f = 13.0 # final y position
theta_f = 90.0 *np.pi/180  # final heading angle of the vehicle 
kappa_f = 0.0 *np.pi/180  # final steering angle 
final_state = [x_f, y_f, theta_f, kappa_f] 

# compute trajectory in a list of point
traject = PathGenerator.compute_spline(initial_state, final_state)
```
You can find a simple demo about how to use it in the file.

![](https://raw.githubusercontent.com/gn02596703/pyTrajectoryGenerator/master/document_source/result_2.png)
![](https://raw.githubusercontent.com/gn02596703/pyTrajectoryGenerator/master/document_source/result_1.png)


