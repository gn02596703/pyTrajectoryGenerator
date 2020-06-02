# README of pyTrajectoryGenerator
###### tags: `Self-Driving Car`, `README`

# Introduction
This is the python implementation of a model predictive path generation method. Only numpy is used in the implementaion. It can be easily incorporated to other application. 

# QuickStart
```python
PathGenerator = TrajectoryGenerator()
    
x_0 = 0.0 # initial x position
y_0 = 0.0 # initial y position
theta_0 = 0.0 # heading angle (degree)
kappa_0 = 0.0 # initial steering angle (degree)
initial_state = [x_0, y_0, theta_0, kappa_0] 
    
final_state = [13.607331971206666, 8.3645834995470061, 1.2021703964156283, 0]

# compute trajectory in a list of point
traject = PathGenerator.compute_spline(initial_state, final_state)
```
You can find a simple demo code in the file.

![](https://raw.githubusercontent.com/gn02596703/pyTrajectoryGenerator/master/document_source/result_2.png)
![](https://raw.githubusercontent.com/gn02596703/pyTrajectoryGenerator/master/document_source/result_1.png)


