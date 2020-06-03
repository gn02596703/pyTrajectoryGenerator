"""
The file can generate a smooth trajectory (a list of waypoint) according to 
given initial state and goal state.

The algorithm is mainly based on following publications
"Trajectory Generation For Car-Like Robots Using Cubic Curvature Polynomials",
Nagy and Kelly, 2001 
"Adaptive Model Predictive Motion Planning for Navigation in Complex Environments",
Thomas M. Howard, 2009, (chapter 4)
"Motion Planning for Self-Driving-Cars", Coursera 4 of Self-Driving-Car Specialization
on Coursera

The implementation also references from Matthew O'Kelly's C++ implementation.
You can find the source code in Autoware Repository.
https://gitlab.com/autowarefoundation/autoware.ai/core_planning/-/blob/master/lattice_planner/lib/libtraj_gen.cpp
"""

import numpy as np

# for visualization and debugging
import matplotlib.pyplot as plt
import time
import pdb

class TrajectoryGenerator():

    def __init__(self):
        
        self.is_converge = False
        self.max_iter = 50
       
        # criteria of terminal state
        self.acceptable_dx = 0.01 # m
        self.acceptable_dy = 0.01 # m
        self.acceptable_dtheta = 1 *np.pi/180 # rad
        self.acceptable_dkappa = 1 *np.pi/180 # rad

        # pertubation value
        self.pertub = 0.0001

        # path sampling resolution
        self.sample_resolution = 1 # m
    
    def _initialize_spline(self, initial_state, final_state):
        """
        The function initializes spline paramters 

        :param: final_state: goal of the path [x, y, theta, kappa]
        :return: a, b, c, s: parameter of the spline
        """
        x_f = final_state[0]
        y_f = final_state[1]
        theta_f = final_state[2] # rad
        kappa_f = final_state[3] # rad
        kappa_0 = initial_state[3] # rad

        
        d = np.sqrt(x_f**2 + y_f**2)
        theta_delta = np.abs(theta_f)
        s = d*(( (theta_delta**2) / 5) +1) + 2*theta_delta/5
        
        # Initialization method from Nagy and Kelly, 2001 
        # a = 6*theta_f/(s**2) - 2*kappa_0/s + 4*kappa_f/s
        # c = 0
        # b = 3*(kappa_0 + kappa_f)/(s**2) + 6*theta_f/(s**3)

        # Initilalization method from Thomas M. Howard, 2009
        a = 0.0
        b = 0.0
        c = kappa_f
        
        return a, b, c, s

    def _compute_theta(self, theta, a_p, b_p, c_p, d_p, s):
        theta_final = theta + d_p*(s**4)/4 + \
                    c_p*(s**3)/3 + b_p*(s**2)/2 + \
                    a_p*s

        return theta_final

    def _compute_x(self, x_0, theta_0, a_p, b_p, c_p, d_p, s):
        theta_s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, s/8.0)
        theta_2s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 2*s/8.0)
        theta_3s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 3*s/8.0)
        theta_4s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 4*s/8.0)
        theta_5s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 5*s/8.0)
        theta_6s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 6*s/8.0)
        theta_7s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 7*s/8.0)
        theta_s = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, s)
        
        x_final = x_0 + (np.cos(theta_0) + 4*np.cos(theta_s_8) + 2*np.cos(theta_2s_8) + \
                                        4*np.cos(theta_3s_8) + 2*np.cos(theta_4s_8) + \
                                        4*np.cos(theta_5s_8) + 2*np.cos(theta_6s_8) + \
                                        4*np.cos(theta_7s_8) + np.cos(theta_s))* s/24.0
        return x_final

    def _compute_y(self, y_0, theta_0, a_p, b_p, c_p, d_p, s):
        theta_s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, s/8.0)
        theta_2s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 2*s/8.0)
        theta_3s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 3*s/8.0)
        theta_4s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 4*s/8.0)
        theta_5s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 5*s/8.0)
        theta_6s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 6*s/8.0)
        theta_7s_8 = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, 7*s/8.0)
        theta_s = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, s)
        
        y_final = y_0 + (np.sin(theta_0) + 4*np.sin(theta_s_8) + 2*np.sin(theta_2s_8) + \
                                        4*np.sin(theta_3s_8) + 2*np.sin(theta_4s_8) + \
                                        4*np.sin(theta_5s_8) + 2*np.sin(theta_6s_8) + \
                                        4*np.sin(theta_7s_8) + np.sin(theta_s))* s/24.0
        return y_final

    def _motion_update_one_shot(self, initial_state, a, b, c, s):
        """
        predict the final state according to initial state
        and spline parameter using one shot method

        :param initial_state: initial state of the vehcle [x, y, theta, kappa]
        :param a: spline parameter
        :param b: spline parameter
        :param c: spline parameter
        :param s: spline parameter
        :return final_state_pred: predicted final state 
        """
        x_0 = initial_state[0]
        y_0 = initial_state[1]
        theta = initial_state[2] # rad
        kappa = initial_state[3] # rad
        theta_0 = np.copy(theta)
        kappa_0 = np.copy(kappa)

        a_p = kappa_0
        b_p = -(11*kappa_0 - 18*a + 9*b - 2*c)/(2* s)
        c_p = 9*(2*kappa_0 - 5*a + 4*b - c)/(2* s**2)
        d_p = -9*(kappa_0 - 3*a + 3*b -c)/(2* s**3)
        kappa_final = a_p + b_p*s + c_p*(s**2) + d_p*(s**3)

        
        theta_final = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, s)
        x_final = self._compute_x(x_0, theta_0, a_p, b_p, c_p, d_p, s)
        y_final = self._compute_y(y_0, theta_0, a_p, b_p, c_p, d_p, s)

        final_state_pred = [x_final, y_final, theta_final, kappa_final]

        return final_state_pred

    def _check_converge(self, final_state, final_state_pred):
        """
        check if the optimization result is converged
        """
        x_diff = float(abs(final_state[0] - final_state_pred[0]))
        y_diff = float(abs(final_state[1] - final_state_pred[1]))
        theta_diff = float(abs(final_state[2] - final_state_pred[2]))
        kappa_diff = float(abs(final_state[3] - final_state_pred[3]))

        converge = (x_diff <= self.acceptable_dx) & \
                    (y_diff <= self.acceptable_dy) & \
                    (theta_diff <= self.acceptable_dtheta) & \
                    (kappa_diff <= self.acceptable_dkappa)

        return converge

    def _compute_correction(self, initial_state, final_state, a, b, c, s):
        """
        compute the correction of spline parameter
        """
        pertub = self.pertub
        pertub_s = pertub *10
    
        pred_no_pertub = self._motion_update_one_shot(initial_state, a, b, c, s)
        pred_pertub_a = self._motion_update_one_shot(initial_state, a +pertub, b, c, s)
        pred_pertub_b = self._motion_update_one_shot(initial_state, a, b +pertub, c, s)
        # no need to correct C, C is constrained by kappa_final
        # # pred_pertub_c = self._motion_update_one_shot(initial_state, a, b, c +pertub, s)
        pred_pertub_s = self._motion_update_one_shot(initial_state, a, b, c, s +pertub_s)

        d_state = np.zeros((3,1))
        d_pertub_state = np.zeros((3,3))
        Jacobian = np.zeros((3,3))
        for i in range(0, 3):
            d_pertub_state[i][0] = (final_state[i] - pred_pertub_a[i]) # a
            d_pertub_state[i][1] = (final_state[i] - pred_pertub_b[i]) # b
            # d_pertub_state[i][2] = (final_state[i] - pred_pertub_c[i]) # c (no update)
            d_pertub_state[i][2] = (final_state[i] - pred_pertub_s[i]) # s
            
            d_state[i] = final_state[i] - pred_no_pertub[i]
            
            Jacobian[i][0] = (d_pertub_state[i][0] -  d_state[i])/pertub # a
            Jacobian[i][1] = (d_pertub_state[i][1] -  d_state[i])/pertub # b
            # Jacobian[i][2] = (d_pertub_state[i][2] -  d_state[i])/pertub # c (no update)
            Jacobian[i][2] = (d_pertub_state[i][2] -  d_state[i])/pertub_s # s

        # inv_Jacobian = np.linalg.inv(Jacobian)
        inv_Jacobian = np.linalg.pinv(Jacobian)
        correction = np.dot(inv_Jacobian, d_state)
        # pdb.set_trace()
        return correction

    def compute_spline(self, initial_state, final_state):
        """
        main function to compute the trajectory by optimizing an spline using
        initial and final state
        
        Total parameter for a spline is
        (kappa_0, a, b, c, s) --> equals (p0, p1, p2, p3, s) in papaer
        
        Consider the constraints by start and goal point kappa, the parameter will be
        (kappa_0, a, b, kappa_f, s) 
        so, only need to estimate 3 parameters (a, b, s) 
        """
        a, b, c, s = self._initialize_spline(initial_state, final_state)
        final_state_pred = self._motion_update_one_shot(initial_state, a, b, c, s)

        converge = self._check_converge(final_state, final_state_pred)
        total_iter = 0
        # pdb.set_trace()
        while (total_iter < self.max_iter) & (converge is not True): # (total_iter < self.max_iter) 
            
            
            correction = self._compute_correction(initial_state, final_state, a, b, c, s)
            a = a - correction[0]
            b = b - correction[1]
            # c = c - correction[2]
            s = s - correction[2]
            
            final_state_pred = self._motion_update_one_shot(initial_state, a, b, c, s)

            converge = self._check_converge(final_state, final_state_pred)
            total_iter = total_iter +1

            # print(total_iter)
            # print(final_state_pred)
            # print(s)

        # sometimes it converge to negative s (travel distance) which 
        # is invalid..., need to figure it out...
        if (converge == True) & (s > 0):
            final_state_pred, point_list = self._path_sampling_one_shot(initial_state, a, b, c, s)
        else:
            point_list = [[-1,-1]]

        return point_list

    def _path_sampling_one_shot(self, initial_state, a, b, c, s):
        """
        run through the spline and sample waypoint in a fix interval

        :param: initial_state: initial state of the vehcle [x, y, theta, kappa]
        :param a: spline parameter
        :param b: spline parameter
        :param c: spline parameter
        :param s: spline parameter
        :return: final_state_pred: predicted final state 
        """
        x_0 = initial_state[0]
        y_0 = initial_state[1]
        theta_0 = initial_state[2] # rad
        kappa_0 = initial_state[3] # rad

        x = np.copy(x_0)
        y = np.copy(y_0)
        theta = np.copy(theta_0)
        kappa = np.copy(kappa_0)

        a_p = kappa_0
        b_p = -(11*kappa_0 - 18*a + 9*b - 2*c)/(2* s)
        c_p = 9*(2*kappa_0 - 5*a + 4*b - c)/(2* s**2)
        d_p = -9*(kappa_0 - 3*a + 3*b -c)/(2* s**3)
        

        resolution = self.sample_resolution

        total_sample = int(s / resolution)

        point_list = []
        point_list.append([x_0, y_0])
        for i in range(1, total_sample +1):
            current_s = float(i) * resolution

            kappa = a_p + b_p*current_s + c_p*(current_s**2) + d_p*(current_s**3)
            theta = self._compute_theta(theta_0, a_p, b_p, c_p, d_p, current_s)
            x = self._compute_x(x_0, theta_0, a_p, b_p, c_p, d_p, current_s)
            y = self._compute_y(y_0, theta_0, a_p, b_p, c_p, d_p, current_s)
            
            point_list.append([x, y])

        final_state_pred = [x, y, theta, kappa]
        return final_state_pred, point_list

def main():
    """
    do trajectory generation demo
    """
    PathGenerator = TrajectoryGenerator()
    
    ## coordinate 
    # Y    
    # ^   /
    # |  /
    # | / <theta>
    # o -- -- -- >X

    x_0 = 0.0 # initial x position
    y_0 = 0.0 # initial y position
    theta_0 = 0.0 *np.pi/180  # initial heading angle of the vehicle 
    kappa_0 = 0.0 *np.pi/180  # initial steering angle  
    initial_state = [x_0, y_0, theta_0, kappa_0] 
    
    x_f = 13.0 # final x position
    y_f = 8.0 # final y position
    theta_f = 0.0 *np.pi/180  # final heading angle of the vehicle 
    kappa_f = 0.0 *np.pi/180  # final steering angle  
    final_state = [x_f, y_f, theta_f, kappa_f] 

    traject = PathGenerator.compute_spline(initial_state, final_state)
    point_array = np.asarray(traject)
    plt.plot(point_array[:,0], point_array[:,1],'o')
    
    sample_resolution = 0.5
    temp_goal_list = []
    for i in range(-2, 3):
        temp_final_state = np.copy(final_state)
        temp_final_state[1] = temp_final_state[1] + float(i)*sample_resolution
        temp_goal_list.append(temp_final_state)
        
    start = time.time()
    point_list = []
    for i in range(0, 5):
        temp_goal = temp_goal_list[i]
        traject = PathGenerator.compute_spline(initial_state, temp_goal)
        point_list.append(traject)
    end = time.time()
    print('Executed time is %f'%(end - start))
    
    # pdb.set_trace()
    for i in range(0,5):
        point_array = np.asarray(point_list[i])
        plt.plot(point_array[:,0], point_array[:,1],'o')
    
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()