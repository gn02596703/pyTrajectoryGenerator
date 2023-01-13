"""
The algorithm generate a smooth trajectory (a list of waypoint) according to 
given initial state and goal state.

The algorithm is mainly based on following publications
"Trajectory Generation For Car-Like Robots Using Cubic Curvature Polynomials",
Nagy and Kelly, 2001 
"Adaptive Model Predictive Motion Planning for Navigation in Complex Environments",
Thomas M. Howard, 2009, (chapter 4)
"Parallel Algorithms for Real-time Motion Planning",
Matthew McNaughton, 2011
"Motion Planning for Self-Driving-Cars", Coursera 4 of Self-Driving-Car Specialization
in Coursera

The implementation also references the idea from Matthew O'Kelly's C++ implementation
in Autoware Repository.
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
        self.max_iter = 100

        # criteria of terminal state
        self.acceptable_dx = 0.01 # m
        self.acceptable_dy = 0.01 # m
        self.acceptable_dtheta = 1 *np.pi/180 # rad
        self.acceptable_dkappa = 1 *np.pi/180 # rad

        # pertubation value
        self.pertub = 0.0001

    def _initialize_spline(self, initial_state, final_state):
        """
        The function initializes spline paramters

        :param: final_state: goal of the path [x, y, theta, kappa]
        :return: p0, p1, p2, p3, s: parameter of the spline
        """
        x_f = final_state[0]
        y_f = final_state[1]
        theta_f = final_state[2] # rad
        kappa_f = final_state[3] # rad
        kappa_0 = initial_state[3] # rad

        # Initilalization the spline parameter
        # from
        # Parallel Algorithms for Real-time Motion Planning , Matthew McNaughton, 2011
        p0 = kappa_0  # constrainted by initiali state
        p1 = 0.0      # parameter to be optimized
        p2 = 0.0      # parameter to be optimized
        p3 = kappa_f  # constrainted by final state
        s = x_f       # parameter to be optimized

        return p0, p1, p2, p3, s

    def _motion_update(self, initial_state, p0, p1, p2, p3, s):
        """
        predict the final state according to
            - initial vehicle state
            - spline parameter
            - vehicle motion model

        :param initial_state: initial state of the vehcle [x, y, theta, kappa]
        :param p0: spline parameter
        :param p1: spline parameter
        :param p2: spline parameter
        :param p3: spline parameter
        :param s: spline parameter
        :return final_state_pred: predicted final state
        """
        x_0 = initial_state[0]
        y_0 = initial_state[1]
        theta_0 = initial_state[2] # rad
        kappa_0 = initial_state[3] # rad

        veh_speed = 3.6 # m/s
        sample_time = 0.001 # second
        step_time = 0.001 # second

        total_travel_distance = 0.0

        # create spline parameter
        a_p = p0
        b_p = -(11*p0 - 18*p1 + 9*p2 - 2*p3)/(2* s)
        c_p = 9*(2*p0 - 5*p1 + 4*p2 - p3)/(2* s**2)
        d_p = -9*(p0 - 3*p1 + 3*p2 -p3)/(2* s**3)

        ## start predicting vehicle state using motion model and spline
        # check if the vehicle is at goal yet or not
        while total_travel_distance < s:
            # calculate step time and moving distance of this prediction
            remaining_distance = s - total_travel_distance
            step_time = remaining_distance/veh_speed
            step_time = min(step_time, sample_time)

            # predictive vehicle state by motion model
            x_0 = x_0 + veh_speed * step_time * np.cos(theta_0)
            y_0 = y_0 + veh_speed * step_time * np.sin(theta_0)
            theta_0 = theta_0 + veh_speed * step_time * kappa_0

            total_travel_distance = total_travel_distance + veh_speed * step_time

            # update kappa according to spline
            kappa_0 = a_p \
                        + b_p*total_travel_distance \
                        + c_p*(total_travel_distance**2) \
                        + d_p*(total_travel_distance**3)

            # update velocity
            veh_speed = 3.6

        final_state_pred = [x_0, y_0, theta_0, kappa_0]
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

    def _compute_correction(self, initial_state, final_state, p0, p1, p2, p3, s):
        """
        compute the correction of spline parameter
        """
        pertub = self.pertub
        pertub_s = pertub *10

        pred_no_pertub = self._motion_update(initial_state, p0, p1, p2, p3, s)
        pred_pertub_p1 = self._motion_update(initial_state, p0, p1 +pertub, p2, p3, s)
        pred_pertub_p2 = self._motion_update(initial_state, p0, p1, p2 +pertub, p3, s)
        pred_pertub_s = self._motion_update(initial_state, p0, p1, p2, p3, s +pertub_s)

        d_state = np.zeros((3,1))
        d_pertub_state = np.zeros((3,3))
        Jacobian = np.zeros((3,3))
        for i in range(0, 3):
            d_pertub_state[i][0] = (final_state[i] - pred_pertub_p1[i]) # a
            d_pertub_state[i][1] = (final_state[i] - pred_pertub_p2[i]) # b
            d_pertub_state[i][2] = (final_state[i] - pred_pertub_s[i]) # s

            d_state[i] = final_state[i] - pred_no_pertub[i]

            Jacobian[i][0] = (pred_pertub_p1[i] -  pred_no_pertub[i])/pertub # p1
            Jacobian[i][1] = (pred_pertub_p2[i] -  pred_no_pertub[i])/pertub # p2
            Jacobian[i][2] = (pred_pertub_s[i] -  pred_no_pertub[i])/pertub_s # s

        inv_Jacobian = np.linalg.pinv(Jacobian)
        correction = np.dot(inv_Jacobian, d_state)
        # pdb.set_trace()
        return correction

    def compute_spline(self, initial_state, final_state):
        """
        main function to compute the trajectory by optimizing an spline using
        initial and final state

        Total parameter for a spline is (p0, p1, p2, p3, s)

        Consider the constraints by start and goal point, the parameter will be
        (kappa_0, p1, p2, kappa_f, s), we only need to optimize 3 parameters (p1, p2, s) 
        """
        p0, p1, p2, p3, s = self._initialize_spline(initial_state, final_state)
        final_state_pred = self._motion_update(initial_state, p0, p1, p2, p3, s)

        converge = self._check_converge(final_state, final_state_pred)
        total_iter = 0

        while (total_iter < self.max_iter) & (converge is not True):

            correction = self._compute_correction(initial_state, final_state, p0, p1, p2, p3, s)
            p1 = p1 + correction[0]
            p2 = p2 + correction[1]
            s = s + correction[2]

            final_state_pred = self._motion_update(initial_state, p0, p1, p2, p3, s)

            converge = self._check_converge(final_state, final_state_pred)
            total_iter = total_iter +1
            print('total iter is %f'%(total_iter))

        # sometimes it converge to negative s (travel distance) which
        # is invalid..., need to figure it out...
        if (converge is True) & (s > 0):
            final_state_pred, point_list = self._path_sampling(initial_state, p0, p1, p2, p3, s)
        else:
            point_list = [[-1,-1]]

        return point_list

    def _path_sampling(self, initial_state, p0, p1, p2, p3, s):
        """
        run through the spline and sample waypoint in a fix interval

        :param: initial_state: initial state of the vehcle [x, y, theta, kappa]
        :param p0: spline parameter
        :param p1: spline parameter
        :param p2: spline parameter
        :param p3: spline parameter
        :param s: spline parameter
        :return: final_state_pred: predicted final state 
        """
        x_0 = initial_state[0]
        y_0 = initial_state[1]
        theta_0 = initial_state[2] # rad
        kappa_0 = initial_state[3] # rad

        veh_speed = 3.6 # m/s
        sample_time = 0.1 # second
        step_time = 0.01 # second

        total_travel_distance = 0.0

        # create spline parameter
        a_p = p0
        b_p = -(11*p0 - 18*p1 + 9*p2 - 2*p3)/(2* s)
        c_p = 9*(2*p0 - 5*p1 + 4*p2 - p3)/(2* s**2)
        d_p = -9*(p0 - 3*p1 + 3*p2 -p3)/(2* s**3)

        # create point list of the path
        point_list = []
        point_list.append([x_0, y_0])

        ## start predicting vehicle state using motion model and spline
        # check if the vehicle is at goal yet or not
        while total_travel_distance < s:
            # calculate step time and moving distance of this prediction
            remaining_distance = s - total_travel_distance
            step_time = remaining_distance/veh_speed
            step_time = min(step_time, sample_time)

            # predictive vehicle state by motion model
            x_0 = x_0 + veh_speed * step_time * np.cos(theta_0)
            y_0 = y_0 + veh_speed * step_time * np.sin(theta_0)
            theta_0 = theta_0 + veh_speed * step_time * kappa_0

            total_travel_distance = total_travel_distance + veh_speed * step_time

            # update kappa according to spline
            kappa_0 = a_p \
                        + b_p*total_travel_distance \
                        + c_p*(total_travel_distance**2) \
                        + d_p*(total_travel_distance**3)

            # update velocity
            veh_speed = 3.6

            point_list.append([x_0, y_0])

        final_state_pred = [x_0, y_0, theta_0, kappa_0]
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

    x_f = 10.0 # final x position
    y_f = 5.0 # final y position
    theta_f = 0.0 *np.pi/180  # final heading angle of the vehicle
    kappa_f = 0.0 *np.pi/180  # final steering angle
    final_state = [x_f, y_f, theta_f, kappa_f]

    planned_path = PathGenerator.compute_spline(initial_state, final_state)
    point_array = np.asarray(planned_path)


    plt.plot(point_array[:,0], point_array[:,1],'o')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()