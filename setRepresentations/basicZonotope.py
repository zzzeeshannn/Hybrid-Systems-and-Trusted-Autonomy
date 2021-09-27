"""
Zonotope representation that starts with an initial set of states (box form)
"""

# Importing functions here
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from copy import deepcopy
from scipy.linalg import expm

# Defining the Zonotope here
class Zonotope:
    # In this representation, we take a box as input as compared to the traditional center and generators
    # The box represents the set of all possible states
    def __init__(self, box, a_mat = None):
        self.box = np.array(box, dtype = float)
        self.a_mat = a_mat if a_mat is not None else np.identity(self.box.shape[0])

        self.dimensions = self.a_mat.shape[0]
        self.generators = self.a_mat.shape[1]

        self.b_vec = np.zeros((self.dimensions, 1))

    def max(self, direction):
        # This function returns the value of the point that is maximum in this set for the given direction
        direction = self.a_mat.transpose().dot(direction)

        box = self.box
        rv = []

        for dim, (lb, ub) in enumerate(box):
            if direction[dim] > 0:
                rv.append([ub])
            else:
                rv.append([lb])

        point = np.array(rv)

        return self.a_mat.dot(point) + self.b_vec

    def verts(self, xdim = 0, ydim = 1):
        # In this method, we try to approximate the zonotope in two dimensions
        # In order to do that, we find the points with maximum value in directions from the center --
        # -- such that it is good enough for an approximation
        vertices = []

        # Here, we find the maximum value of points in 32 directions that covers 360 degrees from the center
        for angle in np.linspace(0, 2*math.pi, 32):

            direction = np.zeros((self.dimensions, ))
            direction[xdim] = math.cos(angle)
            direction[ydim] = math.sin(angle)

            point = self.max(direction)
            final_point = (point[xdim][0], point[ydim][0])

            if vertices and np.allclose(final_point, vertices[-1]):
                continue

            vertices.append(final_point)

        return vertices

    def plot(self, color='k-o', xdim = 0, ydim = 1):
        # Function to plot the zonotope in 2D

        vertices_list = self.verts(xdim=xdim, ydim=ydim)

        xs = [v[xdim] for v in vertices_list]
        xs.append(vertices_list[0][xdim])

        ys = [v[ydim] for v in vertices_list]
        ys.append(vertices_list[0][ydim])

        plt.plot(xs, ys, color)


# This function is used to set the plotting style provided Prof. Stanley Bak
def init_plot():
    try:
        matplotlib.use('TkAgg')
    except:
        pass

    plt.style.use(['bmh', 'bak_matplotlib.mlpstyle'])
    plt.axis('equal')

# This function is used to check if the zonotope is inside the invariant
def is_inside_invariant(zono, mode, boundary):
    rv = True

    if mode == 1:
        min_x_point = zono.max([-1.0, 0.0])
        min_x_val = min_x_point[0]
        # Acceptable error
        tol = 1e-6
        if min_x_val + tol >= boundary:
            rv = False

    return rv

# Defining the main function here
def main():
    # Define the initial box here that represents the set of possible states
    initial_box = [[-5.0, -4.0], [0.0, 1.0]]

    # Define the equations here that govern the propagation of this object in the form of matrix
    # We define two modes here
    # Mode 1: x' = x and y' = 2y
    mode1_dynamics = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype = float)
    # Mode 2: x' = 2x and y' = y
    mode2_dynamics = 2.0

    # Define the mode boundary here which triggers the change from one mode to another
    mode_boundary = 0.0

    # Defining the time step and number of steps here
    num_steps = 10
    time_step = math.pi / 8

    # Define the solution matrix here i.e e^(A*t) where A is your current dynamics matrix
    # This allows us to find the set of states at any time step, z(t) = e^(A*t) * z(0)
    solution_matrix = expm(mode1_dynamics * time_step)

    # Convert the initial box to Zonotope representation
    initial_zonotope = Zonotope(initial_box)

    # Create a list here to hold the zonotope corresponding to the time step and mode
    waiting_list = []
    # Tuple = (Zonotope, Mode, Time Step)
    waiting_list.append((initial_zonotope, 1, 0))

    while waiting_list:
        temp_zonotope, mode, step = waiting_list.pop()

        while step < num_steps and is_inside_invariant(temp_zonotope, mode, mode_boundary):
            # Plot the current zonotope
            if mode == 1:
                temp_zonotope.plot('r')
            else:
                temp_zonotope.plot('b')
            # Increment the time step
            step += 1

            # Advance the zonotope
            if mode == 1:
                temp_zonotope.a_mat = solution_matrix.dot(temp_zonotope.a_mat)
                temp_zonotope.b_vec = solution_matrix.dot(temp_zonotope.b_vec)
            else:
                temp_zonotope.b_vec += np.array([[0], [mode2_dynamics]])

            if mode == 1:
                max_x_point = temp_zonotope.max([1.0, 0.0])
                max_x_val = max_x_point[0]

                if max_x_val >= mode_boundary:
                    waiting_list.append((deepcopy(temp_zonotope), 2, step))

    plt.plot([mode_boundary, mode_boundary], [-2, 6], 'k--')
    plt.show()
    #plt.savefig('basicZono.png')

if __name__ == '__main__':
    init_plot()
    main()
