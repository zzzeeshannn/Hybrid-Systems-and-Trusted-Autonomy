"""
Satellite Rendezvous Simulation.
Simulation code by Prof. Stanley Bak (@stanleybak)
Set based simulation by Zeeshan Shaikh (@zzzeeshannn)

Date: 10/01/2021
"""


# Importing functions here
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.linalg import expm
from plot import init_plot, plot_box

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

        # Here, we find the maximum value of points in 100 directions that covers 360 degrees from the center
        for angle in np.linspace(0, 2*math.pi, 100):

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

def step(state, mode, sol_mats):
    'use matrix exponential solution to find next state'

    if mode == 'far':
        sol_mat = sol_mats[0]
    elif mode == 'near':
        sol_mat = sol_mats[1]
    elif mode == 'passive':
        sol_mat = sol_mats[2]
        
    new_state = sol_mat.dot(state)

    return new_state

def simulate(start_x, start_y, passive_time):
    '''simulate and plot the satellite rendezvous system
    '''
    
    tmax = 270 # minutes
    step_size = 0.5
    num_steps = int(round(tmax / step_size))
    
    passive_step = int(round(passive_time / step_size))

    # x, y, vx, vy
    cur_state = np.array([[start_x], [start_y], [0], [0]])
    cur_mode = 'far'

    sol_mats = get_solution_mats(step_size)

    xs = [start_x]
    ys = [start_y]

    for cur_step in range(num_steps):
        cur_state = step(cur_state, cur_mode, sol_mats)
        assert cur_state.shape == (4, 1)

        xs.append(cur_state[0])
        ys.append(cur_state[1])
        prev_mode = cur_mode

        # check guards
        if cur_mode != 'passive' and cur_step + 1 == passive_step:
            # next step should be in passive mode
            cur_mode = 'passive'
        elif cur_mode == 'far' and cur_state[0] >= -100:
            cur_mode = 'near'

        if prev_mode != cur_mode or cur_step == num_steps - 1:
            # changed mode or reached end of sim, plot now

            if prev_mode == 'far':
                color = 'lime'
                zorder=0
            elif prev_mode == 'near':
                color = 'orange'
                zorder=1
            elif prev_mode == 'passive':
                color = 'cyan'
                zorder=0

            plt.plot(xs, ys, color, lw=1, alpha=0.5, zorder=zorder)
            xs = [cur_state[0]]
            ys = [cur_state[1]]

def get_solution_mats(step_size):
    'get far, near, and passive solution matrices'

    rv = []

    # far
    a_mat = np.array([[0.0, 0.0, 1.0, 0.0], 
                      [0.0, 0.0, 0.0, 1.0], 
                      [-0.057599765881773, 0.000200959896519766, -2.89995083970656, 0.00877200894463775], 
                      [-0.000174031357370456, -0.0665123984901026, -0.00875351105536225, -2.90300269286856]])

    rv.append(expm(a_mat * step_size))

    # near
    a_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [-0.575999943070835, 0.000262486079431672, -19.2299795908647, 0.00876275931760007],
                      [-0.000262486080737868, -0.575999940191886, -0.00876276068239993, -19.2299765959399]])

    rv.append(expm(a_mat * step_size))

    # passive
    a_mat = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0.0000575894721132000, 0, 0, 0.00876276],
                      [0, 0, -0.00876276, 0]])

    rv.append(expm(a_mat * step_size))

    return rv

def main():
    'main entry point'

    random.seed(0) # deterministic random numbers
    init_box = [(-925.0, -875.0), (-425.0, -375.0)]
    
    init_plot()
    plot_box(init_box, 'k-')

    # simulate 100 times
    for _ in range(100):
        start_x = random.uniform(*init_box[0])
        start_y = random.uniform(*init_box[1])
        passive_time = random.randint(100, 120)

        simulate(start_x, start_y, passive_time)

    # --------------- ZONOTOPE ---------------------
    initial_box = [[-925.0, -875.0], [-425.0, -375.0], [0.0, 0.0], [0.0, 0.0]]

    # Convert the initial box/state to a Zonotope representation
    initial_zonotope = Zonotope(initial_box)

    # Defining the required parameters for the system here
    waiting_list = []
    t_max = 270
    step_size = 0.5
    total_steps = int(round(t_max / step_size))
    # counter is used to calculate the  number of matrix multiplications.
    mode1_counter = 0
    mode2_counter = 0
    mode3_counter = 0

    # Defining the initial mode of the system here
    # The system has 3 different states, namely, far - near - passive
    # Initial state of the system is far
    current_mode = "far"
    # previous_mode = current_mode

    waiting_list.append((initial_zonotope, current_mode, 0))

    while waiting_list:
        # Get the zonotope, mode of the system and current time step
        zonotope, mode, step = waiting_list.pop()

        # Get the solution matrices for all the states of the system
        sol_mats = get_solution_mats(step_size)

        while step < total_steps:
            # Update previous_mode
            previous_mode = mode
            # Increase the step
            step += 1

            # Plot the current zonotope
            if previous_mode == 'far':
                color = 'blue'
                zonotope.plot(color)
            elif previous_mode == 'near':
                color = 'red'
                zonotope.plot(color)
            elif previous_mode == 'passive':
                color = 'black'
                zonotope.plot(color)

            # Advance the zonotope based on the current mode of the system
            if mode == "far":
                zonotope.a_mat = sol_mats[0].dot(zonotope.a_mat)
                zonotope.b_vec = sol_mats[0].dot(zonotope.b_vec)
                mode1_counter += 2
            if mode == "near":
                zonotope.a_mat = sol_mats[1].dot(zonotope.a_mat)
                zonotope.b_vec = sol_mats[1].dot(zonotope.b_vec)
                mode2_counter += 2
            if mode == "passive":
                zonotope.a_mat = sol_mats[2].dot(zonotope.a_mat)
                zonotope.b_vec = sol_mats[2].dot(zonotope.b_vec)
                mode3_counter += 2

            # Mode shift to "near"
            if mode == 'far':
                farthest_point = zonotope.max([1, 1, 0, 0])
                seperation_distance = np.sqrt(farthest_point[0] ** 2 + farthest_point[1] ** 2)
                if seperation_distance <= 100:
                    # print(f"Shifting to near mode at : {step}")
                    mode = "near"

            # Non deterministic shift to passive state
            if mode != "passive" and step >= 100/step_size and step <= 120/step_size:
                waiting_list.append((deepcopy(zonotope), mode, step))
                mode = "passive"

            # Time constraints on modes "far" and "near"
            if mode != "passive":
                new_passive_time = 120
                new_passive_step = int(round(new_passive_time / step_size))
                if step >= new_passive_step:
                    mode = "passive"

    # Printing the number of matrix multiplications.
    print(f"Total number of calculations for mode 1: {mode1_counter}")
    print(f"Total number of calculations for mode 2: {mode2_counter}")
    print(f"Total number of calculations for mode 3: {mode3_counter}")
    print(f"Total number of multiplications: {mode1_counter + mode2_counter + mode3_counter}")

    plt.savefig('reachAnalysis2.png')
    # Zoomed picture
    ax = plt.gca()
    ax.set_xlim([-110, 20])
    ax.set_ylim([-70, 60])
    plt.savefig('zoomed_reachAnalysis.png')

    """
    # save plot
    plt.savefig('sim.png')

    # zoom in and re-plot
    ax = plt.gca()
    ax.set_xlim([-110, 20])
    ax.set_ylim([-70, 60])
    plt.savefig('zoomed_sim.png')
    """

if __name__ == '__main__':
    main()
