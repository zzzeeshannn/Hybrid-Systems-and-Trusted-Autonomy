'''
Euler Simulation vs RK45

This program is used to test the two simulation methods: Euler and RK45.
It is used to compare the mathematical accuracy of the two methods by simulating a harmonic oscillator,
plotting the subsequent points for a fixed time step and comparing the errors.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45

# x' = y, y' = -x
def ha_derivative(t, state):
    'return derivative of harmonic oscillator'

    x, y = state

    der_x = y
    der_y = -x

    return der_x, der_y

def run_euler(init, step_size, num_steps):
    'run euler method for a fixed number of steps, return state_history list'

    init = np.array(init)
    state_history = [init.copy()]
    cur_time = 0.0

    for step in range(num_steps):
        cur_state = state_history[-1]
        cur_der = ha_derivative(cur_time, cur_state)
        cur_der = np.array(cur_der)

        # update rule for euler's method
        next_state = cur_state + step_size * cur_der
        
        state_history.append(next_state)
        cur_time += step_size

    return state_history

def get_dense_output_points(rk45, num_points):
    'get points between time steps using dense_output from rk45'

    points = []
    dense_output_func = rk45.dense_output()

    prev_time = rk45.t_old
    cur_time = rk45.t

    step_size = (cur_time - prev_time) / num_points

    for s in range(num_points):
        t = prev_time + s * step_size
        intermediate_state = dense_output_func(t)

        points.append(intermediate_state)

    return points

def run_rk45(init, t_max):
    'run rk45 method and return state history'

    rk45 = RK45(ha_derivative, 0, init, t_max)

    state_history = [init.copy()]

    # loop until we reach t_max
    while abs(rk45.t - t_max) > 1e-6:
        rk45.step()

        # add 10 points between rk45 steps
        state_history += get_dense_output_points(rk45, 10)

        # add point at the end of the step
        state_history.append(rk45.y)

    return state_history

def plot(state_history, color, label=None):
    'plot xs and ys'

    # plot euler results
    xs = [state[0] for state in state_history]
    ys = [state[1] for state in state_history]

    plt.plot(xs, ys, color, label=label)

def main():
    'main entry point'

    init = [-5.0, 0.0]
    step_size = 0.1
    t_max = 6.0
    num_steps = int(t_max / step_size)

    # run Euler method
    state_history_euler = run_euler(init, step_size, num_steps)
    plot(state_history_euler, 'b-o', label='Euler')
    
    # use RK45 to get "correct" answer
    state_history_rk45 = run_rk45(init, t_max)
    plot(state_history_rk45, 'r-o', label='RK45')

    # save figure to a file
    plt.legend()
    plt.plot()
    plt.savefig('out.png')

    # print difference at last point
    last_euler = state_history_euler[-1]
    last_rk45 = state_history_rk45[-1]

    difference = np.linalg.norm(last_rk45 - last_euler)

    print("difference in last point:", difference)

if __name__ == '__main__':
    main()
