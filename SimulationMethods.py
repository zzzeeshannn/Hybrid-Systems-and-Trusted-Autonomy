import numpy as np
import matplotlib.pyplot as plt

"""Defining a function to find the derivate of a state"""

def state_derivative(time, state):
    x, y = state

    x_der = y
    y_der = -x

    return x_der, y_der


def euler_method(init, step_size, steps):  # Starting point, step size, number of steps

    init = np.array(init)
    state_history = [init.copy()]  # Creating a list to hold the states after every derivative
    c_time = 0.0  # Initialize time

    for step in range(steps):
        c_state = state_history[-1]  # Assigning the latest point
        der = state_derivative(c_time, c_state)
        der = np.array(der)
        # print(der)

        # Update Rule for Euler
        new_state = c_state + step_size * der

        # Add the new state to the state history for the next loop
        state_history.append(new_state)
        c_time += step_size

    # Use of time here?
    return state_history


def ab_method(init, step_size, steps):
    initial_point = np.array(init)
    state_history = [initial_point.copy()]
    der_history = []
    c_time = 0.0

    # Using Eulers method to find the second step
    der = state_derivative(c_time, initial_point)
    der = np.array(der)
    der_history.append(der)

    second_point = initial_point + step_size * der
    state_history.append(second_point)
    der_sp = state_derivative(c_time, second_point)
    der_sp = np.array(der_sp)
    der_history.append(der_sp)

    for i in range(2, steps):
        next_point = np.asarray(state_history[i - 1]) + step_size * np.asarray([1.5 * np.asarray(der_history[i - 1]) - 0.5 * np.asarray(der_history[i - 2])])

        state_history.append(next_point[0])
        der_np = state_derivative(c_time, next_point[0])
        der_np = np.array(der_np)
        der_history.append(der_np)

    return state_history

def rk4_method(init, step_size, steps):

    initial_point = np.array(init)
    state_history = [initial_point.copy()]
    c_time = 0.0
    half_step_size = step_size/2

    for step in range(steps):
        k1 = state_derivative(c_time, state_history[-1])
        im_step1 = state_history[-1] + half_step_size * np.asarray(k1)

        k2 = state_derivative(c_time, im_step1)
        im_step2 = state_history[-1] + half_step_size * np.asarray(k2)

        k3 = state_derivative(c_time, im_step2)
        im_step3 = state_history[-1] + step_size * np.asarray(k3)

        k4 = state_derivative(c_time, im_step3)

        next_point = state_history[-1] + 1/6 * step_size * np.asarray((k1 + 2*np.asarray(k2) + 2*np.asarray(k3) + k4))
        state_history.append(next_point)

    return state_history


def plot(state_history, color, label=None):
    xs = [state[0] for state in state_history]
    ys = [state[1] for state in state_history]

    plt.plot(xs, ys, color, label=label)


def main():
    init = [-5.0, 0.0]

    pi = np.pi
    # First step
    s1 = (2 * pi) / 8
    step_size = [s1, s1 / 2, s1 / 4, s1 / 8, s1 / 16, s1 / 32, s1 / 64, s1 / 128, s1 / 256, s1 / 512]
    time_limit = 2 * pi

    # List to store the error per step
    error_euler = []
    error_abmethod = []
    error_rkmethod =[]
    count = 0

    for every_step in step_size:
        count += 1

        steps = int(time_limit / every_step)

        # Euler Method
        e_output = euler_method(init, every_step, steps)

        error_e = np.linalg.norm(e_output[-1] - init)
        error_euler.append(error_e)

        # Adam-Bashford Method
        ab_output = ab_method(init, every_step, steps)

        error_ab = np.linalg.norm(ab_output[-1] - init)
        error_abmethod.append(error_ab)

        #Runge-Kutta 4 Method
        rk_output = rk4_method(init, every_step, steps)

        error_rk = np.linalg.norm(rk_output[-1] - init)
        error_rkmethod.append(error_rk)

    e_ref = [error_euler[0].copy()]
    ab_ref = [error_abmethod[0].copy()]
    rk_ref = [error_rkmethod[0].copy()]

    for i in range(0,9):
        et = float(e_ref[i]/2)
        e_ref.append(et)

        abt = float(ab_ref[i]/4)
        ab_ref.append(abt)

        rkt = float(rk_ref[i])/16
        rk_ref.append(rkt)

    #Plot for Euler Method
    #plt.plot(step_size, error_euler, label = "True Error")
    #plt.plot(step_size, e_ref, label = "Reference Line")
    #plt.title("Step-size v Error Plot for Euler's Method")

    ##Plot for AB Method
    plt.plot(step_size, error_abmethod, label="True Error")
    plt.plot(step_size, ab_ref, label="Reference Line")
    plt.title("Step-size v Error Plot for Adams Bashford Method")

    ##Plot for RK4 Method
    #plt.plot(step_size, error_rkmethod, label="True Error")
    #plt.plot(step_size, rk_ref, label="Reference Line")
    #plt.title("Step-size v Error Plot for RK Method")

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
