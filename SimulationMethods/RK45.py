"""
Author: Zeeshan Shaikh (@zzzeeshannn)
Date: 09/04/2021

Implementing Runge Kutta 4 method over a Harmonic Oscillator
"""

# Importing files here
import numpy as np
import matplotlib.pyplot as plt

# Define the Harmonic Oscillator derivative here
def derivative(state, time = None):
    """
    Input parameters:
    state: current x and y
    time: current time step

    Returns:
    x and y derivative for Harmonic Oscillator
    """
    x, y = state

    x_der = y
    y_der = -x

    return x_der, y_der

# Define Runge Kutta Method here
def rk4_method(der_function, init_point, step_size, num_steps):
    """
    Logic:
    In the Runge Kutta method, next state of the system u(t+h) is given by:
    u(t+h) = u(t) + (h/6) * [k1 + 2*k2 + 2*k3 + k4]
    where,
    k1 = u'(t)
    k2 = u'(t + h/2 * k1)
    k3 = u'(t + h/2 * k2)
    k4 = u'(t + h * k3)

    Input parameters:
    der_function: Derivative function describing the system
    init_point: Starting point for the simulation
    step_size
    num_steps

    :return:
    state_list: The list of states of the system at the end of the simulation
    """
    # Define required parameters here
    state_list = [init_point.copy()]

    # Run Runge Kutta method for num_steps times
    for every_step in range(num_steps):
        # Get the current time and state of the system to pass to the der function
        current_time = step_size * every_step
        current_point = state_list[-1]

        # Calculate K1
        k1 = np.array(der_function(current_point, current_time))
        # Calculate K2
        k2 = np.array(der_function((current_point + (step_size/2) * k1), current_time))
        # Calculate K3
        k3 = np.array(der_function((current_point + (step_size / 2) * k2), current_time))
        # Calculate K4
        k4 = np.array(der_function((current_point + step_size * k3), current_time))
        # Calculate the next point
        next_point = current_point + (step_size/6) * (k1 + 2*k2 + 2*k3 + k4)

        # Save the next point in the list
        state_list.append(next_point)

    return state_list

def main():
    # Define required parameters here
    pi = np.pi
    init_point = np.array([-5.0, 0.0])
    step_size = pi/4
    max_time = 2 * pi
    error_list = []

    # List of step sizes to test and calculate the error
    step_size_list = []
    step_size_list.append(step_size)
    for _ in range(0, 9):
        step_size_list.append(step_size_list[-1]/2)

    for every_step in step_size_list:
        num_steps = int(round(max_time / every_step))

        # Call RK Method
        state_list = rk4_method(derivative, init_point, every_step, num_steps)

        # Calculate error for each time step
        error = np.linalg.norm(state_list[-1] - init_point)
        error_list.append(error)

    # Sanity check: The error for first step should be 0.098487
    print(error_list[0])

    # Calculating the fourth order reference line
    ref = error_list[0]
    ab_ref = []
    ab_ref.append(ref)
    for i in range(9):
        abt = float(ab_ref[-1]/16)
        ab_ref.append(abt)

    # Plot
    plt.plot(error_list, step_size_list, "g-o", label="Error at each time step")
    plt.plot(ab_ref, step_size_list, "g-+", label="Reference line")
    plt.title(("log v log plot for RK4 Method"))
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()