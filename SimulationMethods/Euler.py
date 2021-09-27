"""
Author: Zeeshan Shaikh (@zzzeeshannn)
Date: 09/04/2021
Implementing Euler method over a Harmonic Oscillator
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

# Defining euler method here
def euler_simulation(der_function, init_point, step_size, num_steps):
    """
    Logic:
    In the Euler method, next state of the system u(t+h) is given by:
    u(t+h) = u(t) + h * u'(t)
    Note:
    Modified for AB Method

    Inputs:
    der_function: The function to return the derivative of the dynamical system
    num_steps: Number of steps to run the simulation
    delta: Change in time per step
    init_point: The starting point of the dynamical system

    Returns:
    state_list: List of points/state of the dynamic system at the end of the simulation
    """
    # Initialize the state list with the starting point
    state_list = [init_point.copy()]

    for every_step in range(num_steps):
        # Get the current time and state of the system to pass to the der function
        current_time = step_size * every_step
        current_point = state_list[-1]

        # Call der function
        derivative = np.array(der_function(current_point, current_time), dtype=float)

        # Euler update rule
        new_point = current_point + step_size * derivative

        # Add the point to the state list
        state_list.append(new_point)

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

        # Call Euler Method
        state_list = euler_simulation(derivative, init_point, every_step, num_steps)

        # Calculate error for each time step
        error = np.linalg.norm(state_list[-1] - init_point)
        error_list.append(error)

    # Sanity check: The error for first step should be 31.5562626
    print(error_list[0])

    # Calculating the first order reference line
    ref = error_list[0]
    ab_ref = []
    ab_ref.append(ref)
    for i in range(9):
        abt = float(ab_ref[-1]/2)
        ab_ref.append(abt)

    print(ab_ref)

    # Plot
    plt.plot(step_size_list, error_list, "r-o", label="Error at each time step")
    plt.plot(step_size_list, ab_ref, "r-+", label="Reference line")
    plt.title(("log v log plot for Euler's Method"))
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()