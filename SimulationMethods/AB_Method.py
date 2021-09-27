"""
Author: Zeeshan Shaikh (@zzzeeshannn)
Date: 09/03/2021

Implementing Adams Bashford 2 method over a Harmonic Oscillator
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

# We need the euler simulation for the first state in the AB Method
def euler_simulation(der_function,init_point, step_size, num_steps):
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

    return state_list[-1]

# Define Adam Bashford Method here
def ab2_method(der_function, init_point, step_size, num_steps):
    """
    Logic:
    In the Adam Bashford method, next state of the system u(t+h) is given by:
    u(t+h) = u(t) + (h/2) * [(3 * u'(t)) - u'(t - h)]
    Note:
    Since this logic needs atleast two states, we simulate the first state after the initial state by Euler's Method

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

    # Using Euler Method to find the second point
    new_state = euler_simulation(der_function, init_point, step_size, 1)
    state_list.append(new_state)

    # Run Adam's Bashford method for num_steps - 1 times
    for every_step in range(num_steps - 1):
        # Get the current time and state of the system to pass to the der function
        current_time = step_size * every_step
        current_point = state_list[-1]
        previous_point = state_list[-2]

        # Get the state derivative of the current and previous point
        currentState_der = np.array(derivative(current_point, current_time), dtype=float)
        prevState_der = np.array(derivative(previous_point, current_time-step_size), dtype=float)

        # Get the new point
        next_state = current_point + (step_size/2) * ((3 * currentState_der) - prevState_der)

        # Save in the list of states
        state_list.append(next_state)

    print("-----------------")
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

        # Call AB Method
        state_list = ab2_method(derivative, init_point, every_step, num_steps)

        # Calculate error for each time step
        error = np.linalg.norm(state_list[-1] - init_point)
        error_list.append(error)

    # Sanity check: The error for first step should be 24.6254028
    print(error_list)

    # Calculate the reference line
    ref = error_list[0]
    ab_ref = []
    ab_ref.append(ref)
    for i in range(9):
        abt = float(ab_ref[-1]/4)
        ab_ref.append(abt)

    # Plot
    plt.plot(error_list, step_size_list, "b-o", label="Error at each time step")
    plt.plot(ab_ref, step_size_list, "b-+", label="Reference line")
    plt.title("log v log plot for AB Method")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()