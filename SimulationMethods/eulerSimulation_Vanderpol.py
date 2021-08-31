"""
This program is the basic implementation of the Euler Simulation over a Vanderpol Oscillator.

Dynamics of a Vanderpol Oscillator are:
x' = y
y' = ((1-x^2)*y) - x
"""

# Import libraries here
import numpy as np
import matplotlib.pyplot as plt

def vanderpol_oscillator(state, time):
    """
    Inputs:
    state: Current x and y position
    time: Current time step

    Returns:
    x_der, y_der: First order derivatives for the Harmonic Oscillator

    Note:
    The input parameter "time" is not used in this implementation but is often used and is a standard input parameter.
    """
    x, y = state
    # Derivative here
    x_der = y
    y_der = ((1 - (x**2))*y) - x

    return x_der, y_der

def euler_simulation(der_function, num_steps, delta, init_point):
    """
    This function is used to run the euler simulation

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
        current_time = delta * every_step
        current_point = state_list[-1]
        # Call der function
        derivative = np.array(der_function(current_point, current_time), dtype=float)
        # Euler update rule
        new_point = current_point + delta * derivative
        # Add the point to the state list
        state_list.append(new_point)

    return state_list

def main():
    """
    Main function
    Required parameters:
    init_point: Starting point of the harmonic oscillator
    delta: The time delta/ time elapsed per step
    max_time: Time limit
    num_steps: Essentially max_time/time_step (helpful for runnning loops)
    """
    color_list = ["black", "r", "b"]
    for index, delta in enumerate([0.1, 0.01, 0.001]):
        # Define required parameters
        # Convert the list to a vector
        # This step is essential as it helps with vector addition
        init_point = np.array([-1.0, 0.0])
        max_time = 10.0
        # Need to convert it to int as float dtype cannot be used in range queries
        num_steps = int(round(max_time / delta))
        # Call euler simulation
        points = euler_simulation(vanderpol_oscillator, num_steps, delta, init_point)

        # Plot results
        # Retrieve the x and y points
        xs = [every_point[0] for every_point in points]
        ys = [every_point[1] for every_point in points]
        plt.subplot(1, 3, index+1)
        plt.plot(xs, ys, color_list[index])
        plt.title(delta)

    plt.show()

if __name__ == "__main__":
    main()