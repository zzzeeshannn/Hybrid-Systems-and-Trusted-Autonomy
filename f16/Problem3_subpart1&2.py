"""
Date: 09/11/2021

Hybrid Systems: F16 Aircraft Simulation

This program implements the F16 Aircraft simulation using the Euler and RK4 method defined under:
run_rk4_classic_f16_sim() and run_euler_f16_sim() functions.
It also implements the bisection method as a zero crossing method to change between waypoints, defined under:
besection() function

"""

import time

import numpy as np
import matplotlib.pyplot as plt

import aerobench.plot as plot
from aerobench.waypoint_autopilot import WaypointAutopilot
from aerobench.util import StateIndex


def main():
    'main function'

    ### Initial Conditions ###
    power = 9  # engine power level (0-10)

    # Default alpha & beta
    alpha = np.deg2rad(2.1215)  # Trim Angle of Attack (rad)
    beta = 0  # Side slip angle (rad)

    # Initial Attitude
    alt = 1500  # altitude (ft)
    vt = 540  # initial velocity (ft/sec)
    phi = 0  # Roll angle from wings level (rad)
    theta = 0  # Pitch angle from nose level (rad)
    psi = 0  # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # add three states for the low-level controller integrators
    init += [0, 0, 0]

    tmax = 150  # simulation time

    # list of waypoints
    waypoints = [[-5000.0, -7500.0, alt],
                 [-15000.0, -7500.0, alt],
                 [-20000.0, 0.0, alt + 500.0],
                 [0.0, 15000.0, alt]]

    step_size = 0.05

    # Euler Simulation here
    start_time = time.perf_counter()
    states = run_euler_f16_sim(init, tmax, step_size, waypoints)
    runtime = time.perf_counter() - start_time
    print(f"Simulation Completed in {round(runtime, 2)} seconds")

    # RK4 simulation here
    start_time = time.perf_counter()
    states_rk = run_rk4_classic_f16_sim(init, tmax, step_size, waypoints)
    runtime = time.perf_counter() - start_time
    print(f"Simulation Completed in {round(runtime, 2)} seconds")

    # print final state for Euler
    final_state = states[-1]
    x = final_state[StateIndex.POS_E]
    y = final_state[StateIndex.POS_N]
    z = final_state[StateIndex.ALT]
    print(f"Euler with step size: {step_size}, final state x, y, z: {round(x, 3)}, {round(y, 3)}, {round(z, 3)}")

    # print final state for RK4
    final_state = states_rk[-1]
    x = final_state[StateIndex.POS_E]
    y = final_state[StateIndex.POS_N]
    z = final_state[StateIndex.ALT]
    print(f"RK with step size: {step_size}, final state x, y, z: {round(x, 3)}, {round(y, 3)}, {round(z, 3)}")

    # plot for Euler
    plot.plot_overhead(states, waypoints=waypoints)
    plt.show()
    filename = 'overhead_euler.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot for RK4
    plot.plot_overhead(states_rk, waypoints=waypoints)
    plt.show()
    filename = 'overhead_rk4.png'
    plt.savefig(filename)
    print(f"Made {filename}")


def bisection(previous_time, current_time, previous_state, current_state, waypoint_distance, waypoint_tolerance,
              current_waypoint, autopilot, method = 1):
    """
    Aim:
    To implement bisection method such that the aircraft changes it's waypoint to the next waypoint as soon as it is within the range of
    [200 - 1e-6, 200] ft of the current waypoint

    Logic:
    Bisection method takes in two states of the system:
    State 1: The previous time step and the positional state of the system
    State 2: The current time step and the positional state of the system
    It then calculates the time and state of the system within the said tolerance limits

    :param previous_time: Time step before the system transitioned to a position < 200 ft of the current waypoint
    :param current_time: Time step at which the system is first within 200 ft of the current waypoint
    :param previous_state: The state of the system at "previous_time"
    :param current_state: The current state of the system
    :param waypoint_tolerance: The accepted tolerance in waypoint distance (in ft)
    :param current_waypoint: The waypoint the aircraft is currently headed towards
    :param autopilot: The autopilot system on the F16 aircraft responsible for it's motion, used for it's "der_func" here which defines the ODE that governs its movement

    :return: The time and state of the system when aircraft is within the waypoint tolerance and the waypoint distance at which it shifts.
    """
    flag = 1

    while flag != 0:
        intermediate_time = (previous_time + current_time) / 2
        step_size = intermediate_time - previous_time

        if method == 1:
            # For Runge Kutta method
            # Calculate K1
            k1 = autopilot.der_func(intermediate_time, previous_state)
            # Calculate K2
            k2 = autopilot.der_func(intermediate_time, (previous_state + (step_size / 2) * k1))
            # Calculate K3
            k3 = autopilot.der_func(intermediate_time, (previous_state + (step_size / 2) * k2))
            # Calculate K4
            k4 = autopilot.der_func(intermediate_time, (previous_state + step_size * k3))
            # Calculate the next point
            intermediate_state = previous_state + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            # For Euler's method
            intermediate_state = previous_state + step_size * autopilot.der_func(intermediate_time, previous_state)

        # Retrieve the position of the aircraft at intermediate step
        x = intermediate_state[StateIndex.POS_E]
        y = intermediate_state[StateIndex.POS_N]
        z = intermediate_state[StateIndex.ALT]
        intermediate_position = np.array([x, y, z])

        distance_to_waypoint = np.linalg.norm(current_waypoint - intermediate_position)

        if (distance_to_waypoint - waypoint_distance < 0 and distance_to_waypoint - waypoint_distance > - 1e-6):
            flag = 0
        elif (distance_to_waypoint < 200):
            current_time = intermediate_time
            current_state = intermediate_state
        else:
            previous_time = intermediate_time
            previous_state = intermediate_state

    return intermediate_time, intermediate_state, distance_to_waypoint


# Defining the RK4 method here
def run_rk4_classic_f16_sim(init, tmax, step_size, waypoints):
    """
    This function returns the list of states after running the simulation for the F16 aircraft using Runge-Kutta method

    Logic:
    In the Runge Kutta method, next state of the system u(t+h) is given by:
    u(t+h) = u(t) + (h/6) * [k1 + 2*k2 + 2*k3 + k4]
    where,
    k1 = u'(t)
    k2 = u'(t + h/2 * k1)
    k3 = u'(t + h/2 * k2)
    k4 = u'(t + h * k3)

    :param init: Start point of the aircraft [x, y, z]
    :param tmax: Max running time for the system
    :param step_size: Delta in time steps
    :param waypoints: List of waypoints that the aircraft needs to follow

    :return: "states" - List of states of the aircraft system from time, t = 0 to tmax
    """

    # Define required parameters here
    init_point = np.array(init)
    state_list = [init_point.copy()]
    current_time = 0
    current_waypoint_index = 0
    autopilot = WaypointAutopilot(waypoints[current_waypoint_index])
    waypoint_distance = 200
    waypoint_tolerance = 1e-6

    # Run Runge Kutta method for num_steps times
    while current_time + 1e-6 < tmax:
        # Get the current state of the system to pass to the der function
        current_point = state_list[-1]
        # Update the time of the system
        current_time = current_time + step_size

        # Calculate K1
        k1 = autopilot.der_func(current_time, current_point)
        # Calculate K2
        k2 = autopilot.der_func(current_time, (current_point + (step_size / 2) * k1))
        # Calculate K3
        k3 = autopilot.der_func(current_time, (current_point + (step_size / 2) * k2))
        # Calculate K4
        k4 = autopilot.der_func(current_time, (current_point + step_size * k3))
        # Calculate the next point
        next_point = current_point + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Get the position [x, y, z] of the current waypoint
        current_waypoint = np.array(waypoints[current_waypoint_index])

        # Retrieve the position of the aircraft
        x = next_point[StateIndex.POS_E]
        y = next_point[StateIndex.POS_N]
        z = next_point[StateIndex.ALT]
        current_position = np.array([x, y, z])

        # Print the distance
        distance = np.linalg.norm(current_waypoint - current_position)

        # Zero crossing
        if (distance < waypoint_distance):
            # Retrieve the previous temporal state of the system
            previous_time = current_time - step_size
            # Call the bisection method
            previous_state = current_point
            current_state = next_point

            current_time, new_point, new_distance = bisection(previous_time, current_time, previous_state,
                                                              current_state, waypoint_distance, waypoint_tolerance,
                                                              current_waypoint, autopilot)

            print(f"For RK4 simulation, Waypoint changed at:- Time: {current_time} and distance: {new_distance}")

            current_waypoint_index += 1
            autopilot = WaypointAutopilot(waypoints[current_waypoint_index])
            state_list.append(new_point)
        else:
            state_list.append(next_point)

        # print(f"RK4 Simulation:- Time {round(current_time, 6)}, distance to waypoint: {round(distance, 3)} ft")

    return state_list


def run_euler_f16_sim(init, tmax, step_size, waypoints):
    'run the simulation and return a list of states'

    autopilot = WaypointAutopilot(waypoints[0])

    cur_state = np.array(init)
    states = [cur_state.copy()]  # state history

    cur_time = 0
    cur_waypoint_index = 0

    # waypoint distance and tolerance parameters
    wp_dist = 200
    wp_tol = 1e-6

    while cur_time + 1e-6 < tmax:  # while time != tmax
        # update state
        cur_state = states[-1] + step_size * autopilot.der_func(cur_time, cur_state)
        cur_time = cur_time + step_size

        # print distance to waypoint
        cur_waypoint = np.array(waypoints[cur_waypoint_index])

        x = cur_state[StateIndex.POS_E]
        y = cur_state[StateIndex.POS_N]
        z = cur_state[StateIndex.ALT]

        cur_pos = np.array([x, y, z])
        distance = np.linalg.norm(cur_waypoint - cur_pos)

        if (distance < wp_dist):
            # Retrieve the previous temporal state of the system
            previous_time = cur_time - step_size
            # Call the bisection method
            previous_state = states[-1]
            current_state = cur_state

            cur_time, cur_state, new_distance = bisection(previous_time, cur_time, previous_state, current_state,
                                                          wp_dist, wp_tol, cur_waypoint, autopilot, method=2)
            print(f"For Euler simulation, Waypoint changed at:- Time: {cur_time} and distance: {new_distance}")
            cur_waypoint_index += 1
            autopilot = WaypointAutopilot(waypoints[cur_waypoint_index])
            states.append(cur_state)
        else:
            states.append(cur_state)

    return states


if __name__ == '__main__':
    main()
