'''
Hybrid Systems homework 1

F-16 simulation
'''

import time

import numpy as np
import matplotlib.pyplot as plt

import aerobench.plot as plot
from aerobench.waypoint_autopilot import WaypointAutopilot
from aerobench.util import StateIndex

def main():
    'main function'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = np.deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1500        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = 0           # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # add three states for the low-level controller integrators
    init += [0, 0, 0]
    
    tmax = 150 # simulation time

    # list of waypoints
    waypoints = [[-5000.0, -7500.0, alt],
                 [-15000.0, -7500.0, alt],
                 [-20000.0, 0.0, alt+500.0],
                 [0.0, 15000.0, alt]]

    step_size = 0.05

    start_time = time.perf_counter()
    states = run_euler_f16_sim(init, tmax, step_size, waypoints)
    runtime = time.perf_counter() - start_time
    print(f"Simulation Completed in {round(runtime, 2)} seconds")

    # print final state
    final_state = states[-1]
    x = final_state[StateIndex.POS_E]
    y = final_state[StateIndex.POS_N]
    z = final_state[StateIndex.ALT]
    print(f"Euler with step size: {step_size}, final state x, y, z: {round(x, 3)}, {round(y, 3)}, {round(z, 3)}")

    # plot
    plot.plot_overhead(states, waypoints=waypoints)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")

def run_euler_f16_sim(init, tmax, step_size, waypoints):
    'run the simulation and return a list of states'

    autopilot = WaypointAutopilot(waypoints[0])

    cur_state = np.array(init)
    states = [cur_state.copy()] # state history

    cur_time = 0
    cur_waypoint_index = 0

    # waypoint distance and tolerance paramters
    wp_dist = 200
    wp_tol = 1e-6

    while cur_time + 1e-6 < tmax: # while time != tmax
        # update state
        cur_state = states[-1] + step_size * autopilot.der_func(cur_time, cur_state)

        # print distance to waypoint
        cur_waypoint = np.array(waypoints[cur_waypoint_index])

        x = cur_state[StateIndex.POS_E]
        y = cur_state[StateIndex.POS_N]
        z = cur_state[StateIndex.ALT]

        cur_pos = np.array([x, y, z])
        distance = np.linalg.norm(cur_waypoint - cur_pos)

        if(distance < 200):
            b = cur_time
            a = cur_time - step_size
            temp = states[-1]

            cur_state = midpoint_method(a, b, wp_tol, states, step_size, temp, autopilot.der_func, waypoints, cur_waypoint_index)

            cur_waypoint_index += 1
            autopilot = WaypointAutopilot(waypoints[cur_waypoint_index])

        states.append(cur_state)
        cur_time = cur_time + step_size

        print(f"Time {round(cur_time, 6)}, distance to waypoint: {round(distance, 3)} ft")

    return states

def midpoint_method(a, b, wp_tol, states, step_size, cur_state, func, waypoints, cur_waypoint_index):
    counter = 1
    mid = (a+b)/2
    c = cur_state

    # print distance to waypoint
    cur_waypoint = np.array(waypoints[cur_waypoint_index])

    while counter == 1:
        new_state = states[-1] + step_size * func(mid, c)

        x = new_state[StateIndex.POS_E]
        y = new_state[StateIndex.POS_N]
        z = new_state[StateIndex.ALT]

        cur_pos = np.array([x, y, z])
        distance = np.linalg.norm(cur_waypoint - cur_pos)

        if (distance > 200):
            a = mid
        elif (distance < 200):
            b = mid

        c = new_state
        if(distance > 200 - wp_tol or distance < 200 + wp_tol):
            counter = 0;

    return new_state

if __name__ == '__main__':
    main()
