'''
Hybrid Systems homework 1

F-16 simulation
'''

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45

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
    final_point = []
    difference = 2
    counter = 0

    while (difference > 1):
        start_time = time.perf_counter()
        states = run_euler_f16_sim(init, tmax, step_size, waypoints)
        runtime = time.perf_counter() - start_time
        print(f"Simulation Completed in {round(runtime, 2)} seconds")

        # print final state
        final_state = states[-1]
        final_point.append(final_state)
        x = final_state[StateIndex.POS_E]
        y = final_state[StateIndex.POS_N]
        z = final_state[StateIndex.ALT]
        print(f"Euler with step size: {step_size}, final state x, y, z: {round(x, 3)}, {round(y, 3)}, {round(z, 3)}")

        if (counter > 0):
            c_point = final_point[-1]
            p_point = final_point[-2]

            x1 = c_point[StateIndex.POS_E]
            y1 = c_point[StateIndex.POS_N]
            z1 = c_point[StateIndex.ALT]

            cur_pos1 = np.array([x1, y1, z1])

            x2 = p_point[StateIndex.POS_E]
            y2 = p_point[StateIndex.POS_N]
            z2 = p_point[StateIndex.ALT]

            cur_pos2 = np.array([x2, y2, z2])

            difference = np.linalg.norm(cur_pos2 - cur_pos1)

        print(f"Error Estimate: {difference}")
        step_size /= 2
        counter += 1

    states_rk = run_rk45(step_size, init, tmax, waypoints)

    # print final state
    final_state_rk = states_rk[-1]
    x2 = final_state_rk[StateIndex.POS_E]
    y2 = final_state_rk[StateIndex.POS_N]
    z2 = final_state_rk[StateIndex.ALT]
    print(f"RK45: Final state x, y, z: {round(x2, 3)}, {round(y2, 3)}, {round(z2, 3)}")


    # plot
    plot.plot_overhead(states, waypoints=waypoints)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot
    plot.plot_overhead(states_rk, waypoints=waypoints)
    filename = 'overhead_temp.png'
    plt.savefig(filename)
    print(f"Made {filename}")

def get_dense_output_points(rk45, num_points, waypoints, cur_waypoint_index):
    'get points between time steps using dense_output from rk45'
    distance = 1000
    points = []
    dense_output_func = rk45.dense_output()

    prev_time = rk45.t_old
    cur_time = rk45.t

    step_size = (cur_time - prev_time) / num_points

    for s in range(num_points):
        t = prev_time + s * step_size
        intermediate_state = dense_output_func(t)
        points.append(intermediate_state)

        cur_waypoint = np.array(waypoints[cur_waypoint_index])

        x = intermediate_state[StateIndex.POS_E]
        y = intermediate_state[StateIndex.POS_N]
        z = intermediate_state[StateIndex.ALT]

        cur_pos = np.array([x, y, z])
        distance = np.linalg.norm(cur_waypoint - cur_pos)

        if (distance < 200):
            return points, distance

    return points, distance


def run_rk45(step_size, init, t_max, waypoints):
    'run rk45 method and return state history'
    rtol = 0.00001
    atol = 0.00000001
    cur_waypoint_index = 0
    autopilot = WaypointAutopilot(waypoints[cur_waypoint_index])

    cur_state = np.array(init)

    rk45 = RK45(autopilot.der_func, 0, cur_state, t_max, rtol = rtol, atol = atol)

    state_history = [init.copy()]
    cur_time = 0

    # loop until we reach t_max
    while cur_time + 1e-6 < t_max:

        rk45.step()

        # add 10 points between rk45 steps
        points, distance = get_dense_output_points(rk45, 10, waypoints, cur_waypoint_index)
        state_history += points

        if(distance < 200):
            cur_waypoint_index += 1
            if(cur_waypoint_index>3):
                return state_history

            autopilot = WaypointAutopilot(waypoints[cur_waypoint_index])
            cur_state = state_history[-1]
            rk45 = RK45(autopilot.der_func, 0, cur_state, t_max, rtol = rtol, atol = atol)
            #print(f"Time {round(cur_time, 6)}, distance to waypoint: {round(distance, 3)} ft")

        #print(f"Time {round(cur_time, 6)}, distance to waypoint: {round(distance, 3)} ft")
        # add point at the end of the step
        state_history.append(rk45.y)
        cur_time += step_size

    return state_history

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

        #print(f"Time {round(cur_time, 6)}, distance to waypoint: {round(distance, 3)} ft")

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
