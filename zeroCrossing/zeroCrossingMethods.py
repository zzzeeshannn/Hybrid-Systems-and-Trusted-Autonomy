"""
Author: Zeeshan Shaikh (@zzzeeshannn)
Date: 09/08/2021

Program to test bisection, secant and Newton's method for Zero Crossing
"""

# Importing libraries here
import numpy as np

# Defining the derivative function here
def derivative():
    """
    The derivative of the system is defined as:
    x' = 1
    y' = 0

    :return: x' and y'
    """

    x_der = 1
    y_der = 0

    return x_der, y_der

def check_incircle(state):
    """
    This function checks if the point is in the circle
    :return:
    -1 if it is in the circle
    0 if it is on the circle
    1 if it is outside
    """
    tolerance = 0.01
    x = state[0]
    y = state[1]

    if(x**2 + y**2) < 1 - tolerance:
        return -1
    elif (x**2 + y**2) < 1 + tolerance and (x**2 + y**2) > 1 - tolerance:
        return 0
    else:
        return 1

def der_func(state):
    """
    This function is the derivative of the function of our guard condition
    Since our guard function is x**2 + y**2 = 1, it's derivative is 2*(x + y) = 0
    :return:
    Derivative of the guard function
    """
    x, y = state

    return 2*(x + y)

# Defining the Bisection method here
def bisection(a, b, previous_state, current_state, derivative):
    """
    Logic:
    Bisection method is binary search equivalent for zero crossing method.
    Main conditions:
    f(mid) == 0: return mid
    sign(f(a)) == sign(f(midpoint)): a = mid
    sign(f(b)) == sign(f(midpoint)): a = mid
    where mid = (a + b)/2

    Since we have a temporal system, we use a and b as time of the system and calculate the state of the system
    at those time steps to check if they are within the circle (given in this problem)

    :param a: Time of the system before zero crossing guard
    :param b: Time of the system after zero crossing guard
    :param previous_state: State of the system before zero crossing guard
    :param current_state: State of the system after zero crossing guard
    :param der_function: function defining the ODE governing the system

    :return: Position of zero crossing

    """
    # Defining required parameters here
    flag = 1
    print("--------------- BISECTION METHOD ---------------")
    print(f"Starting at: {current_state} and {previous_state}")
    if(check_incircle(previous_state)*check_incircle(current_state) > 0):
        print("Wrong points for zero crossing")
        return None
    else:
        while (flag != 0):
            # Get the midpoint time step
            mid = (a + b) / 2
            # Calculate the step size
            step_size = mid - a

            # Simulate the next point from the state of the system "previous_state" at time "a" at the new time step "mid_time"
            new_state = previous_state + step_size * np.array(derivative())

            print(f"Current states: {new_state}, {previous_state} and {current_state} with step size: {step_size}")

            if (check_incircle(new_state) == 0):
                flag = 0
                new_x, new_y = new_state

                return new_state, mid
            elif (check_incircle(previous_state) * check_incircle(new_state) > 0):
                a = mid
            else:
                b = mid


# Defining the Secant method here
def secant(previous_state, current_state):
    """
    Logic:
    For the secant method, we calculate x_k where:
    x_k = a - f(a) * ( (b - a) / (f(b) - f(a)) )

    Note:
    As compared to the Bisection method where we iterated the time step, which was then used to find the state of the system at the mid time step,
    this method works directly with the state of the system to calulate x_k

    :param previous_state: State of the system before zero crossing guard
    :param current_state: State of the system after zero crossing guard

    :return: Position of zero crossing
    """
    # Defining required parameters here
    flag = 1
    counter = 1
    print("---------- SECANT METHOD ----------")
    print(f"Starting at: {previous_state} and {current_state}")
    if (check_incircle(previous_state) * check_incircle(current_state) > 0):
        print("Wrong points for zero crossing")
        return None
    else:
        while (flag != 0):
            x_k = previous_state - check_incircle(previous_state) * ((current_state - previous_state)/(check_incircle(current_state) - check_incircle(previous_state)))
            print(f"New state: {x_k} at counter {counter}")
            counter += 1
            if (check_incircle(x_k) == 0):
                flag = 0
                return x_k
            elif (check_incircle(previous_state)*check_incircle(x_k) < 0):
                current_state = x_k
            else:
                previous_state = x_k


# Defining the Newton's method here
def newton(previous_state):
    """
    Logic:
    This function calculates the zero crossing point using Newton's method.
    Here,
    x(n+1) = x(n) - f(x(n))/f'(x(n))

    Note:
    As compared to the Bisection method and Secant method, this method is not guaranteed to converge.
    But when it does, it's faster than the other two methods.

    :param previous_state:

    :return: Zero Crossing point
    """
    max_iter = 100
    temp_state = previous_state
    print("---------- NEWTON METHOD ----------")
    print(f"Starting at: {previous_state}")
    for _ in range(0, max_iter):
        print(temp_state)
        if(check_incircle(temp_state) == 0):
            print(f"Returning: {temp_state}")
            return temp_state
        else:
            temp_der = der_func(temp_state)
            if temp_der == 0:
                print("Derivative does not exists, inavlid calculation")
                return None
            else:
                temp_state -= check_incircle(temp_state)/der_func(temp_state)

    print("Finished running max iterations - No result found.")
    return None

def main():
    # Initializing required parameters here
    # Starting point of the system
    init_point = np.array([-5.0, 0.1])
    flag = 1
    step_size = 0.05
    # List to hold the states of the system
    states = [init_point.copy()]
    current_time = 0.0

    # Repeat the process till state of the system is within the defined circle
    while (flag != 0):
        # Update time of the system
        current_time += step_size
        # Simulate the next point
        next_state = states[-1] + step_size * np.array(derivative())
        # Check if the new point is in the circle
        position = check_incircle(next_state)

        if (position == -1):
            # System is in the circle, trigger zero crossing methods
            # The previous position acts as "a" for the bisection method
            # This works because we haven't appended the new state till now
            a = current_time - step_size
            b = current_time
            previous_state = states[-1]
            current_state = next_state
            # Calling the bisection method here
            bisection_zero_crossing, time = bisection(a, b, previous_state, current_state, derivative)

            # Calling the secant method here
            secant_zero_crossing = secant(previous_state, current_state)

            # Calling the secant method here
            newton_zero_crossing = newton(previous_state)

            # Get the x, y position to print from the bisection method
            bi_x, bi_y = bisection_zero_crossing
            sec_x, sec_y = secant_zero_crossing

            print("----------- FINAL OUTPUT -----------")
            print(f"Using the bisection method, the zero crossing point found was, x: {bi_x} and y: {bi_y} at time step {time}")
            print(f"Using the secant method, the zero crossing point found was, x: {sec_x} and y: {sec_y}")
            if(newton_zero_crossing != None):
                new_x, new_y = newton_zero_crossing
                print(f"Using the newton method, the zero crossing point found was, x: {new_x} and y: {new_y}")
            else:
                print("Bisection method did not converge!")

            flag = 0
        else:
            states.append(next_state)

if __name__ == '__main__':
    main()

