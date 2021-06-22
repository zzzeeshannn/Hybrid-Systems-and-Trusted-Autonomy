# Advanced version of the "basicZonotope" program

# Importing libraries here
import math

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.linalg import expm

# Defining the zonotope here
class Zonotope:

    def __init__(self, center, generators, init_bounds = None):
        # Three inputs here: Center of the zonotope, set of generators and the initial bound of the zonotope
        self.center = center
        self.generators = generators.copy()

        # If initial bounds are passed
        if init_bounds is not None:
            # Generators are in the form of matrices in which each column represents a generator
            # Thus, number of columns = number of generators
            number_generators = self.generators.shape[1]
            assert len(init_bounds) == number_generators
            self.init_bounds = [[inital_bound[0], inital_bound[1]] for inital_bound in init_bounds]

        # Else
        if generators.size > 0:
            assert len(self.center) == self.generators.shape[0]
            if init_bounds is None:
                # Initial bounds are [-1, 1] for zonotopes by default
                self.init_bounds = [[-1, 1] for _ in self.generators.shape[0]]
        else:
            self.init_bounds = []

    def max(self, direction):
        # This function is used to find the maximum value in the zonotope in the passed direction
        output = self.center.copy()
        # Project vector
        projected_vector = np.dot(self.generators.transpose(), direction)

        for pv, row, ib in zip(projected_vector, self.generators.transpose(), self.init_bounds):
            factor = ib[1] if pv >= 0 else ib[0]

            output += factor * row

        return output


# Main function
def main():
    temp = 10