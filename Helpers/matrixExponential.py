"""
This program is used to calculate matrix exponentials.
Created by: Zeeshan Shaikh
Date: 10/01/2021
"""

# Importing files here
import numpy as np
from scipy.linalg import expm
import math

def main():
    a = np.array([[0.0, 1.0], [-1.0, 0.0]])
    val = math.pi/4
    sol = expm(a*val)

    g_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    center = np.array([[0.0], [0.0]])
    print(sol @ g_mat)

if __name__ == "__main__":
    main()