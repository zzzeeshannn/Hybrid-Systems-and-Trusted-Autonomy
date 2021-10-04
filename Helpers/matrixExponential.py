"""
This program is used to calculate matrix exponentials.
Created by: Zeeshan Shaikh
Date: 10/01/2021
"""

# Importing files here
import numpy as np
from scipy.linalg import expm

def main():
    a = np.array([[2.0, 3.0, 1.0], [1.0, 5.0, 0.0], [-1.0, 0.0, 5.0]])
    sol = expm(a/2)

    print(sol)

if __name__ == "__main__":
    main()