'''
Zonotope reach
'''

import math
from copy import deepcopy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.linalg import expm
from scipy.optimize import linprog

class Star:
    'linear star set class'

    def __init__(self, box, a_mat=None):

        self.box = np.array(box, dtype=float)
        self.a_mat = a_mat if a_mat is not None else np.identity(self.box.shape[0])

        self.dims = self.a_mat.shape[0]
        self.gens = self.a_mat.shape[1]
        
        self.b_vec = np.zeros((self.dims, 1))

        # constraints Ax <= b
        self.a_ub_list = []
        self.b_ub_list = []

    def verts(self, xdim=0, ydim=1):
        'get verticies for plotting 2d projections'

        verts = []

        for angle in np.linspace(0, 2*math.pi, 32):
            
            direction = np.zeros((self.dims, ))
            direction[xdim] = math.cos(angle)
            direction[ydim] = math.sin(angle)

            pt = self.max(direction)
            xy_pt = (pt[xdim][0], pt[ydim][0])

            if verts and np.allclose(xy_pt, verts[-1]):
                continue

            verts.append(xy_pt)

        return verts

    def plot(self, color='k-o', xdim=0, ydim=1):
        'plot 2d projections'

        v_list = self.verts(xdim=xdim, ydim=ydim)

        xs = [v[xdim] for v in v_list]
        xs.append(v_list[0][xdim])

        ys = [v[ydim] for v in v_list]
        ys.append(v_list[0][ydim])

        plt.plot(xs, ys, color)

    def max(self, direction):
        '''returns the point in the box that is the maximum in the passed in direction

        if x is the point and c is the direction, this should be the maximum dot of x and c
        '''

        #direction = self.a_mat.transpose().dot(direction)
        direction = np.array(direction, dtype=float)
        direction = direction.dot(self.a_mat)

        # box has two columns and n rows
        # direction is a vector (one column and n rows)

        # returns a point (one column with n rows)

        box = self.box

        a_ub = None
        b_ub = None

        if self.a_ub_list:
            a_ub = np.array(self.a_ub_list, dtype=float)
            b_ub = np.array(self.b_ub_list, dtype=float)

        rv = linprog(-1 * direction, bounds=box, A_ub=a_ub, b_ub=b_ub)
        assert rv.success

        pt = np.array(rv.x)
        pt.shape = (self.dims, 1)

        return self.a_mat.dot(pt) + self.b_vec

def init_plot():
    'initialize plotting style'

    try:
        matplotlib.use('TkAgg') # set backend
    except:
        pass
    
    plt.style.use(['bmh'])

    plt.axis('equal')

def is_inside_invariant(s, mode, boundary):
    'is the zonotope inside the invariant for the mode?'

    rv = True

    if mode == 1:
        min_x_pt = s.max([-1, 0])
        min_x_val = min_x_pt[0]

        tol = 1e-6

        if min_x_val + tol >= boundary: # left the invariant
            rv = False

    return rv

def mode1_intersect_invariant(star, boundary):
    'interset start with constraint x <= boundary'

    max_x = star.max([1, 0])[0]

    if max_x > boundary: # add constraint
        # star.a_mat[0, :] + star.b_vec[0] <= boundary

        row = star.a_mat[0, :]
        rhs = boundary - star.b_vec[0]

        star.a_ub_list.append(row)
        star.b_ub_list.append(rhs)

def main():
    'main entry point'

    init_box = [[-1.0, 1.0], [-1.0, 1.0]]
    s = Star(init_box)

    x = math.sqrt(2) / 2
    g_mat = np.array([[x, x], [-x, 2*x]])

    s.a_mat = g_mat
    s.b_vec = np.array([[0.0], [-0.5]])

    s.plot('b-')

    # plot dotted line at y = 0
    plt.plot([-3, 3], [0, 0], 'k:')

    # For the set where y >= 0
    s2 = deepcopy(s)
    s2.a_ub_list = [s2.a_mat[1, :]]
    s2.b_ub_list = [0 - s2.b_vec[1][0]]

    s2.plot('r-')

    # For the set where y <= 0
    s3 = deepcopy(s)
    s3.a_ub_list = [-1 * s3.a_mat[1, :]]
    s3.b_ub_list = [s3.b_vec[1][0]]

    s3.plot('g-')

    # Projection
    s4 = deepcopy(s)
    #NOTE: Convert to numpy array to avoid list error
    #Delete Note before submission
    affine_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
    s4.a_mat = affine_matrix.dot(s4.a_mat)
    s4.b_vec = affine_matrix.dot(s4.b_vec)

    s4.plot('cyan')

    plt.savefig('nn-star.png')

if __name__ == '__main__':
    init_plot()
    main()
