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

        direction = self.a_mat.transpose().dot(direction)

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
    
    plt.style.use(['bmh', 'bak_matplotlib.mlpstyle'])

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

    init_box = [[-5.0, -4.0], [0.0, 1.0]]
    dynamics_mat = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float) # mode 1: x' = y, y' = -x
    dx_mode_2 = 2.0 # mode 2: x' = 2, y' = 0
    time_step = math.pi / 8
    num_steps = 10
    mode_boundary = 3.9

    sol_mat = expm(dynamics_mat * time_step)

    #init_box.append([-0.2, 0.2])
    #init_a_mat = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    init_zono = Star(init_box)
    
    waiting_list = [] # 3-tuples: zonotope, mode, step number
    waiting_list.append((init_zono, 1, 0))

    while waiting_list:
        s, mode, step = waiting_list.pop()

        while step < num_steps and is_inside_invariant(s, mode, mode_boundary):
            # plot

            if mode == 1:
                mode1_intersect_invariant(s, mode_boundary)
            
            if mode == 1:
                s.plot('r-')
            else:
                s.plot('b:')

            step += 1

            # advance
            if mode == 1:
                s.a_mat = sol_mat.dot(s.a_mat)
                s.b_vec = sol_mat.dot(s.b_vec)
            else:
                s.b_vec += np.array([[dx_mode_2], [0]])

            # check if inside guard
            if mode == 1:
                max_x_pt = s.max([1, 0])
                max_x_val = max_x_pt[0]

                if max_x_val >= mode_boundary: # guard is true
                    successor = deepcopy(s)

                    # star.a_mat[0, :] + star.b_vec[0] <= boundary
                    row = successor.a_mat[0, :]
                    rhs = mode_boundary - successor.b_vec[0]

                    successor.a_ub_list.append(-row)
                    successor.b_ub_list.append(-rhs)
                    
                    waiting_list.append((successor, 2, step))


    plt.plot([mode_boundary, mode_boundary], [-2, 6], 'k--')
                    
    #plt.show()
    plt.savefig('star.png')

if __name__ == '__main__':
    init_plot()
    main()
