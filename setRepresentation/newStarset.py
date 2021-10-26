"""
Zonotope coding (CSE 510)
"""

import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt

from scipy.linalg import expm
from scipy.optimize import linprog


def plot(stateset, color='black', lw=1):
    """plot the set"""

    vert_list = verts(stateset)

    xs = [pt[0] for pt in vert_list]
    ys = [pt[1] for pt in vert_list]

    plt.plot(xs, ys, color, lw=lw)


def verts(stateset, num_directions=50):
    """get vertices of set"""

    result = []

    for theta in np.linspace(0, 2 * np.pi, num_directions):
        # optimize the set in angle theta

        vx = np.cos(theta)
        vy = np.sin(theta)

        direction_vec = np.array([[vx, vy]])

        pt = stateset.maximize(direction_vec)

        if not result or not np.allclose(result[-1], pt):
            result.append(pt)

    result.append(result[0])

    return result


class HPoly:
    """hpoly container"""

    def __init__(self, A_ub, b_ub):
        # A_ub * x <= b_ub

        self.A_ub = A_ub
        self.b_ub = b_ub

    def intersect(self, row, rhs):
        """add constraint to hpoly"""

        self.A_ub.append(list(row))
        self.b_ub.append(rhs)

    def maximize(self, vec):
        """return point that maximizes in the passed-in direction"""

        res = linprog(-1 * vec, A_ub=self.A_ub, b_ub=self.b_ub, bounds=(-np.inf, np.inf))

        assert res.success

        pt = res.x

        pt.shape = (2, 1)

        return pt


class StarSet:
    """Star set container"""

    def __init__(self, center, g_mat):
        self.center = center
        self.g_mat = g_mat

        # self.domain = UnitBox()
        A_ub = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        b_ub = [1, 1, 1, 1]

        self.domain = HPoly(A_ub, b_ub)

    def intersect(self, h, f):
        """intersect the set with hx <= f"""

        domain_row = h @ self.g_mat
        domain_rhs = f - h @ self.center

        # add constraint to the domain
        self.domain.intersect(domain_row, domain_rhs)

    def affine_transform(self, mat, vec=None):
        """transform the set by mat*x + vec"""

        self.g_mat = mat @ self.g_mat
        self.center = mat @ self.center

        if vec is not None:
            self.center += vec

    def maximize(self, direction_vec):
        """maximize zonotope"""

        if isinstance(direction_vec, list):
            direction_vec = np.array([direction_vec], dtype=float)

        # convert direction to domain
        domain_dir = direction_vec @ self.g_mat

        domain_pt = self.domain.maximize(domain_dir)

        range_pt = self.g_mat @ domain_pt + self.center

        return range_pt


def main2():
    """temp main"""

    # -1 <= x <= 1
    # -1 <= y <= 1

    # x <= 1
    # x >= 1  -->  -x <= -1

    # A_ub * x <= b_ub
    A_ub = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b_ub = [1, 1, 1, 1]

    hpoly = HPoly(A_ub, b_ub)

    hpoly.intersect([1, 1], 0)

    plot(hpoly)

    plt.xlim([-6, 6])
    plt.ylim([-6, 6])

    plt.show()


def main():
    """main entry point"""

    step_size = np.pi / 8
    max_steps = 10
    x_guard = 3.9

    # init: x in [-5, -4], y in [0, 1]
    g_mat = np.array([[0.5, 0.0], [0.0, 0.5]])
    center = np.array([[-4.5], [0.5]])
    init = StarSet(center, g_mat)

    # x' = y, y' = -x
    a_mat = np.array([[0.0, 1.0], [-1.0, 0.0]])
    ha_sol_mat = expm(a_mat * step_size)

    waiting_list = []
    init_tuple = (init, 'ha', 0)
    waiting_list.append(init_tuple)

    while waiting_list:
        stateset, mode, step = waiting_list.pop()

        color = 'black' if mode == 'ha' else 'r:'
        plot(stateset, color)

        while step < max_steps:

            if mode == 'ha':
                min_point = stateset.maximize([-1, 0])

                min_x = min_point[0]

                is_inside_invariant = min_x < x_guard
            else:
                is_inside_invariant = True

            if not is_inside_invariant:
                break

            if mode == 'ha':
                # intersect with mode invariant
                # add conition that x <= x_guard

                if stateset.maximize([1, 0])[0] >= x_guard:
                    constraint_row = np.array([1.0, 0.0])
                    constraint_rhs = x_guard

                    stateset.intersect(constraint_row, constraint_rhs)

                mat = ha_sol_mat
                vec = None
            else:
                assert mode == 'move_right'

                mat = np.identity(2)
                vec = np.array([[1.2], [0]])

            stateset.affine_transform(mat, vec)

            color = 'black' if mode == 'ha' else 'r:'
            lw = 1 if mode == 'ha' else 2
            plot(stateset, color, lw=lw)

            step += 1

            # check outgoing transitions
            if mode == 'ha':
                max_x = stateset.maximize([1, 0])[0]

                if max_x >= x_guard:
                    # intersect with x >= x_guard ---> -x <= -x_guard
                    # hx <= f
                    # h = [-1, 0], f = -x_guard
                    constraint_row = np.array([-1.0, 0.0])
                    constraint_rhs = -x_guard

                    copy_of_set = deepcopy(stateset)
                    copy_of_set.intersect(constraint_row, constraint_rhs)

                    tup = (copy_of_set, 'move_right', step)
                    waiting_list.append(tup)

                    plot(copy_of_set, 'r:', lw=2)

    plt.plot([x_guard, x_guard], [-10, 10], 'b--')

    plt.xlim([-6, 10])
    plt.ylim([-6, 6])
    plt.show()


if __name__ == "__main__":
    main()