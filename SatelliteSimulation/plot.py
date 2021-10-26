'''
Satellite rendezvous plot
'''

import matplotlib.pyplot as plt
from matplotlib import collections

def init_plot():
    'plot the background and return axis object'

    # init plot
    plt.style.use(['bmh', 'bak_matplotlib.mlpstyle'])

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(1, 1, 1)

    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    ax.set_title('Satellite Rendezvous')

    ax.set_xlim([-950, 300])
    ax.set_ylim([-450, 300])

    y = 57.735
    line = [(-100.0, y), (-100.0, -y), (0.0, 0.0), (-100.0, y)]
    c1 = collections.LineCollection([line], colors=('gray'), linewidths=2, linestyle='dashed')
    ax.add_collection(c1)

    rad = 5
    line = [(-rad, -rad), (-rad, rad), (rad, rad), (rad, -rad), (-rad, -rad)]
    c2 = collections.LineCollection([line], colors=('red'), linewidths=2)
    ax.add_collection(c2)

    return ax

def plot_box(box, color):
    'plot a box'

    xmin, xmax = box[0]
    ymin, ymax = box[1]

    xs = [xmin, xmax, xmax, xmin, xmin]
    ys = [ymin, ymin, ymax, ymax, ymin]

    plt.plot(xs, ys, color)
