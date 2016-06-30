import numpy as np
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
from numpy import pi, array, linspace, hstack, zeros, transpose
from matplotlib import animation
from IPython.display import HTML, display, Image, clear_output
from ipywidgets import interact_manual
from numpy.random import random, normal

from explauto import SensorimotorModel
from explauto.sensorimotor_model.non_parametric import NonParametric
from explauto import InterestModel
from explauto.interest_model.discrete_progress import DiscretizedProgress
from explauto.utils import rand_bounds, bounds_min_max, softmax_choice, prop_choice
from explauto.environment.dynamic_environment import DynamicEnvironment
from explauto.interest_model.competences import competence_exp, competence_dist
from explauto.environment.modular_environment import FlatEnvironment, HierarchicalEnvironment

from environment import Arm, Ball, Stick, ArmBall, ArmStickBalls
from learning_module import LearningModule
#from utils import compute_explo, display_movement

grid_size = 10


def compute_explo(data, mins, maxs, gs=100):
    n = len(mins)
    if len(data) == 0:
        return 0
    else:
        assert len(data[0]) == n
        epss = (maxs - mins) / gs
        grid = np.zeros([gs] * n)
        for i in range(len(data)):
            idxs = np.array((data[i] - mins) / epss, dtype=int)
            idxs[idxs>=gs] = gs-1
            idxs[idxs<0] = 0
            grid[tuple(idxs)] = grid[tuple(idxs)] + 1
        grid[grid > 1] = 1
        return np.sum(grid)

def display_movement(fig, ax, environment, time_step=0.04):
    fig.show()
    fig.canvas.draw()
    ax.set_aspect('equal')
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    background = fig.canvas.copy_from_bbox(ax.bbox)

    for i in range(50):
        start = time.time()
        fig.canvas.restore_region(background)
        lines = environment.env.plot_update(ax, i)
        for line in lines:
            ax.draw_artist(line)
        fig.canvas.blit(ax.bbox)

        end = time.time()
        remain = start + time_step - end
        if remain > 0:
            time.sleep(remain)
    time.sleep(1)
    
    