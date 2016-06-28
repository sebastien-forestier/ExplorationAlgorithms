import numpy as np
import time

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
    time.sleep(2)