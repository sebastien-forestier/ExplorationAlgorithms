import numpy as np


def compute_explo(data, mins, maxs, gs=100):
    n = len(mins)
    assert len(data[0]) == n
    epss = (maxs - mins) / gs
    grid = np.zeros([gs] * n)
    #print np.size(grid), mins, maxs
    for i in range(len(data)):
        idxs = np.array((data[i] - mins) / epss, dtype=int)
        idxs[idxs>=gs] = gs-1
        idxs[idxs<0] = 0
        #print idxs
        grid[tuple(idxs)] = grid[tuple(idxs)] + 1
    grid[grid > 1] = 1
    return np.sum(grid)
