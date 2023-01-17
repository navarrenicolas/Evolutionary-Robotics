import numpy as np
import matplotlib.pyplot as plt

from map_elites import common as cm
from simulations.navigation.env import MazeEnv
from simulations.navigation.gridmap import OccupancyGridMap


import scipy.sparse as sprs

from simulations.navigation.a_star import a_star
from simulations.navigation.utils import plot_path


## Environment set
env = MazeEnv(1000, 200, 100, 12)
occ_grid = OccupancyGridMap(env.t,1)

sparse_env = sprs.coo_matrix(env.t)
wall_indeces = sparse_env.nonzero()


# a_star([100,100], [207,774], occ_grid, movement='4N')

# a_star([100,100], [207,774], OccupancyGridMap(env.t,1), movement='4N')




dim = 10
N_trajectories = 1
for i in range(N_trajectories):
    # trajectory = np.reshape(cm.make_trajectory(dim//2,[100,100],sparse_env),(dim//2,2))
    # plt.plot(trajectory.T[0],trajectory.T[1])
    points,total_trajectory = cm.make_trajectory(dim//2,[100,100],occ_grid)
    plot_path(total_trajectory)
plt.scatter(wall_indeces[1],wall_indeces[0],linewidths=1,c='m') # wall indeces x and y are flipped I don't know why
# plt.matshow(env.t)
plt.ylim([0,1000])
plt.xlim([0,1000])
plt.show()


## Centroid testing

# # centroids = cm.cvt_navigation(k=100,dim=2,samples=1000,start_pos = [500,100],cvt_use_cache=False)
# # plt.scatter(centroids.T[0],centroids.T[1])

