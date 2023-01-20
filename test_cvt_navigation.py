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


# dim = 10
# N_trajectories = 100
# for i in range(N_trajectories):
#     trajectory = np.reshape(cm.make_trajectory(dim//2,[100,100],sparse_env),(dim//2,2))
    # plt.plot(trajectory.T[0],trajectory.T[1])
    # points,total_trajectory = cm.make_trajectory(dim//2,[100,100],occ_grid)
    # plot_path(total_trajectory)


path10D = np.array([[500,50], [70,80], [30,430], [50, 900], [270,555]])
path20D = np.array([[500,50],[900,100], [930,40], [940, 523], [900,600], [768,332], [714,734],[312,751],[549,259],[222,222]])

plt.plot(path10D[:,0],path10D[:,1],label='10D')
plt.plot(path20D[:,0],path20D[:,1],label='20D')
plt.scatter(wall_indeces[1],wall_indeces[0],linewidths=.5,c='black',marker='s') # wall indeces x and y are flipped I don't know why

plt.ylim([0,1000])
plt.xlim([0,1000])
plt.legend()
plt.show()
