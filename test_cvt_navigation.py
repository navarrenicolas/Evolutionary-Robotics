import numpy as np
import matplotlib.pyplot as plt

from map_elites import common as cm
from simulations.navigation.env import MazeEnv

import scipy.sparse as sprs

## Environment setup
env = MazeEnv(1000, 200, 100, 12)
sparse_env = sprs.coo_matrix(env.t)
wall_indeces = sparse_env.nonzero()


dim = 10
N_trajectories = 1
for i in range(N_trajectories):
    trajectory = np.reshape(cm.make_trajectory(dim//2,[100,100],sparse_env),(dim//2,2))
    plt.plot(trajectory.T[0],trajectory.T[1])

plt.scatter(wall_indeces[0],wall_indeces[1],linewidths=1,c='m')
plt.ylim([0,1000])
plt.xlim([0,1000])
plt.show()



## Centroid testing

# # centroids = cm.cvt_navigation(k=100,dim=2,samples=1000,start_pos = [500,100],cvt_use_cache=False)
# # plt.scatter(centroids.T[0],centroids.T[1])

