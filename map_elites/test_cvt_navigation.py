import common as cm

import numpy as np
import matplotlib.pyplot as plt


# centroids = cm.cvt_navigation(k=100,dim=2,samples=1000,start_pos = [500,100],cvt_use_cache=False)

# plt.scatter(centroids.T[0],centroids.T[1])

dim = 50
N_trajectories = 10
# trajectories = np.zeros(shape=(N_trajectories,dim//2,2))
for i in range(N_trajectories):
    trajectory = np.reshape(cm.make_trajectory(dim//2,[500,100]),(dim//2,2))
    plt.plot(trajectory.T[0],trajectory.T[1])
plt.ylim([0,1000])
plt.xlim([0,1000])
plt.show()