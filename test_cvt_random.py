
from map_elites import cvt_random as cm

import numpy as np
import matplotlib.pyplot as plt
import os

dims = [2,10,20,50,250,1000]
# dims = [2,10]

for i in range(10):
    for dim in dims:
        cm.cvt_navigation(k=5000,dim=dim,samples=10000,start_pos = [500,100],cvt_use_cache=False)
    os.system(f'mkdir centroids_{i}')
    os.system(f'mv centroids/* centroids_{i}')
# plt.scatter(centroids.T[0],centroids.T[1])


# os.system('mkdir centroids')