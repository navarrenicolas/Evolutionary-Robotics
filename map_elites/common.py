#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
#

import math
import numpy as np
import multiprocessing
from pathlib import Path
import sys
import random
from collections import defaultdict
from sklearn.cluster import KMeans

# Modules for pathfinding
from simulations.navigation.a_star import a_star
from simulations.navigation.gridmap import OccupancyGridMap
from simulations.navigation.utils import dist2d


default_params = \
    {
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to paralleliez
        "batch_size": 100,
        # proportion of niches to be filled before starting
        "random_init": 0.1,
        # batch for random initialization
        "random_init_batch": 100,
        # when to write results (one generation = one batch)
        "dump_period": 10000,
        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": True,
        # min/max of parameters
        "min": 0,
        "max": 1,
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2
    }

class Species:
    def __init__(self, x, desc, fitness, centroid=None):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid


def polynomial_mutation(x):
    '''
    Cf Deb 2001, p 124 ; param: eta_m
    '''
    y = x.copy()
    eta_m = 5.0;
    r = np.random.random(size=len(x))
    for i in range(0, len(x)):
        if r[i] < 0.5:
            delta_i = math.pow(2.0 * r[i], 1.0 / (eta_m + 1.0)) - 1.0
        else:
            delta_i = 1 - math.pow(2.0 * (1.0 - r[i]), 1.0 / (eta_m + 1.0))
        y[i] += delta_i
    return y

def sbx(x, y, params):
    '''
    SBX (cf Deb 2001, p 113) Simulated Binary Crossover

    A large value ef eta gives a higher probablitity for
    creating a `near-parent' solutions and a small value allows
    distant solutions to be selected as offspring.
    '''
    eta = 10.0
    xl = params['min']
    xu = params['max']
    z = x.copy()
    r1 = np.random.random(size=len(x))
    r2 = np.random.random(size=len(x))

    for i in range(0, len(x)):
        if abs(x[i] - y[i]) > 1e-15:
            x1 = min(x[i], y[i])
            x2 = max(x[i], y[i])

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[i]
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = min(max(c1, xl), xu)
            c2 = min(max(c2, xl), xu)

            if r2[i] <= 0.5:
                z[i] = c2
            else:
                z[i] = c1
    return z


def iso_dd(x, y, params):
    '''
    Iso+Line
    Ref:
    Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
    GECCO 2018
    '''
    assert(x.shape == y.shape)
    p_max = np.array(params["max"])
    p_min = np.array(params["min"])
    a = np.random.normal(0, params['iso_sigma'], size=len(x))
    b = np.random.normal(0, params['line_sigma'])
    norm = np.linalg.norm(x - y)
    z = x.copy() + a + b * (x - y)
    return np.clip(z, p_min, p_max)


def variation(x, z, params):
    assert(x.shape == z.shape)
    y = sbx(x, z, params)
    return y

def __centroids_filename(k, dim):
    return 'centroids_' + str(k) + '_' + str(dim) + '.dat'


def __write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')


def cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    x = np.random.rand(samples, dim)
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, verbose=1)#,algorithm="full",  n_jobs=-1,)
    k_means.fit(x)
    __write_centroids(k_means.cluster_centers_)

    return k_means.cluster_centers_

def cvt_navigation(k,dim,samples, start_pos, maze, cvt_use_cache=True):
    # check if we have cached values
    # The samples must be selected based on the robot's navigation
    # dim : {2, 10, 20, 50, 250, 1000} 
    # An array of 2-tuples
    #
    # Selection of the centroids
    # 1. Choose a random time step in the trajectory (0,1000)
    # 2. Pick a random point within the radius of that time step
    # 3. Randomly select another time step
    # 4. Pick a point in the map that is 
        # - within the radius of the time step from the origin
        # - within the radius of the time difference between the other centroids
    
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)
    trajectories = np.zeros((samples,dim))
    for i in range(samples):
        trajectories[i,:] = make_trajectory(dim//2,start_pos,maze)
    
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, verbose=1)#,algorithm="full", n_jobs=-1)
    k_means.fit(trajectories)
    __write_centroids(k_means.cluster_centers_)

    return k_means.cluster_centers_

def make_trajectory(trajectory_len,start_pos,maze):
    pos = start_pos
    remaining_steps = 3000
    total_trajectory = []
    # trajectory = np.ndarray((trajectory_len,2))
    trajectory = []
    for i in range(1,trajectory_len):
        if(remaining_steps==0):
            # trajectory[i] = pos
            trajectory.append( pos )
        else:
            pos,path = next_trajectory_point(pos,remaining_steps,maze)
            steps = len(path) - 1
            remaining_steps -= steps
            trajectory.append( pos )
            total_trajectory += path[0:steps]
    return np.ndarray.flatten(np.array(trajectory)), total_trajectory
 

def next_trajectory_point(pos,remaining,maze):
    path_pos = True
    while(path_pos):
        [x,y] = sample_point(pos,remaining,maze)
        path_pos, path_px = check_line(pos,[x,y],remaining,maze)
    return [x,y], path_px

def sample_point(start,remaining,maze):
    [x,y] = [np.random.randint(max(0,start[0]-remaining),min(start[0]+remaining,999)), 
            np.random.randint(max(0,start[1]-remaining),min(start[1]+remaining,999))]
    while(maze.is_occupied_idx([x,y]) or dist2d(start,[x,y])>remaining):
        [x,y] = [np.random.randint(max(0,start[0]-remaining),min(start[0]+remaining,999)), 
            np.random.randint(max(0,start[1]-remaining),min(start[1]+remaining,999))]
    return [x,y]

def check_line(start,stop,remaining,maze):
    path, path_px = a_star(start, stop, maze, movement='4N')
    maze.clear_visited()
    if path:
        if len(path)>remaining:
            print('Path too long')
            return True, []
        print('Found Path')
        return False, path_px
    else:
        # print('Goal is not reachable')
        # print(stop)
        return True, []


def make_hashable(array):
    return tuple(map(float, array))


def parallel_eval(evaluate_function, to_evaluate, pool, params):
    if params['parallel'] == True:
        s_list = pool.map(evaluate_function, to_evaluate)
    else:
        s_list = map(evaluate_function, to_evaluate)
    return list(s_list)

# format: fitness, centroid, desc, genome \n
# fitness, centroid, desc and x are vectors
def __save_archive(archive, gen):
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ' ')
    filename = 'archive_' + str(gen) + '.dat'
    with open(filename, 'w') as f:
        for k in archive.values():
            f.write(str(k.fitness) + ' ')
            write_array(k.centroid, f)
            write_array(k.desc, f)
            write_array(k.x, f)
            f.write("\n")
