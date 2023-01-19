import os
import neat
import numpy as np
import multiprocessing
import pickle
from pathlib import Path
from simulation import Environment
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree


def __save_archive(archive, gen):
    filename = './archives/archive_' + str(gen) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(archive, f, pickle.HIGHEST_PROTOCOL)


def make_hashable(array):
    return tuple(map(float, array))

def __add_to_archive(s, centroid, archive, kdt):
    niche_index = kdt.query(np.array([np.array(centroid)]), k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = make_hashable(niche)
    s.centroid = n
    if n in archive:
        if s.fitness > archive[n].fitness:
            archive[n] = s
            return 1
        return 0
    else:
        archive[n] = s
        return 1

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

    if dim==2:
        x = np.random.rand(samples, dim)
    else:
        raise NotImplementedError

    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, verbose=1)#,algorithm="full")
    k_means.fit(x)
    __write_centroids(k_means.cluster_centers_)

    return k_means.cluster_centers_

class Species:
    def __init__(self, x, desc, fitness, centroid=None):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid


def eval_genome(genome, config):
    print('Simulating Khera...')
    # for genome_id, genome in genomes:
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = Environment()
    perf, desc = env.simulate(net) # run robot for 3k timesteps
    print(f'Performance of Genome : {-perf}, terminated at {desc}')
    s = Species(genome, desc, -perf)
    __add_to_archive(s, desc, archive, kdt)
    # genome.fitness = -perf
    global genCnt
    genCnt += 1
    if genCnt %2000 == 0:
        print(f'Saving Archive at gen: {genCnt%2000}')
        __save_archive(archive, genCnt)

    del env
    return -perf


def run(config_file):
    # Load configuration.
    print('Starting NEAT ...')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    # Run for up to 990 generations.
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1')
    winner = p.run(pe.evaluate, 990)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    N_niches = 5000
    dim_map = 2
    samples = int(1000e03)
    global genCnt
    genCnt = 0

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')

    # creating centroids
    c = cvt(N_niches, dim_map, samples)
    print('Creating KDTree...')
    kdt = KDTree(c, leaf_size = 30, metric='euclidean')

    # create an empty archive
    print('Creating empty archive')
    archive = {}

    run(config_path)

