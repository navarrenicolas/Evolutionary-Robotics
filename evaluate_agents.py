import sys
sys.path.append('simulations/navigation/')
import pickle

import numpy as np
import matplotlib.pyplot as plt


class Rmax_env:
    def __init__(self):
        self.states =  [a+b+c+d+e for a in map(str,range(2)) for b in map(str,range(2)) for c in map(str,range(2)) for d in map(str,range(8)) for e in map(str,range(3))]
        self.actions = [0,1,2]
        self.state_dict = dict([(state,num) for num, state in enumerate(self.states)])

check_agents = True
check_times = False


def show_SA(i):
    # fig = plt.figure(figsize=(10,10))
    plt.matshow(agents[i].nSA.reshape((24,24)))
    plt.show()


def save_SA(i):
    # fig = plt.figure(figsize=(10,10))
    plt.matshow(agents[i].nSA.reshape((24,24)))
    plt.savefig('images/run2/'+str(i)+'.png')
    plt.close()


if check_agents:
    agents = []
    for i in range(50):
        file = f'obj_trial_{i}.pkl'
        with open('./simulations/navigation/log/agents/'+file, 'rb') as inp:
            agents.append(pickle.load(inp))
        save_SA(i)
            

if check_times:
    # times = np.loadtxt('simulations/navigation/log/1674174055.3266644-TrialDurations-exp4-rmax.txt')
    times = np.loadtxt('simulations/navigation/log/1674177845.934214-TrialDurations-exp4-rmax.txt')
    plt.plot(times)
    plt.show()


def img2gif(img_dir,output_file):
    """
    Transform a folder of images (img_dir) into a gif and save it with name output_file
    """
    import glob
    from PIL import Image

    # filepaths
    import os
    files = os.listdir(img_dir)
    # files = [ _ for _ in files if ".png" in _ ]
    # files = [int((_.split("test"))[1].split(".")[0]) for _ in files]
    files.sort()
    fp_in= [img_dir+str(_) for _ in files]
    fp_out = output_file

    img, *imgs = [Image.open(f).resize((800,800)) for f in fp_in]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=120, loop=0)

# img2gif("images/","dynamics/nSA.gif")