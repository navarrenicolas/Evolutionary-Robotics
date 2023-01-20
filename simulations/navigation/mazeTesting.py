#!/usr/bin/env python

from radarGuidance import *
from wallFollower import *

from r_max import Rmax

import random #used for the random choice of a strategy
import numpy as np
import math

import pickle

#--------------------------------------
# Position of the goal:
goalx = 500
goaly = 500
# Initial position of the robot:
initx = 500
inity = 35
# strategy choice related stuff:
choice = -1
choice_tm1 = -1
tLastChoice = 0
rew = 0

i2name=['wallFollower','radarGuidance','straight']

# Parameters of State building:
# threshold for wall consideration
th_neglectedWall = 50
# threshold to consider that we are too close to a wall
# and a punishment should be delivered
th_obstacleTooClose = 20
# angular limits used to define states
angleLMin = 0
angleLMax = 55

angleFMin=56
angleFMax=143

angleRMin=144
angleRMax=199

# Q-learning related stuff:
# definition of states at time t and t-1
S_t = ''
S_tm1 = ''
alpha = 0.4
beta = 4
gamma = 0.95

'''
Environment for R_Max algorithm
'''
class Rmax_env:
  def __init__(self):
    self.states =  [a+b+c+d+e for a in map(str,range(2)) for b in map(str,range(2)) for c in map(str,range(2)) for d in map(str,range(8)) for e in map(str,range(3))]
    self.actions = [0,1,2]
    self.state_dict = dict([(state,num) for num, state in enumerate(self.states)])

file = f'obj_trial_{49}.pkl'
with open('./simulations/navigation/log/agents/'+file, 'rb') as inp:
    rmax = pickle.load(inp)

#--------------------------------------
# the function that selects which controller (radarGuidance or wallFollower) to use
# sets the global variable "choice" to 0 (wallFollower) or 1 (radarGuidance)
# * arbitrationMethod: how to select? 'random','randPersist','qlearning'
def strategyGating(arbitrationMethod,verbose=True):
  global choice
  global choice_tm1
  global tLastChoice
  global rew

  # The chosen gating strategy is to be coded here:
  #------------------------------------------------
  if arbitrationMethod=='random':
    choice = random.randrange(2)
  #------------------------------------------------
  elif arbitrationMethod=='randomPersist':
    if choice == -1:
      choice = 0
      tLastChoice = time.time()
    elif time.time() - tLastChoice > 2:
      choice = random.randrange(2)
      tLastChoice = time.time()
  #------------------------------------------------
  elif arbitrationMethod=='rmax':
    if S_tm1 == '':
      choice_tm1 = choice
      choice = 2 # Go straight
      tLastChoice = time.time()
    elif time.time() - tLastChoice > 2:
          print(f'too long')
          choice_tm1 = choice
          choice = random.randrange(3)
          tLastChoice = time.time()
    elif rew != 0: 
      rmax.learn(S_tm1,rew,S_t,choice)
      choice_tm1 = choice
      choice = rmax.choose_action((S_t,choice))
      tLastChoice = time.time()
      rew = 0

  #------------------------------------------------
  else:
    print(arbitrationMethod+' unknown.')
    exit()

  if verbose:
    print("strategyGating: Active Module: "+i2name[choice])
#--------------------------------------
def filterState(state):
  return {k:v for k,v in Q.items() if k[0] == state}
#--------------------------------------
def getMaxValue(state):
  if len(filterState(state).values())==0:
    print('UNENCOUNTERED STATE')
    return 0
  return max(filterState(state).values())
#--------------------------------------
def softmax(vals):
  return np.exp(vals*beta)/sum(np.exp(vals*beta))
#--------------------------------------
def sampleProbs(vals):
  return np.random.choice(range(len(vals)),1,p=softmax(vals))[0]

#--------------------------------------
def buildStateFromSensors(laserRanges,radar,dist2goal):
  S   = ''
  # determine if obstacle on the left:
  wall='0'
  if min(laserRanges[angleLMin:angleLMax]) < th_neglectedWall:
    wall ='1'
  S += wall
  # determine if obstacle in front:
  wall='0'
  if min(laserRanges[angleFMin:angleFMax]) < th_neglectedWall:
    wall ='1'
    #print("Mur Devant")
  S += wall
  # determine if obstacle on the right:
  wall='0'
  if min(laserRanges[angleRMin:angleRMax]) < th_neglectedWall:
    wall ='1'
  S += wall

  S += str(radar)

  if dist2goal < 125:
    S+='0'
  elif dist2goal < 250:
    S+='1'
  else:
    S+='2'
  #print('buildStateFromSensors: State: '+S)

  return S

#--------------------------------------
def go_straight():
  return (5,5)
#--------------------------------------
def main():
  global S_t
  global S_tm1
  global rew

  settings = Settings('worlds/entonnoir.xml')

  env_map = settings.map()
  robot = settings.robot()

  d = Display(env_map, robot)

#  method = 'randomPersist'
  method = 'qlearning'
  method = 'rmax'
  # experiment related stuff
  startT = time.time()
  trial = 0
  nbTrials = 3000
  trialDuration = np.zeros((nbTrials))

  i = 0
  while i<nbTrials:
    # update the display
    #-------------------------------------
    d.update()
    # get position data from the simulation
    #-------------------------------------
    pos = robot.get_pos()
    # print("##########\nStep "+str(i)+" robot pos: x = "+str(int(pos.x()))+" y = "+str(int(pos.y()))+" theta = "+str(int(pos.theta()/math.pi*180.)))

    # has the robot found the reward ?
    #------------------------------------
    dist2goal = math.sqrt((pos.x()-goalx)**2+(pos.y()-goaly)**2)
    # if so, teleport it to initial position, store trial duration, set reward to 1:
    if (dist2goal<20): # 30
      print('***** REWARD REACHED *****')
      pos.set_x(initx)
      pos.set_y(inity)
      robot.set_pos(pos) # format ?
      # and store information about the duration of the finishing trial:
      currT = time.time()
      trialDuration[trial] = currT - startT
      startT = currT
      print("Trial "+str(trial)+" duration:"+str(trialDuration[trial]))
      save_object(rmax,'log/agents/obj_trial_'+str(trial)+'.pkl')
      trial +=1
      rew = 1

    # get the sensor inputs:
    #------------------------------------
    lasers = robot.get_laser_scanners()[0].get_lasers()
    laserRanges = []
    for l in lasers:
      laserRanges.append(l.get_dist())

    radar = robot.get_radars()[0].get_activated_slice()

    bumperL = robot.get_left_bumper()
    bumperR = robot.get_right_bumper()


    # 2) has the robot bumped into a wall ?
    #------------------------------------
    if bumperR or bumperL or min(laserRanges[angleFMin:angleFMax]) < th_obstacleTooClose:
    # if bumperR or bumperL or min(laserRanges[0:3]) < th_obstacleTooClose:
      rew = -1
      print("***** BING! ***** "+i2name[choice])

    # 3) build the state, that will be used by learning, from the sensory data
    #------------------------------------
    S_tm1 = S_t
    S_t = buildStateFromSensors(laserRanges,radar, dist2goal)

    #------------------------------------
    strategyGating(method,verbose=False)
    if choice==0:
      v = wallFollower(laserRanges,verbose=False)
    elif choice == 1:
      v = radarGuidance(laserRanges,bumperL,bumperR,radar,verbose=False)
    elif choice == 2:
      v = go_straight()
      print('GOING STRAIGHT')

    i+=1
    robot.move(v[0], v[1], env_map)
    time.sleep(0.01)

  # When the experiment is over:
  np.savetxt('log/'+str(startT)+'-TrialDurations-exp4-'+method+'.txt',trialDuration)
  
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


#--------------------------------------

if __name__ == '__main__':
  random.seed()
  main()
