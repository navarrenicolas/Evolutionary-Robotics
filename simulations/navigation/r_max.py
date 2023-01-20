###
# Écrit par Augustin Chartouny
# 18 janvier 2023
#
# Modified by Nicolas Navare
# 19 january 2023
###

import numpy as np

class Rmax:

    def __init__(self, environment,gamma,Rmax,m):
        
        self.environment=environment
        self.size_environment=len(self.environment.states)
        self.size_actions=len(self.environment.actions)
        self.gamma = gamma
        self.Rmax=Rmax
        self.m = m

        self.R_VI=np.ones((self.size_environment,self.size_actions))*self.Rmax
        
        self.R = np.zeros((self.size_environment,self.size_actions))
        self.Rsum=np.zeros((self.size_environment,self.size_actions))
        
        self.nSA = np.zeros((self.size_environment,self.size_actions),dtype=np.int32)
        self.nSAS = np.zeros((self.size_environment,self.size_actions,self.size_environment),dtype=np.int32)
        
        self.tSAS = np.ones((self.size_environment,self.size_actions,self.size_environment))/self.size_environment
        self.Q = np.ones((self.size_environment,self.size_actions))/(1-self.gamma) # Optimistic initialization
        
        self.step_counter=0
                    
    def choose_action(self,cSA):
        self.step_counter+=1
        q_values=self.Q[self.environment.state_dict[cSA[0]],cSA[1]]
        return np.random.choice(np.flatnonzero(q_values == np.max(q_values)))
    
    
    def learn(self,old_state,reward,new_state,action):
                    oS = self.environment.state_dict[old_state]
                    nS = self.environment.state_dict[new_state]

                    self.nSA[oS][action] +=1
                    self.nSAS[oS][action][nS] += 1
                    self.Rsum[oS][action]+=reward
                    self.R[oS][action]=self.Rsum[oS][action]/self.nSA[oS][action]
                    
                    self.compute_reward_VI(oS,reward,action)                
                    self.compute_transitions(oS,nS,action)

                    
                    self.value_iteration()
    
    def compute_transitions(self,old_state,new_state,action):
        self.tSAS[old_state][action] = self.nSAS[old_state][action]/self.nSA[old_state][action]

        
    def compute_reward_VI(self,old_state, reward, action):
        if self.nSA[old_state][action]>=self.m: self.R_VI[old_state][action]=self.R[old_state][action]
        else : self.R_VI[old_state][action]=self.Rmax
    
    
    def value_iteration(self):
        #visited=np.where(self.nSA>=0,1,0) #wihout computing the Q-values only for visited (state,action) couples
        visited=np.where(self.nSA>=1,1,0) #Updates the Q-value (s,a) only when the agent took the action a in the state s at least once
        threshold=1e-3
        converged=False
        while not converged :
            max_Q = np.max(self.Q, axis=1)
            new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)
            
            diff = np.abs(self.Q[visited>0] - new_Q[visited>0])
            self.Q[visited>0]=new_Q[visited>0]
            if np.max(diff) < threshold:
                converged = True