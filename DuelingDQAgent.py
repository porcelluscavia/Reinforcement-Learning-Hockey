import gym
from gym import spaces
import numpy as np
import itertools
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import pylab as plt
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import hockey_env as h_env

import memory as mem   
from feedforward import Feedforward


class QFunction(Feedforward):
    #make this more customizable later!
    def __init__(self, observation_dim, action_dim, learning_rate, hidden_sizes=[100,100]):
        super(QFunction, self).__init__(input_size=observation_dim, hidden_sizes=hidden_sizes, 
                         output_size=action_dim)
#         self.observation_dim = observation_dim
#         self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_units = 512
        
        self.feature_layer = nn.Linear(self.input_size, 512)
        self.value_str = nn.Linear(512, 1)
        self.advantage_str = nn.Linear(512, self.output_size)
        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate, 
                                        eps=0.000001)
        
        self.loss = torch.nn.MSELoss()
        
    def new_forward(self, observations):
    	#advantages and values from states combined to make qvalues
        feats = F.relu(self.feature_layer(observations))
        vals = self.value_str(feats)
        adv = self.advantage_str(feats)
        qs = vals + (adv - adv.mean())
        return qs
    
    def Q_values(self, observations, actions):
        #forward with two networks
        return self.new_forward(observations).gather(1, actions[:,None]) 
    
    def predict(self, x):
        with torch.no_grad():
            return self.new_forward(torch.from_numpy(x.astype(np.float32))).numpy()

    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)
        
    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)
    
class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.    
    """
    def __init__(self, env, Q=None, Q_next= None, **userconfig):
        
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._action_n = env.observation_space.shape[0]
        self._config = {
            "eps": 0.05,            # Epsilon in epsilon greedy policies                        
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate": 0.0002,
            "update_target_every": 20,
            "use_target_net":False
        }
        self._config.update(userconfig)        
        self._eps = self._config['eps']
        
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])
                
        if not Q:
            # Q Network
            self.Q = QFunction(observation_dim=self._observation_space.shape[0], 
                               action_dim=self._action_n,
                               learning_rate = self._config["learning_rate"])
        else:
            self.Q = Q
            
        if not Q_next:
        # Q Network
            self.Q_next = QFunction(observation_dim=self._observation_space.shape[0], 
                                  action_dim=self._action_n,
                                  learning_rate = 0)
        else:
            self.Q_next = Q_next
        
        
        
        self._update_target_net()
        self.train_iter = 0
    
        
    def fit(self, observations, actions, targets):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts)
        # Compute Loss
        loss = self.loss(pred, torch.from_numpy(targets).float())
            
    def _update_target_net(self):        
        self.Q_next.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else: 
            action = random.randint(0,7)        
        return action
    
    
    def store_transition(self, transition):
        self.buffer.add_transition(transition)
            
    def train(self, iter_fit=32):
        losses = []
        self.train_iter+=1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()  
        for i in range(iter_fit):
            self.Q.optimizer.zero_grad()
            # sample from the replay buffer
            data=self.buffer.sample(batch=self._config['batch_size'])
            s = np.stack(data[:,0]) # s_t
            a = np.stack(data[:,1]) # a_t
            rew = np.stack(data[:,2])[:,None] # rew  (batchsize,1)
            s_prime = np.stack(data[:,3]) # s_t+1
            done = np.stack(data[:,4])[:,None] # done signal  (batchsize,1)
            acts = torch.from_numpy(a)
            self.Q.train()
            self.Q_next.train()
            # Forward pass
            curr_Q = self.Q.new_forward(torch.from_numpy(s).float()).gather(1, acts.unsqueeze(1))
            curr_Q = curr_Q.squeeze(1)
            next_Q = self.Q.new_forward(torch.from_numpy(s_prime).float())
            max_next_Q = torch.max(next_Q, 1)[0]
            v_prime = np.max(self.Q_next.predict(s_prime), axis=-1, keepdims=True)
            gamma=self._config['discount']
            expected_Q = rew + gamma * v_prime

            loss = self.Q.loss(curr_Q, torch.from_numpy(expected_Q).float().squeeze(1))
            # Backward pass
            loss.backward()
            self.Q.optimizer.step()
            losses.append(loss.item())
                
        return losses