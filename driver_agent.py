# coding: utf-8
import gym
import tensorflow as tf
import numpy as np
import math

from ReplayMemory import ReplayMemory
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

import os

# If you want, add hyperparameters
MEMORY_SIZE = 100000
BATCH_SIZE  = 32
GAMMA       = 0.99
TAU         = 0.001
LRA         = 0.0001
LRC         = 0.001
ckp_dir     = 'torcs'
ckp_name    = 'trained'

class DriverAgent:
    def __init__(self, env_name, state_dim,action_dim):
        self.name = 'DriverAgent' # name for uploading results
        self.env_name = env_name
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Tensorflow Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        self.sess = tf.Session(config=config)

        # Actor & Critic Network
        self.actor = ActorNetwork(self.sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
        self.critic = CriticNetwork(self.sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
        
        # Replay Memory
        self.memory = ReplayMemory(MEMORY_SIZE)

        # loading networks. modify as you want 
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(ckp_dir + '/' + ckp_name)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    # Train code
    def train(self,state,action,reward,next_state,done):
        # Add information to the replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Get batch from the replay memory
        batch = self.memory.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = [0]*len(batch)

        # Get target Q value of the critic network
        target_Q = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])

        # Calculate answer(???) < I cannot rememeber name
        for i in range(len(batch)):
            if dones[k]:
                y_t[i] = rewards[i]
            else:
                y_t[i] = rewards[i] + GAMMA*target_Q[i]

        # Calculate loss value and gradient for each network, and train both
        loss += self.critic.train([states, actions], y_t)
        a_for_grad = self.actor.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()

    # save your own network
    def saveNetwork(self):
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)
        self.saver.save(self.sess, os.path.join(ckp_dir, ckp_name))
        pass

    def action(self,state):
        # return an action by state.
        action = np.zeros([self.action_dim])
        action_pre = actor.predict(state)
        
        # ACTION: without noise 
        action[0] = action_pre[0][0]
        action[1] = action_pre[0][1]
        action[2] = action_pre[0][2]

        return action

    def noise_action(self,state,epsilon):
        # return an action according to the current policy and exploration noise
        action = np.zeros([self.action_dim])
        noise = np.zeros([self.action_dim])

        action_pre = actor.predict(state)
        
        # NOISE: eps * (theta * (mu - x) + sigma * rand)
        noise[0] = max(epsilon, MIN_EPSILON) * \
                0.6*(0.0-action_pre[0][0]) + 0.30*np.random.randn(1) 
        noise[1] = max(epsilon, MIN_EPSILON) * \
                1.0*(0.5-action_pre[0][1]) + 0.10*np.random.randn(1) 
        noise[2] = max(epsilon, MIN_EPSILON) * \
                1.0*(-0.1-action_pre[0][2]) + 0.05*np.random.randn(1) 

        # ACTION: with noise 
        action[0] = action_pre[0][0] + noise[0]
        action[1] = action_pre[0][1] + noise[1]
        action[2] = action_pre[0][2] + noise[2]

        return action
