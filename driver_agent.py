# coding: utf-8
import gym
import tensorflow as tf
import numpy as np
import math

# If you want, add hyperparameters


class DriverAgent:
    def __init__(self, env_name, state_dim,action_dim):
        self.name = 'DriverAgent' # name for uploading results
        self.env_name = env_name
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # loading networks. modify as you want 
#        self.saver = tf.train.Saver()
#        checkpoint = tf.train.get_checkpoint_state("path_to_save/")
#        if checkpoint and checkpoint.model_checkpoint_path:
#            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
#            print("Successfully loaded:", checkpoint.model_checkpoint_path)
#        else:
#            print("Could not find old network weights")

    def train(self,state,action,reward,next_state,done):
        # train code
        pass
            
    def saveNetwork(self):
        # save your own network
        pass

    def action(self,state):
        # return an action by state.
        return np.array([0,0,0])

    def noise_action(self,state,epsilon):
        # return an action according to the current policy and exploration noise
        return np.array([0,0,0])
