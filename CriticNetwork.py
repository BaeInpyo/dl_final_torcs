import tensorflow as tf
import numpy as np
import math

## for dgx setting
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'

HIDDEN1_UNIT = 300
HIDDEN2_UNIT = 600

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        #Now create the model
        self.output, self.action, self.state = self.create_critic_network('pred_critic', state_size, action_size)  
        self.target_output, self.target_action, self.target_state = self.create_critic_network('target_critic', state_size, action_size)  
        self.action_grads = tf.gradients(self.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

        self.copy_op = []
        self.pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred_critic')
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')
        for pred_var, target_var in zip(self.pred_vars, self.target_vars):
            self.copy_op.append(target_var.assign(
                TAU*pred_var.value() + (1-TAU)*target_var.value()))
    
    def gradients(self, states, actions):
	    return self.sess.run(self.action_grads, feed_dict={self.state: states, self.action: actions})[0]
    
    def target_train(self):
	    self.sess.run(self.copy_op)

    def train(self, state_and_action, y):
        self.sess.run(self.optimize, feed_dict={self.state: state_and_action[0], self.action: state_and_action[1], self.y: y})

    def create_critic_network(self, name, state_size, action_dim):
        with tf.variable_scope(name):
            ## will come from agent
            input_state = tf.placeholder(tf.float32, shape=[None, state_size])
            ## will come from action network
            input_action = tf.placeholder(tf.float32, shape=[None, action_dim])
            ## will come from agent
            self.y = tf.placeholder(tf.float32, shape=[None, action_dim])

            wf1 = tf.get_variable(name='wf1', shape=[state_size, HIDDEN1_UNIT])
            wf2_s = tf.get_variable(name='wf2_s', shape=[HIDDEN1_UNIT, HIDDEN2_UNIT])
            wf2_a = tf.get_variable(name='wf2_a', shape=[action_dim, HIDDEN2_UNIT])
            wf2 = tf.get_variable(name='wf2', shape=[HIDDEN2_UNIT, HIDDEN2_UNIT])
            wlogits = tf.get_variable(name='wlogits', shape=[HIDDEN2_UNIT, action_dim])

            bf1 = tf.constant(0.0, shape=[HIDDEN1_UNIT])
            bf2_s = tf.constant(0.0, shape=[HIDDEN2_UNIT])
            bf2_a = tf.constant(0.0, shape=[HIDDEN2_UNIT])
            bf2 = tf.constant(0.0, shape=[HIDDEN2_UNIT])
            blogits = tf.constant(0.0, shape=[action_dim])

            fc1 = tf.nn.relu(tf.add(tf.matmul(input_state, wf1), bf1))
            fc2_s = tf.add(tf.matmul(fc1, wf2_s), bf2_s)
            fc2_a = tf.add(tf.matmul(input_action, wf2_a), bf2_a)
            fc2_sum = tf.add(fc2_s, fc2_a)
            fc2 = tf.nn.relu(tf.add(tf.matmul(fc2_sum, wf2), bf2))
            logits = tf.add(tf.matmul(fc2, wlogits), blogits)
            losses = tf.reduce_mean(tf.square(self.y - logits))
            self.optimize = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(losses)

            return logits, input_action, input_state

 
    def predict(self, state_and_action):
        return self.sess.run(self.output, feed_dict={self.state: state_and_action[0], self.action: state_and_action[1]})

    def target_predict(self, state_and_action):
        return self.sess.run(self.target_output, feed_dict={self.target_state: state_and_action[0], self.target_action: state_and_action[1] })
