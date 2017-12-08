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
        self.L2 = 0.0001

        #Now create the model
        self.output, self.action, self.state, self.vars = self.create_critic_network('pred_critic', state_size, action_size)  
        self.target_state, self.target_action, self.target_output, self.target_update = \
                self.create_target_network(state_size, action_size, self.vars)
        #self.target_output, self.target_action, self.target_state, _ = self.create_critic_network('target_critic', state_size, action_size)  
        
        # Gradient
        self.action_grads = tf.gradients(self.output, self.action)  #GRADIENTS for policy update
        
        # Optimizer
        self.create_training_method()

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

        self.target_train()
        #self.copy_op = []
        #self.pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred_critic')
        #self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')
        #for pred_var, target_var in zip(self.pred_vars, self.target_vars):
        #    self.copy_op.append(target_var.assign(
        #        TAU*pred_var.value() + (1-TAU)*target_var.value()))
    
    def create_training_method(self):
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.vars])
        self.losses = tf.reduce_mean(tf.square(self.y - self.output)) + weight_decay
        self.optimize = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.losses)

    def create_critic_network(self, name, state_size, action_dim):
        with tf.variable_scope(name):
            input_state = tf.placeholder(tf.float32, shape=[None, state_size])
            input_action = tf.placeholder(tf.float32, shape=[None, action_dim])
            
            #wf1 = tf.get_variable(name='wf1', shape=[state_size, HIDDEN1_UNIT])
            #wf2_s = tf.get_variable(name='wf2_s', shape=[HIDDEN1_UNIT, HIDDEN2_UNIT])
            #wf2_a = tf.get_variable(name='wf2_a', shape=[action_dim, HIDDEN2_UNIT])
            #wf2 = tf.get_variable(name='wf2', shape=[HIDDEN2_UNIT, HIDDEN2_UNIT])
            #wlogits = tf.get_variable(name='wlogits', shape=[HIDDEN2_UNIT, action_dim])
            #
            #bf1 = tf.Variable(tf.zeros([HIDDEN1_UNIT]))
            #bf2_s = tf.Variable(tf.zeros([HIDDEN2_UNIT]))
            #bf2_a = tf.Variable(tf.zeros([HIDDEN2_UNIT]))
            #bf2 = tf.Variable(tf.zeros([HIDDEN2_UNIT]))
            #blogits = tf.Variable(tf.zeros([action_dim]))
            #
            #fc1 = tf.nn.relu(tf.add(tf.matmul(input_state, wf1), bf1))
            #fc2_s = tf.add(tf.matmul(fc1, wf2_s), bf2_s)
            #fc2_a = tf.add(tf.matmul(input_action, wf2_a), bf2_a)
            #fc2_sum = tf.add(fc2_s, fc2_a)
            #fc2 = tf.nn.relu(tf.add(tf.matmul(fc2_sum, wf2), bf2))
            #logits = tf.add(tf.matmul(fc2, wlogits), blogits)
            
            wf1 = self.variable([state_size, HIDDEN1_UNIT], state_size)
            wf2_s = self.variable([HIDDEN1_UNIT, HIDDEN2_UNIT], HIDDEN1_UNIT+action_dim)
            wf2_a = self.variable([action_dim, HIDDEN2_UNIT], HIDDEN1_UNIT+action_dim)
            wf3 = tf.Variable(tf.random_uniform([HIDDEN2_UNIT, 1],-3e-3,3e-3))
            b1 = self.variable([HIDDEN1_UNIT], state_size)
            b2 = self.variable([HIDDEN2_UNIT], HIDDEN1_UNIT+action_dim)
            b3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

            fc1 = tf.nn.relu(tf.matmul(input_state, wf1) + b1)
            fc2 = tf.nn.relu(tf.matmul(fc1, wf2_s) + tf.matmul(input_action, wf2_a) + b2)
            logits = tf.identity(tf.matmul(fc2, wf3) + b3)

            #params = [wf1, bf1, wf2_s, bf2_s, wf2_a, bf2_a, wf2, bf2, wlogits, blogits]
            params = [wf1, b1, wf2_s, wf2_a, b2, wf3, b3]
            
            return logits, input_action, input_state, params

    def create_target_network(self, state_size, action_size, net):
        input_state = tf.placeholder(tf.float32, shape=[None, state_size])
        ## will come from action network
        input_action = tf.placeholder(tf.float32, shape=[None, action_size])

        ema = tf.train.ExponentialMovingAverage(decay=1-self.TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        #fc1 = tf.nn.relu(tf.add(tf.matmul(input_state, target_net[0]), target_net[1]))
        #fc2_s = tf.add(tf.matmul(fc1, target_net[2]), target_net[3])
        #fc2_a = tf.add(tf.matmul(input_action, target_net[4]), target_net[5])
        #fc2_sum = tf.add(fc2_s, fc2_a)
        #fc2 = tf.nn.relu(tf.add(tf.matmul(fc2_sum, target_net[6]), target_net[7]))
        #logits = tf.add(tf.matmul(fc2, target_net[8]), target_net[9])

        fc1 = tf.nn.relu(tf.matmul(input_state, target_net[0]) + target_net[1])
        fc2 = tf.nn.relu(tf.matmul(fc1, target_net[2]) + tf.matmul(input_action, target_net[3]) + target_net[4])
        logits = tf.identity(tf.matmul(fc2, target_net[5]) + target_net[6])

        return input_state, input_action, logits, target_update
 
    def predict(self, state_and_action):
        return self.sess.run(self.output, feed_dict={self.state: state_and_action[0], self.action: state_and_action[1]})

    def target_predict(self, state_and_action):
        return self.sess.run(self.target_output, feed_dict={self.target_state: state_and_action[0], self.target_action: state_and_action[1] })

    def target_train(self):
        self.sess.run(self.target_update)
        #self.sess.run(self.copy_op)

    def train(self, state_and_action, y):
        return self.sess.run([self.optimize, self.losses], feed_dict={self.state: state_and_action[0], self.action: state_and_action[1], self.y: y})

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={self.state: states, self.action: actions})[0]

    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
