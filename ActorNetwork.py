import tensorflow as tf
import numpy as np
import math

HIDDEN1_UNIT = 300
HIDDEN2_UNIT = 400

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        #Now create the model
        self.output, self.weights, self.input = \
                self.create_actor_network('pred', state_size)
        self.target_input, self.target_output, self.target_update, self.target_net = \
                self.create_target_network(state_size, self.weights)
        #self.target_output, _, self.target_input = \
        #        self.create_actor_network('target', state_size)

        # Gradient
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(
                self.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)

        # Optimizer
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)

        # Initialize variable
        self.sess.run(tf.global_variables_initializer())

        #self.copy_op = []
        #self.pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
        #self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        #for pred_var, target_var in zip(self.pred_vars, self.target_vars):
        #    self.copy_op.append(target_var.assign(
        #        TAU*pred_var.value() + (1-TAU)*target_var.value()))

        # Set target weight
        self.target_train()

    def create_actor_network(self, name, state_size):
        with tf.variable_scope(name):
            input_state = tf.placeholder(tf.float32, shape=[None, state_size])

            #wf1 = tf.get_variable(name='wf1', shape=[state_size, HIDDEN1_UNIT])
            #wf2 = tf.get_variable(name='wf2', shape=[HIDDEN1_UNIT, HIDDEN2_UNIT])
            #wst = tf.get_variable(name='wst', shape=[HIDDEN2_UNIT, 1], initializer=tf.truncated_normal_initializer(stddev=1e-4))
            #wac = tf.get_variable(name='wac', shape=[HIDDEN2_UNIT, 1], initializer=tf.truncated_normal_initializer(stddev=1e-4))
            #wbr = tf.get_variable(name='wbr', shape=[HIDDEN2_UNIT, 1], initializer=tf.truncated_normal_initializer(stddev=1e-4))

            #bf1 = tf.constant(value=0.0, name='bf1', shape=[HIDDEN1_UNIT])
            #bf2 = tf.constant(value=0.0, name='bf2', shape=[HIDDEN2_UNIT])
            #bst = tf.constant(value=0.0, name='bst', shape=[1])
            #bac = tf.constant(value=0.0, name='bac', shape=[1])
            #bbr = tf.constant(value=0.0, name='bbr', shape=[1])

            wf1 = tf.Variable(tf.random_uniform(
                [state_size, HIDDEN1_UNIT],
                -1/math.sqrt(state_size), 1/math.sqrt(state_size)))
            wf2 = tf.Variable(tf.random_uniform(
                [HIDDEN1_UNIT, HIDDEN2_UNIT],
                -1/math.sqrt(HIDDEN1_UNIT), 1/math.sqrt(HIDDEN1_UNIT)))
            wst = tf.Variable(tf.random_uniform([HIDDEN2_UNIT, 1], -1e-4, 1e-4))
            wac = tf.Variable(tf.random_uniform([HIDDEN2_UNIT, 1], -1e-4, 1e-4))
            wbr = tf.Variable(tf.random_uniform([HIDDEN2_UNIT, 1], -1e-4, 1e-4))

            bf1 = tf.Variable(tf.random_uniform(
                [HIDDEN1_UNIT],
                -1/math.sqrt(state_size), 1/math.sqrt(state_size)))
            bf2 = tf.Variable(tf.random_uniform(
                [HIDDEN2_UNIT],
                -1/math.sqrt(HIDDEN1_UNIT), 1/math.sqrt(HIDDEN1_UNIT)))
            bst = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4))
            bac = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4))
            bbr = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4))

            fc1 = tf.nn.relu(tf.add(tf.matmul(input_state, wf1), bf1))
            fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, wf2), bf2))

            steering = tf.nn.tanh(tf.add(tf.matmul(fc2, wst), bst))
            accel = tf.nn.sigmoid(tf.add(tf.matmul(fc2, wac), bac))
            brake = tf.nn.sigmoid(tf.add(tf.matmul(fc2, wbr), bbr))

            logits = tf.concat([steering, accel, brake], 1)
            
            params = [wf1, bf1, wf2, bf2, wst, bst, wac, bac, wbr, bbr]
            #params = [wf1, wf2, wst, wac, wbr]
            
            return logits, params, input_state

    def create_target_network(self, state_size, net):
        input_state = tf.placeholder(tf.float32, shape=[None, state_size])
        ema = tf.train.ExponentialMovingAverage(decay=1-self.TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        fc1 = tf.nn.relu(tf.matmul(input_state, target_net[0]) + target_net[1])
        fc2 = tf.nn.relu(tf.matmul(fc1, target_net[2]) + target_net[3])

        steer = tf.tanh(tf.matmul(fc2, target_net[4]) + target_net[5])
        accel = tf.sigmoid(tf.matmul(fc2, target_net[6]) + target_net[7])
        brake = tf.sigmoid(tf.matmul(fc2, target_net[8]) + target_net[9])

        output_action = tf.concat([steer, accel, brake], 1)
        return input_state, output_action, target_update, target_net

    def predict(self, input_state):
        return self.sess.run(self.output, feed_dict={self.input: input_state})

    def target_predict(self, input_state):
        return self.sess.run(self.target_output, feed_dict={self.target_input: input_state})

    def train(self, input_state, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.input: input_state,
            self.action_gradient: action_grads
        })

    def target_train(self):
        self.sess.run(self.target_update)
        #self.sess.run(self.copy_op)

