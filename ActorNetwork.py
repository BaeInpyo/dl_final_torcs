import tensorflow as tf
import numpy as np

HIDDEN1_UNIT = 300
HIDDEN2_UNIT = 600

class ActorNetwork(object):
    def __init__(self, sess, stat_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        #Now create the model
        self.output, self.weights, self.input = \
                self.create_actor_network('pred', state_size, action_size)
        self.target_output, _, self.target_input = \
                self.create_actor_network('target', state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(
                self.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

        self.copy_op = []
        self.pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        for pred_var, target_var in zip(self.pred_vars, self.target_vars):
            self.copy_op.append(target_vars.assign(
                TAU*pred_var.value() + (1-TAU)*target_val.value())

    def train(self, input_state, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.input: input_state,
            self.action_gradient: action_grads
        })

    def target_train(self):
        self.sess.run(self.copy_op)

    def create_actor_network(self, name, state_size, action_dim):
        with tf.variable_scope(name):
            V = merge([Steering,Acceleration,Brake],mode='concat')
            model = Model(input=S,output=V)

            input_state = tf.placeholder(tf.float32, shape=[state_size])

            wf1 = tf.get_variable(name='wf1', shape=[state_size, HIDDEN1_UNIT])
            wf2 = tf.get_variable(name='wf2', shape=[HIDDEN1_UNIT, HIDDEN2_UNIT])
            wst = tf.get_variable(name='wst', shape=[HIDDEN2_UNIT, 1])
            wac = tf.get_variable(name='wac', shape=[HIDDEN2_UNIT, 1])
            wbr = tf.get_variable(name='wbr', shape=[HIDDEN2_UNIT, 1])

            bf1 = tf.constant(0.0, shape=[HIDDEN1_UNIT])
            bf2 = tf.constant(0.0, shape=[HIDDEN2_UNIT])
            bst = tf.constant(0.0, shape=[1])
            bac = tf.constant(0.0, shape=[1])
            bbr = tf.constant(0.0, shape=[1])


            fc1 = tf.nn.relu(tf.add(tf.matmul(input_state, wf1), bf1))
            fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, wf2), bf2))

            steering = tf.nn.tanh(tf.add(tf.matmul(fc2, wst), bst))
            accel = tf.nn.sigmoid(tf.add(tf.matmul(fc2, wac), bac))
            brake = tf.nn.sigmoid(tf.add(tf.matmul(fc2, wbr), bbr))

            logits = tf.concat([steering, accel, brake], 1)
            
            params = [wf1, bf1, wf2, bf2, wst, bst, wac, bac, wbr, bbr]
            
            return logits, params, input_state


    def predict(self, input_state):
        return self.sess.run(self.output, feed_dict={self.input: input_state})

    def target_predict(self, input_state):
        return self.sess.run(self.target_output, feed_dict={self.target_input: input_state})
