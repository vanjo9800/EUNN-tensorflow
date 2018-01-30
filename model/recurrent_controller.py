import numpy as np
import tensorflow as tf
from model.dnc.controller import BaseController
from model.EUNN import EUNNCell

class RecurrentController(BaseController):

    def network_vars(self):
        if self.model == "LSTM":
            self.network_cell = tf.nn.rnn_cell.BasicLSTMCell(256)
            self.state = self.network_cell.zero_state(self.batch_size, tf.float32)
        elif self.model == "EUNN":
            self.network_cell = EUNNCell(256,2,False,True)
            #self.state = tf.zeros([1,256])
            self.state = tf.complex(tf.zeros([1,256]),tf.zeros([1,256]))

    def network_op(self, X, state):
        X = tf.convert_to_tensor(X)
        result1, result2 = self.network_cell(X, state)
        return (tf.real(result1),result2)
        #return (result1, result2)

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()
