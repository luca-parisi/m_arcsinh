"""Additional utility for Machine Learning models in TensorFlow and Keras"""

# The modified arcsinh or 'm-arcsinh' as a custom activation function in TensorFlow (tf_m_arcsinh)
# and Keras (m_arcsinh)

# Author: Luca Parisi <luca.parisi@ieee.org>


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# m-arcsinh as a custom activation function in TensorFlow

'''
# Example of usage of the m-arcsinh in TensorFlow as a custom activation function of a dense layer
dense_layer = tf.layers.dense(inputs=inputs, units=128)
dense_layer_activation = tf_m_arcsinh(dense_layer)
'''

# Defining the m-arcsinh function
def m_arcsinh(x):
  return (1/3*np.arcsinh(x))*(1/4*np.sqrt(np.abs(x)))

# Vectorising the m-arcsinh function  
np_m_arcsinh = np.vectorize(m_arcsinh)

# Defining the derivative of the function m-arcsinh
def d_m_arcsinh(x):
  return (np.sqrt(np.abs(x))/(12*np.sqrt(x**2+1)) + (x*np.arcsinh(x))/(24*np.abs(x)**(3/2)))

# Vectorising the derivative of the m-arcsinh function
np_d_m_arcsinh = np.vectorize(d_m_arcsinh)

# Defining the gradient function of the m-arcsinh
def m_arcsinh_grad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_m_arcsinh(x)
    return grad * n_gr


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
# Generating a unique name to avoid duplicates
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+2))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        
np_m_arcsinh_32 = lambda x: np_m_arcsinh(x).astype(np.float32)


def tf_m_arcsinh(x,name=None):
    with tf.name_scope(name, "m_arcsinh", [x]) as name:
        y = py_func(np_m_arcsinh_32,  # forward pass function
                        [x],
                        [tf.float32],
                        name=name,
                         grad= m_arcsinh_grad)  # The function that overrides gradient
        y[0].set_shape(x.get_shape())  # Specify input rank
        return y[0]

np_d_m_arcsinh_32 = lambda x: np_d_m_arcsinh(x).astype(np.float32)


def tf_d_m_arcsinh(x,name=None):
    with tf.name_scope(name, "d_m_arcsinh", [x]) as name:
        y = tf.py_func(np_d_m_arcsinh_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]


# m-arcsinh as a custom layer in Keras 

'''
# Example of usage of the m-arcsinh as a Keras layer in a sequential model between two dense layers
number_of_classes = 10
model.add(keras.layers.Dense(128))
model.add(m_arcsinh())
model.add(keras.layers.Dense(number_of_classes))
'''

class m_arcsinh(Layer):

    def __init__(self):
        super(m_arcsinh,self).__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs,name=None):
        return tf_m_arcsinh(inputs,name=None)

    def get_config(self):
        base_config = super(m_arcsinh, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
