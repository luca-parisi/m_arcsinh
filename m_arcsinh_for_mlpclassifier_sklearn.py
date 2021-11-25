"""Additional utility for the 'MLPClassifier' class in the neural network module in scikit-learn"""

# The modified arcsinh or 'm-arcsinh' as a custom activation function

# Author: Luca Parisi <luca.parisi@ieee.org>

import numpy as np


def m_arcsinh(X):
    """Compute a modified arcsinh (m-arcsinh) hyperbolic function 
	in place.
    Further details on this function are available at:  
    https://arxiv.org/abs/2009.07530 
    (Parisi, L., 2020; License: http://creativecommons.org/licenses/by/4.0/). 
    If you are using this function, please cite this paper as follows: 
    arXiv:2009.07530 [cs.LG].
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    return (1/3*np.arcsinh(X))*(1/4*np.sqrt(np.abs(X)))
    
	
def inplace_m_arcsinh_derivative(Z, delta):
    """Apply the derivative of the hyperbolic m-arcsinh function.
    It exploits the fact that the derivative is a relatively 
    simple function of the output value from hyperbolic m-arcsinh.
    Further details on this function are available at:  
    https://arxiv.org/abs/2009.07530 
    (Parisi, L., 2020; License: http://creativecommons.org/licenses/by/4.0/). 
    If you are using this function, please cite this paper as follows: 
    arXiv:2009.07530 [cs.LG].
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the hyperbolic tangent activation
        function during the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified in place.
    """
    delta *= (np.sqrt(np.abs(Z))/(12*np.sqrt(Z**2+1)) + (Z*np.arcsinh(Z))/(24*np.abs(Z)**(3/2)))
