# The modified arcsinh or 'm-arcsinh' as a custom kernel for the SVC class in scikit-learn

def m_arcsinh(data, Y):
    """Compute a modified arcsinh (m-arcsinh) hyperbolic function 
	in place.

    Further details on this function are available at:  
    https://arxiv.org/abs/2009.07530 
    (Parisi, L., 2020; License: http://creativecommons.org/licenses/by/4.0/). 
    If you are using this function, please cite this paper as follows: 
    arXiv:2009.07530 [cs.LG].

    Parameters
    ----------
    data: {ndarray, dataframe} of shape (n_samples, n_features)
    The data matrix. If as_frame=True, data will be a pandas DataFrame 
    (see https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).

    y: {ndarray, Series} of shape (n_samples,)
    The classification target. If as_frame=True, target will be a pandas Series
    (see https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).
    """
    return np.dot((1/3*np.arcsinh(data))*(1/4*np.sqrt(np.abs(data))), (1/3*np.arcsinh(Y.T))*(1/4*np.sqrt(np.abs(Y.T))))
