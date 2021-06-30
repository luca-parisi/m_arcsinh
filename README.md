# m-arcsinh in scikit-learn
## An Efficient and Reliable Kernel and Activation Function for Support Vector Machine (SVM) and Multi-Layer Perceptron (MLP)


The modified 'arcsinh' or **'m_arcsinh'** is a Python custom kernel and activation function available for the Support Vector Machine (SVM) implementation for classification ['SVC'](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and Multi-Layer Perceptron (MLP) or ['MLPClassifier'](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) classes in scikit-learn for Machine Learning-based classification. It is distributed under the [CC BY 4.0 license](http://creativecommons.org/licenses/by/4.0/).

Details on this function, implementation and validation against gold standard kernel and activation functions for SVM and MLP respectively are available at the following: **[Parisi, L., 2020](https://arxiv.org/abs/2009.07530)**. 


### Dependencies

As it is compatible with scikit-learn, please note the [dependencies of scikit-learn](https://github.com/scikit-learn/scikit-learn) to be able to use the 'm-arcsinh' function in the 'SVC' and 'MLPClassifier' classes.


### Usage

You can use the m-arcsinh function as a custom:

* [kernel function](https://github.com/luca-parisi/m_arcsinh_scikit_learn/blob/master/svc_m_arcsinh.py) in the 'SVC' class in scikit learn as per the following two steps:

    1. defining the kernel function 'm_arcsinh' as per the ['svc_m_arcsinh.py' file in this repository](https://github.com/luca-parisi/m_arcsinh_scikit_learn/blob/master/svc_m_arcsinh.py) or as follows: 
    
       ```python
        import numpy as np
        
        def m_arcsinh(data, Y):

            return np.dot((1/3*np.arcsinh(data))*(1/4*np.sqrt(np.abs(data))), (1/3*np.arcsinh(Y.T))*(1/4*np.sqrt(np.abs(Y.T))))
       ```
       
    2. after importing the relevant 'svm' class from scikit-learn:  
        
        ```python
        from sklearn import svm 
        classifier = svm.SVC(kernel=m_arcsinh, gamma=0.001, random_state=13, class_weight='balanced')
        ```
        
* [activation function](https://github.com/luca-parisi/m_arcsinh_scikit_learn/blob/master/mlpclassifier_m_arcsinh.py) in the 'MLPClassifier' class in scikit learn as per the following two steps:

    1. updating the 'base.py' file under your local installation of scikit-learn (sklearn/neural_network/_base.py), similarly to this [commit](https://github.com/scikit-learn/scikit-learn/pull/18419/commits/3e1141dc3448615018888e8da07622452b092f4f), including the m-arcsinh in the 'ACTIVATIONS' dictionary
    2. after importing the relevant 'MLPClassifier' class from scikit-learn, you can use the 'm_arcsinh' as any other activation functions within it:
    
    ```python
       from sklearn.neural_network import MLPClassifier
       classifier =  MLPClassifier(activation='m_arcsinh', random_state=1, max_iter=300)
     ```

### Citation request

If you are using this function, please cite the papers by:
* **[Parisi, L., 2020](https://arxiv.org/abs/2009.07530)**.
* **[Parisi, L. et al., 2021](https://www.naun.org/main/NAUN/mcs/2021/a142002-007(2021).pdf)**.
