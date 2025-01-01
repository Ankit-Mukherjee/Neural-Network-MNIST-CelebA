import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    sig=1.0 /(1.0 + np.exp(-z))
    return  sig



def preprocess():
    data = loadmat('mnist_all.mat')
    
    # Initialize lists for data
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    
    # Process training and test data
    for i in range(10):
        train_i = data['train' + str(i)]
        train_data.append(train_i)
        train_label.extend([i] * train_i.shape[0])
        
        test_i = data['test' + str(i)]
        test_data.append(test_i)
        test_label.extend([i] * test_i.shape[0])
    
    # Convert to numpy arrays
    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_data = train_data.astype('float32') / 255.0
    test_data = test_data.astype('float32') / 255.0
    
    # Create validation set
    indices = np.random.permutation(train_data.shape[0])
    validation_size = int(train_data.shape[0] * 0.1)
    
    validation_idx = indices[:validation_size]
    train_idx = indices[validation_size:]
    
    validation_data = train_data[validation_idx, :]
    validation_label = train_label[validation_idx]
    train_data = train_data[train_idx, :]
    train_label = train_label[train_idx]
    
    # Feature selection to remove redundant features
    feature_count = train_data.shape[1]
    min_list = np.amin(train_data, axis=0)
    max_list = np.amax(train_data, axis=0)
    
    redundant_feature_list = [i for i in range(feature_count) if min_list[i] == max_list[i]]
    
    train_data = np.delete(train_data, redundant_feature_list, axis=1)
    validation_data = np.delete(validation_data, redundant_feature_list, axis=1)
    test_data = np.delete(test_data, redundant_feature_list, axis=1)
    
    print('Preprocessing done')
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    training_data_bias = np.concatenate((training_data, np.ones((training_data.shape[0], 1))), axis=1)
    
    # Forward propagation
    a1 = sigmoid(np.dot(training_data_bias, w1.T))  # Hidden layer activation
    a1_bias = np.concatenate((a1, np.ones((a1.shape[0], 1))), axis=1)  # Add bias to hidden layer
    a2 = sigmoid(np.dot(a1_bias, w2.T))  # Output layer activation
    
    # One-hot encode labels
    y = np.zeros((training_data.shape[0], n_class))
    y[range(training_data.shape[0]), training_label.astype(int)] = 1
    
    # Compute error with regularization
    error_term = -np.sum(y * np.log(a2 + 1e-20) + (1 - y) * np.log(1 - a2 + 1e-20)) / training_data.shape[0]
    reg_term = (lambdaval / (2 * training_data.shape[0])) * (np.sum(w1 * w1) + np.sum(w2 * w2))
    obj_val = error_term + reg_term
    
    # Backpropagation
    delta2 = (a2 - y) / training_data.shape[0]  # Output layer error
    
    # Hidden layer error (removing bias term)
    delta1 = (np.dot(delta2, w2)[:, :-1]) * (a1 * (1 - a1))
    
    # Compute gradients with regularization
    grad_w2 = np.dot(delta2.T, a1_bias) + (lambdaval / training_data.shape[0]) * w2
    grad_w1 = np.dot(delta1.T, training_data_bias) + (lambdaval / training_data.shape[0]) * w1
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """Predicts the label of data given the parameters w1, w2 of the Neural Network."""
    # Add bias to input data
    data_bias = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)

    # Forward propagation
    a1 = sigmoid(np.dot(data_bias, w1.T))
    a1_bias = np.concatenate((a1, np.ones((a1.shape[0], 1))), axis=1)
    a2 = sigmoid(np.dot(a1_bias, w2.T))

    # Get predicted labels (adjust axis if needed)
    labels = np.argmax(a2, axis=1)  # Change to axis=0 if required

    return labels


"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0.1

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


