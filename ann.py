import numpy as np
import time
import matplotlib.pyplot as plt

features_file = './data/features.txt'
targets_file = './data/targets.txt'


def sigmoid(input):
    return 1/(1+ np.exp(-input))


def ann_assignment(num_hidden_layers = 1, hidden_layers_len = 8, output_layer_len = 7, input_layer_len = 10,
                   alpha = 0.01, epochs = 20):

    # Create array of random numbers for weights and thresholds
    weights_i_j = np.random.uniform(-2.4 / input_layer_len, 2.4 / input_layer_len, (input_layer_len, hidden_layers_len))
    weights_j_k = np.random.uniform(-2.4 / hidden_layers_len, 2.4 / hidden_layers_len,
                                    (hidden_layers_len, output_layer_len))

    thresholds_j = np.random.uniform(-2.4 / input_layer_len, 2.4 / input_layer_len, (num_hidden_layers, hidden_layers_len))
    thresholds_k = np.random.uniform(-2.4 / hidden_layers_len, 2.4 / hidden_layers_len, (1, output_layer_len))

    # load features and targets
    features = np.loadtxt(features_file, delimiter=",")
    targets_loaded = np.loadtxt(targets_file, delimiter=",")

    # format targets to be same shape as features
    targets = np.zeros((7854, 7))
    index = 0
    for target in targets_loaded:
        targets[index, int(target) - 1] = 1
        index += 1

    # Randomly split data into training and test set
    training_fraction = 0.65
    choice = np.random.choice(range(features.shape[0]), size=(round(len(features) * training_fraction),), replace=False)
    ind = np.zeros(features.shape[0], dtype=bool)
    ind[choice] = True
    rest = ~ind

    train_features = features[ind]
    test_features = features[rest]
    train_targets = targets[ind]
    test_targets = targets[rest]

    # create ann with parameters
    create_ann(num_hidden_layers, hidden_layers_len, output_layer_len, input_layer_len, weights_i_j, weights_j_k,
               thresholds_j, thresholds_k, train_features, train_targets, alpha, epochs, test_features, test_targets)


# ANN for example in the book
def ann_xor(num_hidden_layers = 1, hidden_layers_len = 2, output_layer_len = 1, input_layer_len = 2, alpha = 0.1,
            epochs = 1000):

    weights_i_j = np.array([[0.5, 0.9], [0.4, 1.0]])
    weights_j_k = np.array([[-1.2], [1.1]])

    thresholds_j = np.array([[0.8, -0.1]])
    thresholds_k = np.array([[0.3]])

    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    features2 = np.array([[1, 1]])

    targets = np.array([[0], [1], [1], [0]])
    targets2 = np.array([[0]])

    create_ann(num_hidden_layers, hidden_layers_len, output_layer_len, input_layer_len, weights_i_j, weights_j_k,
               thresholds_j, thresholds_k, features, targets, alpha, epochs,  features2, targets2)


# Forward propagate through the ann
def forward_propagate(inputs, weights, thresholds, all_activations):
    activations = inputs
    # Initial activations for input layer is just the input
    all_activations[0] = inputs

    for i, (w, t) in enumerate(zip(weights, thresholds)):
        # Calculate activation for next layer using weights*output - threshold
        net_inputs = np.dot(activations, w) - t[0]
        # Pass through the sigmoid function
        activations = sigmoid(net_inputs)
        # Update activations
        all_activations[i+1] = activations

    return activations


# Backward propagate through the ann
def back_propagate(error, all_derivatives, all_thresh_derivatives, all_activations, all_weights):
    # Starting from the last layer
    for i in reversed(range(len(all_derivatives))):
        activations = all_activations[i+1]

        delta = error * activations * (1- activations) # derivative of sigmoid

        delta_reshaped = delta.reshape(delta.shape[0], -1).T
        current_activations = all_activations[i]
        current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

        all_thresh_derivatives[i] = delta * -1
        all_derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
        error = np.dot(delta, all_weights[i].T)
    return error


def train(epochs, features, targets, all_weights, all_thresholds, all_activations, all_derivatives,
          all_thresh_derivatives, alpha):

    epoch = 0
    error_list = []
    while epoch < epochs:
        sum_errors = 0
        epoch += 1
        # Train each feature
        for (feature, target) in zip(features, targets):
            # Forward propagation
            output_layer = forward_propagate(feature, all_weights, all_thresholds, all_activations)
            error = target - output_layer
            # Backwards propagation
            error = back_propagate(error, all_derivatives, all_thresh_derivatives, all_activations, all_weights)

            # Update weights and thresholds
            for i in range(len(all_weights)):
                all_weights[i] += all_derivatives[i] * alpha
                all_thresholds[i] += all_thresh_derivatives[i] * alpha

            sum_errors += np.sum((target - output_layer) ** 2)

        print(epoch, ": ", sum_errors / len(features))
        error_list.append(sum_errors / len(features))

    print("DONE TRAINING")
    return error_list


def create_ann(num_hidden_layers, hidden_layers_len, output_layer_len, input_layer_len, weights_i_j, weights_j_k,
               thresholds_j, thresholds_k, features, targets, alpha, epochs, test_features, test_targets):

    # Create array for each layer
    hidden_layer = np.zeros((num_hidden_layers, hidden_layers_len))
    output_layer = np.zeros((output_layer_len))
    input_layer = np.zeros((input_layer_len))

    # Create list of all activations for each layer
    all_activations = [input_layer, hidden_layer[0], output_layer]

    # Create list of all weight derivatives for each layer
    all_derivatives = [np.zeros((input_layer_len, hidden_layers_len)), np.zeros((hidden_layers_len, output_layer_len))]

    # Create list of all threshold derivatives for each layer
    all_thresh_derivatives = [np.zeros((hidden_layers_len)), np.zeros((output_layer_len))]

    # Create list of all weights for each layer
    all_weights = [weights_i_j, weights_j_k]

    # Create list of all thresholds for each layer
    all_thresholds = [thresholds_j, thresholds_k]

    # Train ann with train set
    train(epochs, features, targets, all_weights, all_thresholds, all_activations, all_derivatives, all_thresh_derivatives, alpha)

    # Calculate error against test set
    count_errors = 0
    count_correct = 0
    for i in range(len(test_features)):
        x = test_features[i]
        output = forward_propagate(x, all_weights, all_thresholds, all_activations)
        if (np.argmax(output) != np.argmax(test_targets[i])):
            count_errors += 1.0
        else:
            count_correct += 1.0

    print()
    print("Accuracy: ", ((count_correct / (count_errors + count_correct)) * 100), "%")

# Set parameters of ann
num_hidden_layers = 1
hidden_layers_len = 20
output_layer_len = 7
input_layer_len = 10
alpha = 0.01
epochs = 50
ann_assignment(num_hidden_layers, hidden_layers_len, output_layer_len, input_layer_len, alpha, epochs)

