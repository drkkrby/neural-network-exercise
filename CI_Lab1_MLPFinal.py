import numpy as np
import matplotlib.pyplot as plt

# Paths to the different files
features_file = './data/features.txt'
targets_file = './data/targets.txt'
prediction_file = './data/unknown.txt'

# Method to calculate sigmoid
def sigmoid(z):
    return 1/(1 + np.exp(-z))


def ann_assignment(num_hidden_layers, hidden_layers_len, output_layer_len, input_layer_len,
                   alpha, epochs):

    # Create array of random numbers for weights and thresholds
    weights_i_j = np.random.uniform(-2.4 / input_layer_len, 2.4 / input_layer_len, (input_layer_len, hidden_layers_len))
    weights_j_k = np.random.uniform(-2.4 / hidden_layers_len, 2.4 / hidden_layers_len,
                                    (hidden_layers_len, output_layer_len))

    thresholds_j = np.random.uniform(-2.4 / input_layer_len, 2.4 / input_layer_len, (num_hidden_layers, hidden_layers_len))
    thresholds_k = np.random.uniform(-2.4 / hidden_layers_len, 2.4 / hidden_layers_len, (1, output_layer_len))

    # load features and targets
    features = np.loadtxt(features_file, delimiter=",")
    targets_loaded = np.loadtxt(targets_file, delimiter=",")
    x_pred = np.loadtxt(prediction_file, delimiter=",")

    # format targets to be same shape as features
    targets = np.zeros((7854, 7))
    index = 0
    for target in targets_loaded:
        targets[index, int(target) - 1] = 1
        index += 1

    # Randomly split data into training and test set
    # training_fraction = 0.65
    # choice = np.random.choice(range(features.shape[0]), size=(round(len(features) * training_fraction),), replace=False)
    # ind = np.zeros(features.shape[0], dtype=bool)
    # ind[choice] = True
    # rest = ~ind
    # :5105 for train, 5105:6675 for validation, 6675: for testing

    # Split data into training, validation, and test sets
    train_features = features[:5105]
    test_features = features[6675:]
    train_targets = targets[:5105]
    test_targets = targets[6675:]
    val_features = features[5105:6675]
    val_targets = targets[5105:6675]

    # create ann with parameters
    accuracy = create_ann(num_hidden_layers, hidden_layers_len, output_layer_len, input_layer_len, weights_i_j, weights_j_k,
               thresholds_j, thresholds_k, train_features, train_targets, alpha, epochs, test_features, test_targets, val_features, val_targets, x_pred)
    return accuracy

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
          all_thresh_derivatives, alpha, val_features, val_targets):

    epoch = 0
    epoch_list = list()
    error_list = []
    acc_train_list = list()
    acc_val_list = list()
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

        # Count accuracy on validation set
        count_errors = 0
        count_correct = 0
        for i in range(len(val_features)):
            x = val_features[i]
            output = forward_propagate(x, all_weights, all_thresholds, all_activations)
            # If prediction is not equal to expected then error count goes up by one, if it is equal correct count goes up by 1
            if (np.argmax(output) != np.argmax(val_targets[i])):
                count_errors += 1.0
            else:
                count_correct += 1.0
        # Make it into a percentage and add it to list for graphing purposes
        acc_val = (100 * count_correct) / (count_errors + count_correct)
        acc_val_list.append(acc_val)

        # Same procedure but for accuracy on the training set
        count_errors1 = 0
        count_correct1 = 0
        for i in range(len(features)):
            x = features[i]
            output = forward_propagate(x, all_weights, all_thresholds, all_activations)
            if (np.argmax(output) != np.argmax(targets[i])):
                count_errors1 += 1.0
            else:
                count_correct1 += 1.0

        acc_train = (100 * count_correct1) / (count_errors1 + count_correct1)
        acc_train_list.append(acc_train)
        epoch_list.append(epoch)

        print("Epoch: {}, SSE: {}, Accuracy on validation set: {}%, Accuracy on training set: {}".format(epoch, sum_errors, acc_val, acc_train))
        error_list.append(sum_errors)

    print("DONE TRAINING")

    # Plot the accuracy on training and validation set per epoch
    plt.plot(epoch_list, acc_train_list, label="Accuracy on Training Set")
    plt.plot(epoch_list, acc_val_list, label="Accuracy on Validation Set")
    plt.legend(bbox_to_anchor=(0.35, 0.5), loc='upper left')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.show()

    return error_list


def create_ann(num_hidden_layers, hidden_layers_len, output_layer_len, input_layer_len, weights_i_j, weights_j_k,
               thresholds_j, thresholds_k, features, targets, alpha, epochs, test_features, test_targets, val_features, val_targets, x_pred):

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
    train(epochs, features, targets, all_weights, all_thresholds, all_activations,
          all_derivatives, all_thresh_derivatives, alpha, val_features, val_targets)

    # Calculate error against test set
    count_errors = 0
    count_correct = 0
    for i in range(len(test_features)):
        x = test_features[i]
        output = forward_propagate(x, all_weights, all_thresholds, all_activations)
        # If prediction is not equal to expected then error count goes up by one, if it is equal correct count goes up by 1
        if (np.argmax(output) != np.argmax(test_targets[i])):
            count_errors += 1.0
        else:
            count_correct += 1.0
    accuracy = (100 * count_correct)/(count_errors + count_correct)
    print("Accuracy on test set: {}%".format(accuracy))

    # Create a confusion matrix, initially all zeros
    confusion_matrix = np.zeros((7, 7))
    # For each input of the test set
    for i in range(len(test_features)):
        x = test_features[i]
        output = np.argmax(forward_propagate(x, all_weights, all_thresholds, all_activations))
        actual = np.argmax(test_targets[i])
        # Increment number in the correct cell
        confusion_matrix[output, actual] += 1
    print(confusion_matrix)

    # Forward propagate predictions and save as comma-separated file
    res = np.zeros(len(x_pred) -1)
    # For each input of unknown file but the last one
    for i in range(len(x_pred) - 1):
        x = x_pred[i]
        # Prediction from our model
        output = np.argmax(forward_propagate(x, all_weights, all_thresholds, all_activations)) + 1
        # Save to array
        res[i] = output
    # Make prediction for the final number
    x_final = x_pred[len(x_pred) -1]
    final_output = int(np.argmax(forward_propagate(x_final, all_weights, all_thresholds, all_activations)) + 1)

    # Write array to .txt file
    with open("Group_52_classes.txt", "w") as txt_file:
        for line in res:
            # All numbers are followed by a comma except for the last number
            txt_file.write(str(int(line)) + ",")
        txt_file.write(str(final_output))

    return accuracy

# Set parameters of ann
num_hidden_layers = 1
hidden_layers_len = 20
output_layer_len = 7
input_layer_len = 10
alpha = 0.01
epochs = 30

accuracy_20 = ann_assignment(num_hidden_layers, hidden_layers_len, output_layer_len, input_layer_len, alpha, epochs)

# Get the accuracy of our network on the test set depending on the number of hidden neurons
accuracy_7 = ann_assignment(num_hidden_layers, 7, output_layer_len, input_layer_len, alpha, epochs)

accuracy_30 = ann_assignment(num_hidden_layers, 30, output_layer_len, input_layer_len, alpha, epochs)

accuracy_10 = ann_assignment(num_hidden_layers, 10, output_layer_len, input_layer_len, alpha, epochs)

accuracy_15 = ann_assignment(num_hidden_layers, 15, output_layer_len, input_layer_len, alpha, epochs)

accuracy_25 = ann_assignment(num_hidden_layers, 25, output_layer_len, input_layer_len, alpha, epochs)

# Plot accuracy vs number of hidden neurons
x = ["7", "10", "15", "20", "25", "30"]
y = [round(accuracy_7 * 100)/100, round(accuracy_10 * 100)/100, round(accuracy_15 * 100)/100, round(accuracy_20 * 100)/100, round(accuracy_25 * 100)/100, round(accuracy_30 * 100)/100]
plt.plot(x, y, 'o', color='black')
plt.ylim(85, 95)
plt.xlim(5, 32)
plt.xlabel("# of Neurons in Hidden Layer")
plt.ylabel("Accuracy on Test Set(%)")
for i, txt in enumerate(y):
    plt.annotate(txt, (x[i], y[i]))
plt.show()

# Calculate the accuracy for the 'optimal' hyperparameters (section 2.5)
accuracy_optimal = ann_assignment(num_hidden_layers, 23, output_layer_len, input_layer_len, 0.1, epochs)
print("Accuracy on test set for optimal hyperparameters: {}%".format(accuracy_optimal))


