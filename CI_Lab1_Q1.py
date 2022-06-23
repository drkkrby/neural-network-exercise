import numpy as np
import matplotlib.pyplot as plt

features = np.array([
    [0, 0],
    [0, 1],
    [1,0],
    [1,1]
])
# print(features.shape[0], features.shape[1])
# print(features.reshape(2, 4))

Labels_AND = np.array([0, 0, 0, 1])
Labels_OR = np.array([0, 1, 1, 1])
Labels_XOR = np.array([0, 1, 1, 0])

w_and = [0.1, 0.1]
w_or = [0.1, 0.1]
w_xor = [0.1, 0.1]

threshold = 0.5
learning_rate = 0.1
epoch = 20

epoch_and = np.empty(epoch)
epoch_or = np.empty(epoch)
epoch_xor = np.empty(epoch)

for i in range(0, epoch):
    global_delta_and = 0
    global_delta_or = 0
    global_delta_xor = 0
    print("epoch: ", i)
    for j in range(0, features.shape[0]):
        actual_and = Labels_AND[j]
        instance_and = features[j]
        actual_or = Labels_OR[j]
        instance_or = features[j]
        actual_xor = Labels_XOR[j]
        instance_xor = features[j]

        x0_and = instance_and[0]
        x1_and = instance_and[1]
        x0_or = instance_or[0]
        x1_or = instance_or[1]
        x0_xor = instance_xor[0]
        x1_xor = instance_xor[1]

        sum_and = w_and[0] * x0_and + w_and[1] * x1_and
        if(sum_and > threshold):
            fire_and = 1
        else:
            fire_and = 0

        sum_or = w_or[0] * x0_or + w_or[1] * x1_or
        if (sum_or > threshold):
            fire_or = 1
        else:
            fire_or = 0

        sum_xor = w_xor[0] * x0_xor + w_xor[1] * x1_xor
        if (sum_xor > threshold):
            fire_xor = 1
        else:
            fire_xor = 0

        delta_and = actual_and - fire_and
        global_delta_and = global_delta_and + abs(delta_and)
        #print("Prediction AND: ", fire_and, "Actual AND: ", actual_and)


        delta_or = actual_or - fire_or
        global_delta_or = global_delta_or + abs(delta_or)
        #print("Prediction OR: ", fire_or, "Actual OR: ", actual_or)

        delta_xor = actual_xor - fire_xor
        global_delta_xor = global_delta_xor + abs(delta_xor)
        #print("Prediction XOR: ", fire_xor, "Actual XOR: ", actual_xor)

        w_and[0] = w_and[0] + delta_and * learning_rate
        w_and[1] = w_and[1] + delta_and * learning_rate

        w_or[0] = w_or[0] + delta_or * learning_rate
        w_or[1] = w_or[1] + delta_or * learning_rate

        w_xor[0] = w_xor[0] + delta_xor * learning_rate
        w_xor[1] = w_xor[1] + delta_xor * learning_rate
    # print(global_delta_and)
    # print(global_delta_or)
    # print(global_delta_xor)
    # print("-------------------------------")
    epoch_and[i] = global_delta_and
    epoch_or[i] = global_delta_or
    epoch_xor[i] = global_delta_xor
    # if(global_delta_and == 0 & global_delta_or == 0 & global_delta_xor == 0):
    #     break
#print(epoch_xor)
# plt.plot(epoch_and)
# plt.ylabel('some numbers')
# plt.show()
n_epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
plt.plot(n_epoch, epoch_and, label='AND')
plt.plot(n_epoch, epoch_or, label='OR')
plt.plot(n_epoch, epoch_xor, label='XOR')
plt.legend()
plt.ylim(0, 4)
plt.xlim(1, 20)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

