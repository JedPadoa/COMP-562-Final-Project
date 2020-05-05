import numpy as np
import neurolab as nl

def gdx_nn(data, num_orig_labels, num_train, labels, num_test, orig_labels):

    print("Training on neural network using gradient descent with momentum backpropagation and adaptive learning rate")

    nn = nl.net.newff([[0, 1] for _ in range(len(data[0]))], [128, 16, num_orig_labels])

    nn.trainf = nl.train.train_gdx

    error_progress = nn.train(data[:num_train, :], labels[:num_train, :], epochs = 10000, show = 1000, goal = 0.001)

    pred_test = nn.sim(data[num_train:, :])
    for i in range(num_test):
        print('\n Original: ', orig_labels[np.argmax(labels[i])])
        print('Predicted: ', orig_labels[np.argmax(pred_test[i])])