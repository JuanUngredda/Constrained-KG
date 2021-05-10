import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import time
import numpy as np
# import os

# import pathlib
#
# checkpoint_dir = pathlib.Path(__file__).parent.absolute()
# checkpoint_dir = str(checkpoint_dir) + "/checkpoints/"

class FC_NN_test_function():
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, max_time=0.003):
        self.batch_size = 250#500
        self.rho = 0.9
        self.epsilon = 1e-07
        self.epochs = 3
        self.samples = 100
        self.num_classes = 10
        self.max_time = max_time
        # self.discrete_idx = discrete_idx
        self.checkpoints = {"x":[], "model":[]}
        (self.master_x_train, self.master_y_train), (self.master_x_test, self.master_y_test) = mnist.load_data()

    def encode_input(self, x, discrete):
        encode = "_"
        for i in range(len(x)):
            if np.isin(i, discrete):
                encode = encode + str(int(x[i])) + "_"
            else:
                val = '%.5f' % (x[i])
                encode = encode + str(val) + "_"
        return encode

    def check_available_models(self, x):
        files = self.checkpoints["x"]
        available = False
        if len(files)==0:
            return available, None

        for f in range(len(files)):
            if np.all(x.reshape(-1)==files[f].reshape(-1)):
                model = self.checkpoints["model"][f]
                available=True
                return available, model
        return available, None

    def f(self, X, true_val = False, verbose=0):
        """
        Load Mnist data, creates a nueral network, create an
        optimizer, put all three together and train. Finally
        perform test and return test error.
        """

        batch_size = self.batch_size
        #self.learning_rate
        rho = self.rho
        epsilon = self.epsilon
        epochs = self.epochs
        num_classes = self.num_classes

        if len(X.shape) == 1:
            X = np.array(X).reshape(1, -1)

        validation_score = np.zeros((X.shape[0], 1))

        # print("self.master_x_train",self.master_x_train.shape)
        # print("self.master_y_train",self.master_x_test.shape)

        x_concat = np.concatenate((self.master_x_train, self.master_x_test))
        y_concat = np.concatenate((self.master_y_train, self.master_y_test))

        for index in range(X.shape[0]):


            train_size = 6.0/7
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_concat, y_concat, train_size=train_size)

            print("index", index, X.shape[0])
            x = X[index]

            available, model = self.check_available_models(x)
            x = x.reshape(1, -1)
            learning_rate = x[:, 0]
            out_val = []
            # Part 1: get the dataset

            x_train = self.x_train.reshape(60000, 784)
            x_test = self.x_test.reshape(10000, 784)
            self.x_test = x_test
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')

            x_train /= 255
            x_test /= 255
            y_train = keras.utils.to_categorical(self.y_train, num_classes)
            y_test = keras.utils.to_categorical(self.y_test, num_classes)

            if available:
                print("available model: ",available)
                self.model = model
            else:
                print("available model: ", available)

                # Part 2: Make model

                model = Sequential()
                model.add(Dense(int(np.power(2, x[:, 4][0])), activation='relu', input_shape=(784,)))
                model.add(Dropout(x[:, 1][0]))
                model.add(Dense(int(np.power(2, x[:, 5][0])), activation='relu'))
                model.add(Dropout(x[:, 2][0]))
                model.add(Dense(int(np.power(2, x[:, 6][0])), activation='relu'))
                model.add(Dropout(x[:, 3][0]))
                model.add(Dense(num_classes, activation='softmax'))
                if verbose == 1: model.summary()

                # Part 3: Make optimizer
                optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=rho, epsilon=epsilon)

                # Part 4: compile
                model.compile(loss='categorical_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])

                # Part 5: train

                model.fit(x_train, y_train,
                                    batch_size=batch_size,
                                    epochs=self.epochs,
                                    verbose=verbose,
                                    validation_data=(x_test, y_test))

                self.checkpoints["x"].append(x.reshape(-1))
                self.checkpoints["model"].append(model)
                self.model = model
            # Part 6: get test measurements
            score = model.evaluate(x_test, y_test, verbose=0)
            out_val.append(score[1])
            print("x", x, "fval", np.mean(out_val))
            validation_score[index, 0] = np.mean(out_val)

        return validation_score  # test classification error

    def c(self, X, true_val=True, verbose=0):

        if len(X.shape) == 1:
            X = np.array(X).reshape(1, -1)

        X_mean_average = np.zeros((X.shape[0], 1))
        for index in range(X.shape[0]):
            x = X[index]
            # print("x",x, "verbose", verbose, "true_val", true_val)
            start = time.time()
            available, model = self.check_available_models(x)
            if available:
                self.model =model
            else:
                self.f(x)
            stop = time.time()
            # print("training time", stop-start)
            if verbose == 1: self.model.summary()
            samples = self.samples
            average_time = np.zeros(samples)

            for i in range(samples):
                start = time.time()
                #np.argmax(self.model.predict(x=self.x_test, batch_size=batch_size), axis=-1)
                #self.model.predict_classes(x=self.x_test, batch_size=batch_size)
                self.model(self.x_test, training=False)
                stop = time.time()
                average_time[i] = stop - start

            #print("average time", average_time)
            # import matplotlib.pyplot as plt
            # plt.hist(average_time, density=True, bins=40)
            # plt.show()
            # plt.hist(np.log(average_time), density=True, bins=40)
            # plt.show()
            # mean_subset = []
            #
            # for _ in range(50):
            #     subsets = np.random.choice(range(len(average_time)), 100, replace=False)
            #     mean_subset.append(np.mean(average_time[subsets]))
            #
            # plt.hist(np.array(mean_subset).reshape(-1), density=True)
            # plt.show()
            # print("subset mean", np.mean(mean_subset))
            # print("mse", np.std(mean_subset) / np.sqrt(len(mean_subset)))
            #
            # print("sum time", np.sum(average_time))
            print("x", x, "cval", np.log(np.mean(average_time))-np.log(self.max_time))
            # print("std", np.std(average_time))
            # print("mse", np.std(average_time) / np.sqrt(len(average_time)))
            X_mean_average[index, 0] = np.mean(average_time)

        return np.log(X_mean_average) - np.log(self.max_time)

import tensorflow as tf
#ALWAYS check cost in
# --- Function to optimize
print("NN TS activate")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

objective_function = FC_NN_test_function()
print("Verbose execution")

x = np.array([[0.1, 0.5, 0.5, 0.5, 5, 5, 5],
              [0.1, 0.5, 0.5, 0.5, 7, 7, 7],
              [0.1, 0.5, 0.5, 0.5, 9, 9, 9]])

start = time.time()
cval = objective_function.c(x)

#test_error = objective_function.f(X = np.array([[0.2,0.2,3,3]]), verbose=1)
                                                               
# print("FINISHED")
                                          
# test_error = objective_function.c(X = np.array([[0.0,0.0,13,13]]), true_val=True, verbose=0)

# print("Test error:", test_error)


