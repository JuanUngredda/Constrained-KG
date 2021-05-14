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
        self.batch_size = 500
        self.rho = 0.9
        self.epsilon = 1e-07
        self.epochs = 3
        self.samples = 500
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

    def cube_to_hypers(self,x):
        # the GP is over the unit cube [0, 1]^7,
        # dropout rates are same [0, 1]^3
        # other hypers vary on exponential scales

        # learning rate [0, 1] -> [0.0001, 0.01], exponentially
        # batch_size [0, 1] -> [16, 256], exponentially
        # beta_2 [0, 1] -> [0.7, 0.99], exponentially
        # beta_2 [0, 1] -> [0.9, 0.999], exponentially
        hypers = x.copy()
        hypers[0] = 0.0001 * np.exp(x[0] * np.log(100)) #learning rate
        hypers[1] = 0.8 * x[1] #drop out
        hypers[2] = 0.8 * x[2] #drop out
        hypers[3] = x[3]#number of neu #1 - 0.01 * np.exp((1 - x[4]) * np.log(30))
        hypers[4] = x[4]#number of neur #1 - 0.0001 * np.exp((1 - x[5]) * np.log(100))
        hypers[5] = 1 - 0.01 * np.exp((1-x[5]) * np.log(30)) #beta1
        hypers[6] = 1 - 0.0001 * np.exp((1-x[6]) * np.log(100)) #beta2
        return hypers

    @staticmethod
    def hypers_to_cube(self, hypers):
        # the GP is over the unit cube [0, 1]^7,
        # dropout rates are same [0, 1]^3
        # other hypers vary on exponential scales
        x = copy(hypers)
        x[0] = 1.25 * hypers[0]
        x[1] = 1.25 * hypers[1]
        x[2] = 1.25 * hypers[2]
        x[3] = np.log(hypers[3] * 10000.0) / np.log(100.0)
        x[4] = 1 - np.log((1 - hypers[4]) * 100.0) / np.log(30.0)
        x[5] = 1 - np.log((1 - hypers[5]) * 10000.0) / np.log(100.0)
        x[6] = np.log(hypers[6] / 16.0) / np.log(16)
        return x

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
            x_val_stand = X[index]
            print("x_val_stand",x_val_stand)
            x = self.cube_to_hypers(x_val_stand)

            available, model = self.check_available_models(x)
            x = x.reshape(1, -1)

            learning_rate = x[:, 0][0]
            beta_1 = x[:,5][0]
            beta_2 = x[:,6][0]
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

            if true_val:
                num_replications = 10
            else:
                num_replications = 1

            for _ in range(num_replications):
                if available:
                    print("available model: ",available)
                    self.model = model
                else:
                    print("available model: ", available)

                    # Part 2: Make model

                    model = Sequential()
                    model.add(Dense(int(np.power(2, x[:, 3][0])), activation='relu', input_shape=(784,)))
                    model.add(Dropout(x[:, 1][0]))
                    model.add(Dense(int(np.power(2, x[:, 4][0])), activation='relu'))
                    model.add(Dropout(x[:, 2][0]))
                    model.add(Dense(num_classes, activation='softmax'))
                    if verbose == 1: model.summary()

                    # Part 3: Make optimizer
                    print(beta_1,beta_2)
                    adam = keras.optimizers.Adam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2
                    )

                    # Part 4: compile
                    model.compile(loss='categorical_crossentropy',
                                  optimizer=adam,
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
        train_size = 6.0 / 14
        # x_concat = np.concatenate((self.master_x_train, self.master_x_test))
        # y_concat = np.concatenate((self.master_y_train, self.master_y_test))
        # x_train, x_test, y_train, y_test = train_test_split(x_concat, y_concat,test_size=125)
        # x_test = x_test.reshape(125, 784)
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
                self.model(self.x_test) #.predict(x_test.astype('float32'))#, training=False)
                stop = time.time()
                average_time[i] = stop - start

            print("x", x, "cval", np.log(np.mean(average_time))-np.log(self.max_time))
            # print("std", np.std(average_time))
            # print("mse", np.std(average_time) / np.sqrt(len(average_time)))
            X_mean_average[index, 0] = np.mean(average_time)

        return np.log(X_mean_average) - np.log(self.max_time)

# import tensorflow as tf
# #ALWAYS check cost in
# # --- Function to optimize
# print("NN TS activate")
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#
# objective_function = FC_NN_test_function()
# print("Verbose execution")
#
# x = np.array([[0.1, 0.5, 0.5, 0.5, 5, 5, 5],
#               [0.1, 0.5, 0.5, 0.5, 7, 7, 7],
#               [0.1, 0.5, 0.5, 0.5, 9, 9, 9]])
#
# start = time.time()
# cval = objective_function.c(x)

#test_error = objective_function.f(X = np.array([[0.2,0.2,3,3]]), verbose=1)
                                                               
# print("FINISHED")
                                          
# test_error = objective_function.c(X = np.array([[0.0,0.0,13,13]]), true_val=True, verbose=0)

# print("Test error:", test_error)


