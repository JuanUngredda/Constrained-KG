import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import time
import numpy as np

class FC_NN_test_function():
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, max_time=0.003):
        self.batch_size = 500
        self.learning_rate = 0.001
        self.rho = 0.9
        self.epsilon = 1e-07
        self.epochs = 3
        self.samples = 1000
        self.num_classes = 10
        self.max_time = max_time


    def f(self, X, true_val = False, verbose=0):
        """
        Load Mnist data, creates a nueral network, create an
        optimizer, put all three together and train. Finally
        perform test and return test error.
        """
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        rho = self.rho
        epsilon = self.epsilon
        epochs = self.epochs
        num_classes = self.num_classes

        if len(X.shape) == 1:
            X = np.array(X).reshape(1, -1)

        validation_score = np.zeros((X.shape[0], 1))

        for index in range(X.shape[0]):
            print("index", index, X.shape[0])
            x = X[index]
            x = np.array(x).reshape(-1)
            x = x.reshape(1, -1)
            if true_val:
                reps = 1
            else:
                reps = 1
            out_val = []
            for i in range(reps):
                # Part 1: get the dataset
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                x_train = x_train.reshape(60000, 784)
                x_test = x_test.reshape(10000, 784)
                self.x_test = x_test
                x_train = x_train.astype('float32')
                x_test = x_test.astype('float32')
                x_train /= 255
                x_test /= 255
                y_train = keras.utils.to_categorical(y_train, num_classes)
                y_test = keras.utils.to_categorical(y_test, num_classes)

                # Part 2: Make model

                model = Sequential()
                model.add(Dense(int(np.power(2, x[:, 2][0])), activation='relu', input_shape=(784,)))
                model.add(Dropout(x[:, 0][0]))
                model.add(Dense(int(np.power(2, x[:, 3][0])), activation='relu'))
                model.add(Dropout(x[:, 1][0]))
                model.add(Dense(num_classes, activation='softmax'))
                if verbose == 1: model.summary()

                # Part 3: Make optimizer
                optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=rho, epsilon=epsilon)

                # Part 4: compile
                model.compile(loss='categorical_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])

                # Part 5: train
                # print("batch_size", batch_size, "epochs", epochs)
                history = model.fit(x_train, y_train,
                                    batch_size=batch_size,
                                    epochs=self.epochs,
                                    verbose=verbose,
                                    validation_data=(x_test, y_test))

                self.model = model
                # Part 6: get test measurements
                score = model.evaluate(x_test, y_test, verbose=1)
                out_val.append(score[1])
            print("out_val",out_val)
            print("np.mean(out_val)",np.mean(out_val))
            print("np.std", np.std(out_val))
            print("mse",np.std(out_val)/len(out_val) )
            validation_score[index, 0] = np.mean(out_val)

        return validation_score  # test classification error

    def c(self, X, true_val=True, verbose=0):

        batch_size = self.batch_size
        learning_rate = self.learning_rate
        rho = self.rho
        epsilon = self.epsilon
        epochs = self.epochs

        if len(X.shape) == 1:
            X = np.array(X).reshape(1, -1)

        X_mean_average = np.zeros((X.shape[0], 1))
        for index in range(X.shape[0]):

            x = X[index]
            print("x",x, "verbose", verbose, "true_val", true_val)
            self.f(x,true_val=true_val , verbose=verbose)
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
            average_time = average_time[1:]
            #print("average time", average_time)
            print("np.mean(average_time)", np.mean(average_time))
            print("std", np.std(average_time))
            print("mse", np.std(average_time) / np.sqrt(len(average_time)))
            X_mean_average[index, 0] = np.mean(average_time)

        return X_mean_average - self.max_time

objective_function = FC_NN_test_function()
#print("Verbose execution")

#test_error = objective_function.f(X = np.array([[0.2,0.2,3,3]]), verbose=1)
                                                               
# print("FINISHED")
                                          
# test_error = objective_function.c(X = np.array([[0.0,0.0,13,13]]), true_val=True, verbose=0)

# print("Test error:", test_error)


