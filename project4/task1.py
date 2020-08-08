import sys
import time
import numpy as np
from random import uniform
import tensorflow as tf
from tensorflow import keras


class Model:
    def __init__(self):
        '''
            Initialize values
            m : 1000 -> train samples
            n : 100 -> test samples
            k : 1000 -> training steps
        '''
        self.m = 1000
        self.n = 100
        self.k = 1000
        # Load data
        try:
            training_set = np.load("./training_set.npz")
            test_set = np.load("./test_set.npz")
            self.x_train = training_set['x_train']
            self.x_train = np.swapaxes(self.x_train,0,1)
            self.y_train = training_set['y_train']
            self.x_test = test_set['x_test']
            self.x_test = np.swapaxes(self.x_test,0,1)
            self.y_test = test_set['y_test']
        # In case of no existing data
        except:
            self._random_generator()


    def table1(self):
        # BCE
        model1 = keras.Sequential([
            keras.layers.Dense(3, input_dim=2, activation='sigmoid'),
            keras.layers.Dense(1, input_dim=3, activation='sigmoid')
        ])
        model1.compile(
            optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9), 
            loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy']
        )
        model1.fit(self.x_test, self.y_test, batch_size=1000, epochs=1000, verbose=0)
        _, train_accuracy = model1.evaluate(self.x_train, self.y_train, verbose=0)
        print("BCE accuracy(train): {}".format(train_accuracy))
        _, test_accuracy = model1.evaluate(self.x_test, self.y_test, verbose=0)
        print("BCE accuracy(test): {}".format(test_accuracy))
       
        # MeanSquaredError
        model2 = keras.Sequential([
            keras.layers.Dense(3, input_dim=2, activation='sigmoid'),
            keras.layers.Dense(1, input_dim=3, activation='sigmoid')
        ])
        model2.compile(
            optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9), 
            loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy']
        )
        model2.fit(self.x_test, self.y_test, batch_size=1000, epochs=1000, verbose=0)
        _, train_accuracy = model2.evaluate(self.x_train, self.y_train, verbose=0)
        print("Mean Squared Error accuracy(train): {}".format(train_accuracy))
        _, test_accuracy = model2.evaluate(self.x_test, self.y_test, verbose=0)
        print("Mean Squared Error accuracy(test): {}".format(test_accuracy))


    def table2(self):
        # SGD
        model1 = keras.Sequential([
            keras.layers.Dense(3, input_dim=2, activation='sigmoid'),
            keras.layers.Dense(1, input_dim=3, activation='sigmoid')
        ])
        model1.compile(
            optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9), 
            loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy']
        )
        # train
        train_start = time.process_time()
        model1.fit(self.x_test, self.y_test, batch_size=1000, epochs=1000, verbose=0)
        _, train_accuracy = model1.evaluate(self.x_train, self.y_train, verbose=0)
        train_end = time.process_time()
        print("SGD accuracy(train): {}".format(train_accuracy))
        print("SGD time(train): {}".format(train_end-train_start))

        # test
        test_start = time.process_time()
        _, test_accuracy = model1.evaluate(self.x_test, self.y_test, verbose=0)
        test_end = time.process_time()
        print("SGD accuracy(test): {}".format(test_accuracy))
        print("SGD time(test): {}".format(test_end-test_start))


        # RMSProp
        model2 = keras.Sequential([
            keras.layers.Dense(3, input_dim=2, activation='sigmoid'),
            keras.layers.Dense(1, input_dim=3, activation='sigmoid')
        ])
        model2.compile(
            optimizer=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None), 
            loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy']
        )
        # train
        train_start = time.process_time()
        model2.fit(self.x_test, self.y_test, batch_size=1000, epochs=1000, verbose=0)
        _, train_accuracy = model2.evaluate(self.x_train, self.y_train, verbose=0)
        train_end = time.process_time()
        print("RMSProp accuracy(train): {}".format(train_accuracy))
        print("RMSProp time(train): {}".format(train_end-train_start))
        
        # test
        test_start = time.process_time()
        _, test_accuracy = model2.evaluate(self.x_test, self.y_test, verbose=0)
        test_end = time.process_time()
        print("RMSProp accuracy(test): {}".format(test_accuracy))
        print("RMSProp time(test): {}".format(test_end-test_start))

        # Adam
        model3 = keras.Sequential([
            keras.layers.Dense(3, input_dim=2, activation='sigmoid'),
            keras.layers.Dense(1, input_dim=3, activation='sigmoid')
        ])
        model3.compile(
            optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None), 
            loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy']
        )
        # train
        train_start = time.process_time()
        model3.fit(self.x_test, self.y_test, batch_size=1000, epochs=1000, verbose=0)
        _, train_accuracy = model3.evaluate(self.x_train, self.y_train, verbose=0)
        train_end = time.process_time()
        print("Adam accuracy(train): {}".format(train_accuracy))
        print("Adam time(train): {}".format(train_end-train_start))

        # test
        test_start = time.process_time()
        _, test_accuracy = model3.evaluate(self.x_test, self.y_test, verbose=0)
        test_end = time.process_time()
        print("Adam accuracy(test): {}".format(test_accuracy))
        print("Adam time(test): {}".format(test_end-test_start))


    def table4(self): 
        '''Run in colab environment'''
        # Batch size 1
        model1 = keras.Sequential([
          keras.layers.Dense(3, input_dim=2, activation='sigmoid'),
          keras.layers.Dense(1, input_dim=3, activation='sigmoid')
        ])
        model1.compile(
          optimizer=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None), 
          loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy']
        )
        # train
        train_start = time.process_time()
        model1.fit(self.x_test, self.y_test, batch_size=1, epochs=1000, verbose=0)
        _, train_accuracy = model1.evaluate(self.x_train, self.y_train, verbose=0)
        train_end = time.process_time()
        print("RMSProp accuracy(train): {}".format(train_accuracy))
        print("RMSProp time(train): {}".format(train_end-train_start))

        # test
        test_start = time.process_time()
        _, test_accuracy = model1.evaluate(self.x_test, self.y_test, verbose=0)
        test_end = time.process_time()
        print("RMSProp accuracy(test): {}".format(test_accuracy))
        print("RMSProp time(test): {}".format(test_end-test_start))

        
        # Batch size 32
        model2 = keras.Sequential([
        keras.layers.Dense(3, input_dim=2, activation='sigmoid'),
        keras.layers.Dense(1, input_dim=3, activation='sigmoid')
        ])
        model2.compile(
        optimizer=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None), 
        loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy']
        )
        # train
        train_start = time.process_time()
        model2.fit(self.x_test, self.y_test, batch_size=32, epochs=1000, verbose=0)
        _, train_accuracy = model2.evaluate(self.x_train, self.y_train, verbose=0)
        train_end = time.process_time()
        print("RMSProp accuracy(train): {}".format(train_accuracy))
        print("RMSProp time(train): {}".format(train_end-train_start))

        # test
        test_start = time.process_time()
        _, test_accuracy = model2.evaluate(self.x_test, self.y_test, verbose=0)
        test_end = time.process_time()
        print("RMSProp accuracy(test): {}".format(test_accuracy))
        print("RMSProp time(test): {}".format(test_end-test_start))


        # Batch size 128
        model3 = keras.Sequential([
          keras.layers.Dense(3, input_dim=2, activation='sigmoid'),
          keras.layers.Dense(1, input_dim=3, activation='sigmoid')
        ])
        model3.compile(
          optimizer=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None), 
          loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy']
        )
        # train
        train_start = time.process_time()
        model3.fit(self.x_test, self.y_test, batch_size=128, epochs=1000, verbose=0)
        _, train_accuracy = model3.evaluate(self.x_train, self.y_train, verbose=0)
        train_end = time.process_time()
        print("RMSProp accuracy(train): {}".format(train_accuracy))
        print("RMSProp time(train): {}".format(train_end-train_start))

        # test
        test_start = time.process_time()
        _, test_accuracy = model3.evaluate(self.x_test, self.y_test, verbose=0)
        test_end = time.process_time()
        print("RMSProp accuracy(test): {}".format(test_accuracy))
        print("RMSProp time(test): {}".format(test_end-test_start))


    # Generate random data
    def _random_generator(self):
        x1_test = []; x2_test = []; y_test = []
        x1_train = [];  x2_train = []; y_train = []
        # Generate training samples
        for i in range(self.m):
            x1_train.append(uniform(-2,2))
            x2_train.append(uniform(-2,2))
            if x1_train[-1]*x1_train[-1] > x2_train[-1]:
                y_train.append(1)
            else:
                y_train.append(0)
        self.x_train = np.array([x1_train,x2_train])
        self.x_train = np.swapaxes(self.x_train,0,1)
        self.y_train = np.array(y_train)

        # Generate test samples
        for i in range(self.n):
            x1_test.append(uniform(-2,2))
            x2_test.append(uniform(-2,2))
            if x1_test[-1]*x1_test[-1] > x2_test[-1]:
                y_test.append(1)
            else:
                y_test.append(0)
        self.x_test = np.array([x1_test,x2_test])
        self.x_test = np.swapaxes(self.x_test,0,1)
        self.y_test = np.array(y_test)


if __name__ == "__main__":
    test = Model()
    test.table1()
    print("------------------------------------------------------------")
    test.table2()
    print("------------------------------------------------------------")
    test.table4()

