from random import uniform
import time
import numpy as np
import pandas as pd
import sys

class LogisticRegression:
    def __init__(self, w, b, alpha=0.001):
        '''
            Initialize values
            m : 1000
            n : 100
            k : 1000
        '''
        self.m = 1000
        self.n = 100
        self.k = 1000
        self.w = w
        self.b = b
        self.alpha = alpha
        self.calculator = np.vectorize(lambda x: 1 if x>=0.5 else 0)
        # Load data
        try:
            training_set = np.load("./training_set.npz")
            test_set = np.load("./test_set.npz")
            self.x_train = training_set['x_train']
            self.y_train = training_set['y_train']
            self.x_test = test_set['x_test']
            self.y_test = test_set['y_test']
        # In case of no existing data
        except:
            self._random_generator()


    # Representation: for easy comparison
    def __repr__(self):
        training_time,training_acc = self.training_session()
        test_time,test_acc = self.test_session()
        
        ret_str = "Learning Rate: {:.5f}\n".format(self.alpha)
        ret_str += "Weight: {}\n".format(self.w)
        ret_str += "Bias: {}\n".format(self.b)
        ret_str += "Training Time: {:.2f}s\n".format(training_time)
        ret_str += "Training Accuracy: {:.2f}%\n".format(training_acc/self.m*100)
        ret_str += "Test Time: {:.5f}s\n".format(test_time)
        ret_str += "Test Accuracy: {:.2f}%".format(test_acc/self.n*100)
            
        return ret_str

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
        self.y_test = np.array(y_test)


    # Calculate accuracy in given set
    def _accuracy_calculator(self, tar, is_test):
        tar = self.calculator(tar)
        acc = np.sum(self.y_test==tar) if is_test else np.sum(self.y_train==tar)

        return acc


    # Execute forward propagation
    def _forward_prop(self, sample):
        self.Z = np.dot(self.w,sample)+self.b
        self.A = np.exp(self.Z)/(1+np.exp(self.Z))


    # Execute backward propagation
    def _backward_prop(self, x_sample, y_sample):
        dZ = self.A - y_sample
        dw = np.matmul(dZ,np.transpose(x_sample))/self.m
        db = np.sum(dZ)/self.m
        self.w -= self.alpha * dw
        self.b -= self.alpha * db


    # Execute whole training session
    def training_session(self):
        train_start = time.process_time()
        for i in range(self.k):
            self._forward_prop(self.x_train)
            self._backward_prop(self.x_train,self.y_train)
        train_end = time.process_time()

        ret_time = train_end-train_start
        acc = self._accuracy_calculator(self.A, False)

        return ret_time,acc


    # Execute whole test session
    def test_session(self):
        test_start = time.process_time()
        self._forward_prop(self.x_test)
        test_end = time.process_time()

        ret_time = test_end-test_start
        acc = self._accuracy_calculator(self.A, True)

        return ret_time,acc
        

# Generate distribution of data: initial weight/bias, learning rate, accuracy
def generate_data():
    weight_list=[]; bias_list=[]; rate_list=[]; accuracy_list=[]

    for i in range(10000):
        w = np.random.uniform(low=0.0,high=10.0,size=(1,2))
        b = np.random.uniform(low=0.0,high=10.0,size=(1,1))
        alpha = uniform(0,10)
        regression = LogisticRegression(w.copy(),b.copy(),alpha)
        regression.training_session()
        _,acc = regression.test_session()

        weight_list.append(w)
        bias_list.append(b)
        rate_list.append(alpha)
        accuracy_list.append(acc)

    raw_data = {'weight': weight_list,
                'bias': bias_list,
                'learning rate': rate_list,
                'accuracy': accuracy_list}
    df = pd.DataFrame(raw_data)
    df.sort_values(by=['accuracy'],axis=0,ascending=False,inplace=True)
    df = df[:100]
    df.to_csv("task1.csv", mode='w')


if __name__ == "__main__":
    '''
        First, we need to know proper range of initial weight/bias, learning rate.
        Using generate_data function, we can examine distribution data.
        After examination, we can choose proper range
    '''
    #generate_data()
    #sys.exit()
    '''
        <Proper range>
        Learning rate: 0.1 - 0.2
        Initial bias: 5.0 - 7.5
        Initial weight: [2.5 - 3.5], [1.25 - 2.25]
    '''
    train_time_list = []; train_acc_list = []
    test_time_list = []; test_acc_list = []
    # Calculate mean value(100 iteration) 
    for i in range(100):
        w1 = np.random.uniform(low=2.5,high=3.5,size=(1,1))
        w2 = np.random.uniform(low=1.25,high=2.25,size=(1,1))
        w = np.array([np.append(w1,w2)])
        b = np.random.uniform(low=5.0,high=7.5,size=(1,1))
        alpha = uniform(0.1,0.2)

        regression = LogisticRegression(w,b,alpha)
        train_t,train_acc = regression.training_session()
        test_t,test_acc = regression.test_session()

        train_time_list.append(train_t)
        train_acc_list.append(train_acc/regression.m*100)
        test_time_list.append(test_t)
        test_acc_list.append(test_acc/regression.n*100)
    # Print result
    print("Mean Training time: {:.5f}s".format(np.mean(train_time_list)))
    print("Mean Training accuracy: {:.3f}%".format(np.mean(train_acc_list)))
    print("Mean Test time: {:.5f}s".format(np.mean(test_time_list)))
    print("Mean Test accuracy: {:.3f}%".format(np.mean(test_acc_list)))
