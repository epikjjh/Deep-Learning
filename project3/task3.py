from random import uniform
import time
import numpy as np
import pandas as pd
import sys

class LogisticRegression:
    def __init__(self, w1, w2, b1, b2, alpha=0.001):
        '''
            Initialize values
            m : 1000
            n : 100
            k : 1000
        '''
        self.m = 1000
        self.n = 100
        self.k = 1000
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2
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
        ret_str += "Weight: {} {}\n".format(self.w1,self.w2)
        ret_str += "Bias: {} {}\n".format(self.b1,self.b2)
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
        self.Z1 = np.dot(self.w1,sample)+self.b1
        self.Z1 = np.clip(self.Z1,-1e2,1e2)
        self.A1 = 1/(1+np.exp(-1.0*self.Z1))
        self.Z2 = np.dot(self.w2,self.A1)+self.b2
        self.Z2 = np.clip(self.Z2,-1e2,1e2)
        self.A2 = 1/(1+np.exp(-self.Z2))


    # Execute backward propagation
    def _backward_prop(self, x_sample, y_sample):
        dZ2 = self.A2 - y_sample
        dw2 = np.matmul(dZ2,np.transpose(self.A1))/self.m
        db2 = np.sum(dZ2, axis=1, keepdims=True)/self.m
        dZ1 = np.dot(np.transpose(dw2),dZ2)*(self.A1*(1-self.A1))
        dw1 = np.matmul(dZ1,np.transpose(x_sample))/self.m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/self.m
        self.w2 -= self.alpha * dw2
        self.b2 -= self.alpha * db2
        self.w1 -= self.alpha * dw1
        self.b1 -= self.alpha * db1


    # Execute whole training session
    def training_session(self):
        train_start = time.process_time()
        for i in range(self.k):
            self._forward_prop(self.x_train)
            self._backward_prop(self.x_train,self.y_train)
        train_end = time.process_time()

        ret_time = train_end-train_start
        acc = self._accuracy_calculator(self.A2, False)

        return ret_time,acc


    # Execute whole test session
    def test_session(self):
        test_start = time.process_time()
        self._forward_prop(self.x_test)
        test_end = time.process_time()

        ret_time = test_end-test_start
        acc = self._accuracy_calculator(self.A2, True)

        return ret_time,acc
        

# Generate distribution of data: initial weight/bias, learning rate, accuracy
def generate_data():
    weight1_list=[]; weight2_list=[]; bias1_list=[]; bias2_list=[]
    rate_list=[]; accuracy_list=[]

    for i in range(10000):
        w1 = np.random.uniform(low=-10.0,high=10.0,size=(3,2))
        w2 = np.random.uniform(low=-10.0,high=10.0,size=(1,3))
        b1 = np.random.uniform(low=-10.0,high=10.0,size=(3,1))
        b2 = np.random.uniform(low=-10.0,high=10.0,size=(1,1))
        alpha = uniform(0,10)

        regression = LogisticRegression(w1.copy(),w2.copy(),b1.copy(),b2.copy(),alpha)
        regression.training_session()
        _,acc = regression.test_session()

        weight1_list.append(w1)
        weight2_list.append(w2)
        bias1_list.append(b1)
        bias2_list.append(b2)
        rate_list.append(alpha)
        accuracy_list.append(acc)

    raw_data = {'weight 1': weight1_list,
                'weight 2': weight2_list,
                'bias1': bias1_list,
                'bias2': bias2_list,
                'learning rate': rate_list,
                'accuracy': accuracy_list}
    df = pd.DataFrame(raw_data)
    df.sort_values(by=['accuracy'],axis=0,ascending=False,inplace=True)
    df = df[:100]
    df.to_csv("task3.csv", mode='w') 


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
        Learning rate: 0.9 - 1.0
        Initial bias: [-6.0 - -5.0, -2.0 - -1.0, -1.0 - 0.0], [-4.0 - -3.0]
        Initial weight: [-5.0 - -4.0, -2.0 - -1.0, -6.0 - -5.0, -0.5 - 0.5, 6.0 - 7.0, -8.0 - -7.0], 
                        [3.0 - 4.0, 3.0 - 4.0, 3.0 - 4.0]
    '''
    train_time_list = []; train_acc_list = []
    test_time_list = []; test_acc_list = []
    # Calculate mean value(100 iteration) 
    for i in range(100):
        w1_1 = np.random.uniform(low=-5.0,high=-4.0,size=(1,1))
        w1_2 = np.random.uniform(low=-2.0,high=-1.0,size=(1,1))
        w1_3 = np.random.uniform(low=-6.0,high=-5.0,size=(1,1))
        w1_4 = np.random.uniform(low=-0.5,high=0.5,size=(1,1))
        w1_5 = np.random.uniform(low=6.0,high=7.0,size=(1,1))
        w1_6 = np.random.uniform(low=-8.0,high=-7.0,size=(1,1))
        w1 = np.array([np.append(w1_1,w1_2),np.append(w1_3,w1_4),np.append(w1_5,w1_6)])
        w2_1 = np.random.uniform(low=3.0,high=4.0,size=(1,1))
        w2_2 = np.random.uniform(low=3.0,high=4.0,size=(1,1))
        w2_3 = np.random.uniform(low=3.0,high=4.0,size=(1,1))
        w2 = np.array([np.append(w2_1,np.append(w2_1,w2_2))])
        b1_1 = np.random.uniform(low=-6.0,high=-5.0,size=(1,1))
        b1_2 = np.random.uniform(low=-2.0,high=-1.0,size=(1,1))
        b1_3 = np.random.uniform(low=-1.0,high=0.0,size=(1,1))
        b1 = np.array([b1_1,b1_2,b1_3]).reshape(3,1)
        b2 = np.random.uniform(low=-4.0,high=-3.0,size=(1,1))
        alpha = uniform(0.9,1.0)

        regression = LogisticRegression(w1,w2,b1,b2,alpha)
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
