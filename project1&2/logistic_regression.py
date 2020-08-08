from random import randint
from math import exp
from math import log

x1_train = [];  x2_train = []; y_train = []
n = 100; m = 1000
for i in range(m):
    x1_train.append(randint(-10,10))
    x2_train.append(randint(-10,10))
    if x1_train[-1]+x2_train[-1] > 0:
        y_train.append(1)
    else:
        y_train.append(0)

x1_test = []; x2_test = []; y_test = []
for i in range(n):
    x1_test.append(randint(-10,10))
    x2_test.append(randint(-10,10))
    if x1_test[-1]+x2_test[-1] > 0:
        y_test.append(1)
    else:
        y_test.append(0)

# Train session
# alpha = 0.0001
w1 = 0; w2 = 0; b = 0; alpha = 0.0001
for i in range(100):
    J = 0; dw1 = 0; dw2 = 0; db = 0; correct = 0
    for j in range(m):
        # dw1 = x1 * (a-y)
        # dw2 = x2 * (a-y)
        # db = a-y
        z_i = w1*x1_train[j]+w2*x2_train[j]+b
        a_i = exp(z_i)/(1+exp(z_i))
        # if y_hat > 0.5: set y_hat = 1 (for accuracy evaluation)
        if a_i > 0.5:
            #a_i = 1
            if(y_train[j] == 1):
                correct += 1
        else:
            #a_i = 0
            if(y_train[j] == 0):
                correct += 1
        J += -1*(y_train[j]*log(max(1e-6,a_i)) + (1-y_train[j])*log(max(1e-6,1-a_i)))
        dz_i = a_i - y_train[j]
        dw1 += x1_train[j]*dz_i
        dw2 += x2_train[j]*dz_i
        db += dz_i
    # Average
    J /= m
    dw1 /= m
    dw2 /= m
    db /= m

    w1 -= alpha * dw1
    w2 -= alpha * dw2
    b -= alpha * db
    print("Training Session: {}".format(i+1))
    print("Training Cost: {}".format(J))
    print("Training Accuracy: {}%".format(correct/1000*100))
    print("W1: {0}, W2: {1}, b: {2}".format(w1,w2,b))
    print("====================================================================================")

# Test session
J = 0 ; correct = 0
for i in range(n):
    # dw1 = x1 * (a-y)
    # dw2 = x2 * (a-y)
    # db = a-y
    z_i = w1*x1_test[i]+w2*x2_test[i]+b
    a_i = exp(z_i)/(1+exp(z_i))
    # if y_hat > 0.5: set y_hat = 1 (for accuracy evaluation)
    if a_i > 0.5:
        #a_i = 1
        if(y_test[i] == 1):
            correct += 1
    else:
        #a_i = 0
        if(y_test[i] == 0):
            correct += 1
    J += -1*(y_test[i]*log(max(1e-6,a_i)) + (1-y_test[i])*log(max(1e-6,1-a_i)))

# Average
J /= n
print("Test Cost: {}".format(J))
print("Test Accuracy: {}%".format(correct/100*100))
