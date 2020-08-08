from random import uniform
import numpy as np

x1_test = []; x2_test = []; y_test = []
x1_train = [];  x2_train = []; y_train = []
m = 1000; n = 100; k = 1000

# Generate training samples
for i in range(m):
    x1_train.append(uniform(-2,2))
    x2_train.append(uniform(-2,2))
    if x1_train[-1]*x1_train[-1] > x2_train[-1]:
        y_train.append(1)
    else:
        y_train.append(0)
x_train = np.array([x1_train,x2_train])
y_train = np.array(y_train)

# Generate test samples
for i in range(n):
    x1_test.append(uniform(-2,2))
    x2_test.append(uniform(-2,2))
    if x1_test[-1]*x1_test[-1] > x2_test[-1]:
        y_test.append(1)
    else:
        y_test.append(0)
x_test = np.array([x1_test,x2_test])
y_test = np.array(y_test)

np.savez("./training_set.npz",x_train=x_train,y_train=y_train)
np.savez("./test_set.npz",x_test=x_test,y_test=y_test)
