# 4.4 gradient 
import numpy as np 

def function_2(x):
    return x[0]**2 + x[1]**2 


def numerical_gradient(f, x):
    h = 1e-4 
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h)
        x[idx] = tmp_val + h 
        fxh1 = f(x)
        
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val 
        
    return grad 

numerical_gradient(function_2, np.array([3.0, 4.0])) # array([6., 8.])

numerical_gradient(function_2, np.array([0.0, 2.0])) # array([0., 4.])

numerical_gradient(function_2, np.array([3.0, 0.0])) # array([6., 0.])


# 4.4.1 gradient descent 

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x 
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr *grad 
        
    return x 

def function_2(x):
    return x[0]**2 + x[1]**2 

init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100) # array([-6.11110793e-10,  8.14814391e-10])


# 4.4.2 

import sys, os
sys.path.append(os.pardir)
import numpy as np 
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.w = np.random.randn(2, 3) # initialization with normal distribution 
        
    def predict(self, x):
        return np.dot(x, self.w)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss 
    
    
net = simpleNet()
print(net.w)

