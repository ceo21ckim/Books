# ReLU 역전파
from audioop import cross
import numpy as np 
from common.functions import *

class ReLU:
    def __init__(self):
        self.mask = None 
        
    def forward(self, x):
        self.mask = (x <= 0 )
        out = x.copy()
        out[self.mask] = 0 
        
        return out 
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout 
        
        return dx 
    
x = np.array([
    [1.0, -0.5], 
    [-2.0, 3.0]
])

mask = (x <= 0)
print(mask)


# Sigmoid 역전파 
class Sigmoid:
    def __init__(self):
        self.out = None 
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        return out 
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out 
        
        return dx 
    
# Affine / Softmax 
x = np.random.rand(2)
w = np.random.rand(2, 3)
b = np.random.rand(3)

x.shape 
w.shape 
b.shape 

x_dot_w = np.array([
    [0, 0, 0], 
    [10, 10, 10]
])

b = np.array([1, 2, 3])

x_dot_w 

x_dot_w + b

dy = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

db = np.sum(dy, axis = 0 )

db  


class Affine:
    def __init__(self, w, b):
        self.w = w 
        self.b = b 
        self.x = None 
        self.dw = None 
        self.db = None 
        
    def forward(self, x):
        self.x = x 
        out = np.dot(x, self.w) + self.b 
        
        return out 
    
    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx 
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None 
        self.y = None 
        self.x = None 
        
    def forward(self, x, t):
        self.t = t 
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss 
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size 
        return dx 
