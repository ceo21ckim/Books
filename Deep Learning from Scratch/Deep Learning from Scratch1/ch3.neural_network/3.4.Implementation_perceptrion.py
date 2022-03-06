from re import X
import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

x = np.array([1.0, 0.5]) # 1 by 2
w1 = np.array([
    [0.1, 0.3, 0.5], 
    [0.2, 0.4, 0.6]
    ]) # 2 by 3

b1 = np.array([
    [0.1, 0.2, 0.3]
]) # 1 by 3

x.shape, w1.shape, b1.shape

a1 = np.dot(x, w1) + b1 

z1 = sigmoid(a1) # 1 by 3

w2 = np.array([
    [0.1, 0.4], 
    [0.2, 0.5], 
    [0.3, 0.6]
]) # 3 by 2 

b2 = np.array([
    [0.1, 0.2]
])

a2 = np.dot(z1, w2) + b2 

a2.shape # 1 by 2 

z2 = sigmoid(a2)


def identity_function(x):
    return x 

w3 = np.array([
    [0.1, 0.3], 
    [0.2, 0.4]
])


b3 = np.array([0.1, 0.2])


a3 = np.dot(z2, w3) + b3 

y = identity_function(a3)


## 만든 코드
class perceptron:
    def __init__(self):
        self.w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        self.b1 = np.array([0.1, 0.2, 0.3])
        
        self.w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        self.b2 = np.array([0.1, 0.2])
        
        self.w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.b3 = np.array([0.1, 0.2])
        
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def identity_function(self, x):
        return x 
        
    def fit(self, x):
        w1, w2, w3 = self.w1, self.w2, self.w3 
        b1, b2, b3 = self.b1, self.b2, self.b3 
        
        layer1 = np.dot(x, w1) + b1 
        layer1 = self.sigmoid(layer1)
        
        layer2 = np.dot(layer1, w2) + b2 
        layer2 = self.sigmoid(layer2)
        
        layer3 = np.dot(layer2, w3) + b3 
        output = identity_function(layer3)
        
        return output
        

model = perceptron()

x = np.array([0.1, 0.5])
y = model.fit(x)

y



## 교재 코드
def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b1'] = np.array([0.1, 0.2])
    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network 

def forward(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, w1) + b1 
    z1 = sigmoid(a1)
    
    a2 = np.dot(z1, w2) + b2 
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2, w3) + b3 
    y = identity_function(a3)
    
    return y 

network = init_network()
x = np.array([1.0, 0.5])

y = forward(network, x)

print(y)