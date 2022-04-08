# weight decay

import numpy as np 
import matplotlib.pyplot as plt
from common.util import shuffle_dataset 
from dataset.mnist import load_mnist
from common.functions import *
from common.layers import *
from common.util import *
from common.gradient import numerical_gradient 
from collections import OrderedDict 
from common.multi_layer_net import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100)
node_num = 100 
hidden_layer_size = 5 
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
        
    # w = np.random.randn(node_num, node_num) * 1 
    w = np.random.randn(node_num, node_num) * 0.01
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z



for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + '-layer')
    plt.hist(a.flatten(), 30, range = (0, 1))
plt.show()

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x 
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
            
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)        
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        
        return grads 
    
    def gradient(self, x, t):
        self.loss(x, t)
        
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads 

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr 
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
node_num = 100
w = np.random.randn(node_num, node_num) / np.sqrt(node_num)



(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

x_train = x_train[:300]
y_train = y_train[:300]

network = MultiLayerNet(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100, 100], output_size = 10)
optimizer = SGD(lr=0.01)

max_epochs = 201 
train_size = x_train.shape[0]
batch_size = 100 

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    
    grads = network.gradient(x_batch, y_batch)
    optimizer.update(network.params, grads)
    
    if i % iter_per_epoch == 0 :
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break
        

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio 
        self.mask = None 
    
    def forward(self, x, train_flg = True):
        if train_flg:
            self.mask = np.random.randn(*x.shape) > self.dropout_ratio 

(x_train, y_train), (x_test, y_test) = load_mnist()

x_train, y_train = shuffle_dataset(x_train, y_train)

validation_rate = 0.20 
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
y_val = y_train[:validation_num]
x_train = x_train[validation_num:]
y_train = y_train[validation_num:]

weight_decay = 10 ** np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)

