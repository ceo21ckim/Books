import sys, os 
sys.path.append(os.pardir)

from common.functions import * 
from common.gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = weight_init_std * np.random.randn(output_size)
        
    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        layer1 = np.dot(x, w1) + b1
        output1 = sigmoid(layer1)
        
        layer2 = np.dot(output1, w2) + b2 
        output = softmax(layer2)
        
        return output 
    
    def loss(self, x, t):
        pred_y = self.predict(x)
        
        return cross_entropy_error(pred_y, t)
    
    def accuracy(self, x, t):
        pred_y = self.predict(x)
        pred_y = np.argmax(pred_y, axis=1)
        true_y = np.argmax(t, axis=1)
        
        accuracy = np.sum(pred_y == true_y ) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        
        grads = {}
        grads['w1'] = numerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        
        grads['w2'] = numerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        
        return grads
    
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=100)
net.params['w1'].shape 
net.params['b1'].shape 
net.params['w2'].shape
net.params['b2'].shape

# 4.5.2 mini-batch

from dataset.mnist import load_mnist 


(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

epochs = 10000
train_size = x_train.shape[0]
batch_size = 100
lr = 0.1

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)


for i in range(epochs):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    
    grad = network.numerical_gradient(x_batch, y_batch)
    
    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= lr * grad[key]
        
    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)
    
    if (i + 1) % 1 == 0 :
        print(f'loss : {train_loss_list[-1]:.4f}')
    
    

# 4.5.3 testing

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
lr = 0.1 

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 한 에폭당 반복 수 
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    
    # 기울기 계산
    grad = network.numerical_gradient(x_batch, y_batch)
    
    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= lr * grad[key]
        
    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss) 
    
    
    if i % iter_per_epoch == 0 :
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        
        train_acc_list(train_acc)
        test_acc_list(test_acc)
        
        print('train acc, test acc | ' + str(train_acc) + ', ' + str(test_acc))
        
        