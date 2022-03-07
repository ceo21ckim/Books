# 4.2.1 SSE(Sum of Squares for error)
import numpy as np 

# 2일 확률이 가장 높다고 추정.
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])

# 실제 라벨은 2 
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

(1/2)*np.sum((y-t)**2)


# def SSE
def SSE(y, t):
    return 0.5 * np.sum((y-t)**2)

SSE(y, t)

# 실제 라벨이 3일 경우 ?
t = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

SSE(y, t) # 0.6974


pred_y = np.array([0.1, 0.05, 0.05, 0.6, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0])
true_y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

SSE(pred_y, true_y) 

pred_y = np.array([0.1, 0.05, 0.05, 0.0, 0.6, 0.0, 0.1, 0.1, 0.0, 0.0])

SSE(pred_y, true_y)



# 4.2.2 cross entropy error 

def CE(pred_y, true_y):
    delta = 1e-7
    return -np.sum(true_y * np.log(pred_y + delta))
    
    

pred_y = np.array([0.1, 0.05, 0.05, 0.6, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0])
true_y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

CE(pred_y, true_y) 

pred_y = np.array([0.1, 0.05, 0.05, 0.0, 0.6, 0.0, 0.1, 0.1, 0.0, 0.0])

CE(pred_y, true_y)


# 4.2.3 미니배치 학습
import sys, os 
# 절대경로 설정
sys.path.append('..')

import numpy as np 
from dataset.mnist import load_mnist


(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(y_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

# 원핫인코딩으로 라벨이 주어졌을 경우 
def cross_entropy_error(pred_y, true_y):
    if pred_y.ndim == 1:
        true_y = true_y.reshape(1, true_y.size) # or true_y[np.newaxis, :]
        pred_y = pred_y.reshape(1, pred_y.size) # or pred_y[np.newaxis, :]
        
    batch_size = pred_y.shape[0]
    return -np.sum(true_y * np.log(y + 1e-7)) / batch_size 

# 숫자 라벨로 주어졌을 경우 
def cross_etropy_error(pred_y, true_y):
    if pred_y.ndim == 1:
        true_y = true_y.reshape(1, true_y.shape)
        pred_y = pred_y.reshape(1, pred_y.shape)
        
    batch_size = pred_y.shape[0]
    return -np.sum(np.log(pred_y[np.arange(batch_size), true_y] + 1e-7)) / batch_size 