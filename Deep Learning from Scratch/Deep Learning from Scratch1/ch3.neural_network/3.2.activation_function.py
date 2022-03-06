import numpy as np 

# 3.2.2 계단함수 구현하기
def step_function(x):
    if x > 0 :
        return 1
    else : 
        return 0 
    
    
def step_function(x):
    y = x>0 
    return y.astype(np.int32)


x =np.array([-1.0, 1.0, 2.0])

y = x>0

y = y.astype(np.int32)


# 3.2.3 계단함수 그래프 

x = np.linspace(-1.0, 1.0, 100)
y = step_function(x)

import matplotlib.pyplot as plt 

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(x, y)
plt.show()



# 3.2.4 시그모이드 함수 구현하기 

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)

x = np.linspace(-5, 5, 100)
y = sigmoid(x)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(-6, 6)
ax.plot(x, y)
plt.show()


# 3.2.5 시그모이드 함수와 계단 함수 비교 
x = np.linspace(-5, 5, 100)
step_y = step_function(x)
sigmoid_y = sigmoid(x)


fig, ax = plt.subplots(figsize=(7, 7))
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(-6, 6)

ax.plot(x, step_y, linestyle='--', label='step')
ax.plot(x, sigmoid_y, label='sigmoid')
ax.legend()
plt.show()



# 3.2.7 비선형 함수(non-linear function)
# ReLU(Rectified Linear Unit)

def ReLU(x):
    if x <= 0 :
        return 0
    else: 
        return x 
    
def ReLU(x):
    return np.maximum(0, x)

