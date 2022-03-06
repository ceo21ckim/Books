# 2.3.1 간단한 구현
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7 
    tmp = x1*w1 + x2*w2 
    if tmp <= theta :
        return 0 
    else :
        return 1 
    
AND(0, 0)
AND(1, 0)
AND(0, 1)
AND(1, 1)


# 2.3.2 가중치와 편향 도입(weight, bias)
import numpy as np 
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = 0.7 

w*x 

np.sum(w*x)

np.sum(w*x) + b


# 2.3.3 가중치와 편향 구현하기
# AND gate를 표현하는 조합의 대표적인 예가 (0.5, 0.5, 0.7)이기 때문에 해당 값을 사용함
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.7 
    
    tmp = sum(w*x) + b 
    if tmp <= 0 :
        return 0 
    else: 
        return 1
    

# NAND gate를 표현하는 조합의 대표적인 예가 (-0.5, -0.5, -0.7)이기 때문에 해당 값을 사용함
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = -0.7 
    
    tmp = np.sum(w*x) + b 
    
    if tmp <= 0 :
        return 0 
    else:
        return 1
    
    
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2 
    
    tmp = np.sum(w*x) + b 
    
    if tmp <= 0:
        return 0 
    else:
        return 1