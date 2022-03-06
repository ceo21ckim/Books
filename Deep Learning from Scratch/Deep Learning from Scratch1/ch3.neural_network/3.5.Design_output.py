# 3.5.1 항등 함수와 소프트맥스 함수 구현하기
import numpy as np 
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)

sum_exp_a = np.sum(exp_a)

y = exp_a / sum_exp_a 


def softmax(x):
    _exp = np.exp(x)
    return _exp / np.sum(_exp)


# exp는 지수함수이므로 값이 너무 커지면 값을 계산할 수 없게 된다. inf로 가버림..
# 그렇기에 상수를 뺴주어 값을 조정한다. (상수를 빼도 softmax의 값은 그대로임.)
a = np.array([1010, 1000, 990])
np.exp(a) / np.sum(np.exp(a))

# 중앙값을 빼든, 평균을 빼든, 최댓값을 빼든 softmax의 결과는 동일하다. 
c = np.median(a)
a - c

np.exp(a-c) / np.sum(np.exp(a-c))

def softmax(x):
    c = np.median(x)
    _exp = np.exp(a-c)
    return _exp / np.sum(_exp)

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(y)

np.sum(y) # softmax의 합은 무조건 1이 나와야 된다. 


