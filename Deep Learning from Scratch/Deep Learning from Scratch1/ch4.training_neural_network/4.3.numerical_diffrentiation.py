import numpy as np 
# 전방 차분
def numerical_diff(f, x):
    h = 1e-50
    return (f(x+h) - f(x)) / h 

# 반올림 오차를 고려해 h를 1e-4로 설정하고, 중앙 차분을 진행함.
def numerical_diff(f, x):
    h = 1e-4 
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x 


numerical_diff(function_1, 5)

numerical_diff(function_1, 10)

def function_2(x):
    return x[0]**2 + x[1]**2 

# x_0 = 3, x_1 = 4 일 경우 
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

numerical_diff(function_tmp1, 3.0)


def function_tmp2(x1):
    return 3**2.0 + x1*x1 

numerical_diff(function_tmp2, 4.0)


