import numpy as np 

# 3.3.1 다차원 배열 

a = np.array([1, 2, 3, 4])
print(a)

np.ndim(a) # 1 : 차원을 확인할 떄 사용.

a.shape # (4, )

a.shape[0] # 4


b = np.array([
    [1, 2], 
    [3, 4], 
    [5, 6]
])

print(b)

np.ndim(b) # 2

b.shape # (3, 2)


# 3.3.2 행렬의 곱 
a = np.array([
    [1, 2], 
    [3, 4]
])

b = np.array([
    [5, 6], 
    [7, 8]
])

a.shape, b.shape 

np.dot(a, b) # 3 by 3 matrix



a = np.array([
    [1, 2, 3], 
    [4, 5, 6]
])

a.shape # (2, 3)

b = np.array([
    [1, 2], 
    [3, 4], 
    [5, 6]
])

b.shape # (3, 2)

np.dot(a, b) # 2 by 2 matrix


c = np.array([
    [1, 2], 
    [3, 4]
])


c.shape # (2, 2)

a.shape # (2, 3)

np.dot(c, a) # 2 by 3 matrix 

np.dot(a, c) # error 


a = np.array([
    [1, 2],
    [3, 4], 
    [5, 6]
])

a.shape # (3, 2)

b = np.array([7, 8])

b.shape # (2, )

np.dot(a, b) # 3 by 1 matrix 


# 3.3.3 신경망에서 행렬 곱 

x = np.array([1, 2])
x.shape # (2, )

w = np.array([
    [1, 3, 5], 
    [2, 4, 6]
])

w.shape # (2, 3)

y = np.dot(w, x)

print(y) # [5 11 17] : 3 by 1 matrix 