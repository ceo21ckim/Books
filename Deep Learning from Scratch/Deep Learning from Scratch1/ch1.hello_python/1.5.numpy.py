# 1.5.1 넘파이 가져오기
import numpy as np 


# 1.5.2 넘파이 배열 생성하기 
x = np.array([1.0, 2.0, 3.0])
print(x)

type(x)


# 1.5.3 넘파이의 산술 연산
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

x + y # element-wise add 

x - y # element-wise sub

x * y # element-wise mul

x / y # element-wise div 



# 1.5.4 넘파이의 N차원 배열 

a = np.array([
    [1, 2], 
    [3, 4]
])

print(a)

a.shape 

a.dtype # int32 


b = np.array([
    [3, 0], 
    [0, 6]
], dtype='int64')

print(b)

b.shape 

b.dtype # int64


a*b 

(a*b).dtype



# 1.5.5 브로드캐스트(broadcast)
x = np.array([
    [51, 55], 
    [14, 19], 
    [0, 4]
])

print(x)

x[0][1] # 55 (0, 1) 위치의 index


for row in x:
    print(row)
