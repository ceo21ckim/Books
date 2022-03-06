import numpy as np 
import matplotlib.pyplot as plt 

# 1.6.1 단순한 그래프 그리기
x = np.arange(0, 6, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.yticks(np.linspace(-1, 1, 6))
plt.show()


# 1.6.2 pyplot의 기능

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle='--', label='cos')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin & cos')
plt.legend()
plt.show()


# 1.6.3 이미지 표시하기
from matplotlib.image import imread 

# 동일한 디렉터리에 image파일을 집어넣어야 됨.
img = imread('ch1.hello_python/image.jpg')

img.shape  

plt.imshow(img)
plt.show()