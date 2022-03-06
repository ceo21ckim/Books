import sys, os
import numpy as np 
import pickle
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image # 이미지를 다룰 때 사용하는 함수


# 이미지를 보여주는 함수
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    

# 부모디렉터리에서 dataset파일로 접근해 mnist.py를 실행한 후 내부에 존재하는 load_mnist를 가져옴.
def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, y_test 


class DNN: # Dongeon Neural Network
    def __init__(self):
        self.network = self.init_network()
        self.w1 = self.network['W1']
        self.b1 = self.network['b1']
        
        self.w2 = self.network['W2']
        self.b2 = self.network['b2']
        
        self.w3 = self.network['W3']
        self.b3 = self.network['b3']
        
    def init_network(self):
        with open('dataset/sample_weight.pkl', 'rb') as f:
            network = pickle.load(f)  
        return network
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        _exp = np.exp(x)
        return _exp / np.sum(_exp)
    
        
    def predict(self, x):
        w1, w2, w3 = self.w1, self.w2, self.w3 
        b1, b2, b3 = self.b1, self.b2, self.b3 
        
        layer1 = np.dot(x, w1) + b1 
        layer1 = self.sigmoid(layer1)
        
        layer2 = np.dot(layer1, w2) + b2 
        layer2 = self.sigmoid(layer2)
        
        layer3 = np.dot(layer2, w3) + b3 
        output = self.softmax(layer3)
        
        return output
    

if __name__ == '__main__':
    
    x, y = get_data()

    model = DNN()

    accuracy_cnt = 0 
    for i in range(len(x)):
        pred_y = model.predict(x[i]).argmax()
        
        if pred_y == y[i]:
            accuracy_cnt += 1 
            
    accuracy_cnt /= len(x)

    print(f'accuracy : {accuracy_cnt*100:.2f}%  <default>')


    x, y = get_data()
    model = DNN()

    batch_size = 32 
    accuracy_cnt = 0 

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        pred_y = model.predict(x_batch).argmax(axis=1)
        accuracy_cnt += np.sum(pred_y == y[i:i+batch_size])
        
    accuracy_cnt /= len(x)
    accuracy_cnt
    
    print(f'accuracy : {accuracy_cnt*100:.2f}% <with batch>')
    