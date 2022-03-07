
class mul_layer:
    def __init__(self):
        self.x = None 
        self.y = None 
        
    def forward(self, x, y):
        self.x = x 
        self.y = y 
        
        out = self.x * self. y 
        
        return out 
    
    def backward(self, dout):
        dx = dout * self.y 
        dy = dout * self.x 
        
        return dx, dy 
    
    
apple = 100 
apple_num = 2 
tax = 1.1 

mul_apple_layer = mul_layer()
mul_tax_layer = mul_layer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)


# backward 
dprice = 1 

# apple_price = x, dtax = y
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)
    
    
    
# add layer
class add_layer:
    def __init__(self):
        self.x = None 
        self.y = None 
        
    def forward(self, x, y):
        self.x = x 
        self.y = y 
        return x + y 
    
    def backward(self, dout):
        dx = dout * 1 
        dy = dout * 1 
        
        return dx, dy
    
apple = 100 
apple_num = 2 
orange = 150 
orange_num = 3 
tax = 1.1 

mul_apple_layer = mul_layer()
mul_orange_layer = mul_layer()
add_apple_orange_layer = add_layer()
mul_tax_layer = mul_layer()

# forward
apple_price = mul_apple_layer.forward(apple_num, apple)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)


# backward

dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple_num, dapple = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print(price)
print(dapple_num, dapple, dorange, dorange_num, dtax)