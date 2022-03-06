# 1.3.1 산술 연산
1-2

4*5

7/5

3**2

# 1.3.2 자료형
type(10) # int

type(2.718) # float

type('hello') # str


# 1.3.3 변수
x = 10 
print(x) # 10

x = 100 
print(x) # 100

y = 3.14 
x*y 

type(x*y) # float


# 1.3.4 리스트 
a = [1, 2, 3, 4, 5]
print(a)

len(a)

a[0]

a[4]

a[4] = 99
print(a)

# slicing 
print(a)

a[0:2] # == a[:2]

a[1:]

a[:3]

a[:-1]

a[:-2]


# 1.3.5 딕셔너리
me = {'height' : 180}
me['height'] # 180

me['weight'] = 70
print(me)


# 1.3.6 bool 
hungry = True 
sleepy = False

type(hungry) # bool 

not hungry # False 

hungry and sleepy # False 

hungry or sleepy # True 


# 1.3.7 if문
hungry = True 
if hungry:
    print("I'm hungry")
    
hungry = False 
if hungry:
    print("I'm hungry")
    
else: 
    print("I'm not hungry")
    print("I'm sleepy")
    

# 1.3.8 for문
for i in [1, 2, 3]:
    print(i)
    
# 1.3.9 함수

def hello():
   print("Hello World!")
   
hello()


def hello(object):
    print("Hello " + object + '!')
    
hello('cat')


