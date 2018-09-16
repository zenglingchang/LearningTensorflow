import math
import numpy as np
from numpy import mat
from random import uniform
from random import random
from random import randint
from matplotlib import pyplot as plt
import pdb
pi=3.1415926

# generate set of points of Circle
def drawCircle(r, n, bias):
    _x = [math.cos(2*pi/n*x)*r + uniform(-bias,bias) for x in xrange(0,n+1)]
    _y = [math.sin(2*pi/n*x)*r + uniform(-bias,bias) for x in xrange(0,n+1)]
    return _x,_y

# generate set of points of n times equations
def drawLine(k,b,n,bias):
    _x = np.linspace(0,100,n).tolist()
    _y = [x*k+b+np.random.normal(-bias,bias)for x in _x]
    for i in range(int(len(_x)/5)):
        _y.append(randint(0,100))
        _x.append(randint(0,100))
    return _x,_y
    
def sin_line():
    setX = np.linspace(0, 100, 10)
    setY = [50*math.sin(2*np.pi*x/100) + np.random.normal(0,12) for x in setX]
    return setX,setY
    
# A function return w0 + w1*x + w2*w^2 +....+ wn-1*w^n-1
def f(x, w, n):
    result = 0
    for i in range(n):
        result += w[i,0]*(x**i)
    return result
    
# function of Least_Squares x: featrue data  y: target data n: power times of featrue
def Least_Squares(x,y,n,l_2=0): 
    
    # initialize
    A = []
    Y = []
    for i in range(len(x)):
        A.append([x[i]**power for power in range(n)])
        Y.append([y[i]])
    A = mat(A)
    Y = mat(Y)
    
    # Add l2_norm
    temp = A.T*A
    for i in range(n):
        temp[i,i]+=l_2
        
    # mat function to solute the Least_Squares question
    w = temp.I*A.T*Y
    loss = (A*w-Y)
    return w,sum(loss.T*loss)/len(x)
    
# mincnt means minsize of set of random points 
# maxItercnt means times of traverse 
# maxErrorThreshold means threshold of error
# setx,sety and n is the same as Least_Squares
def ransacLine(setX, setY, mincnt, maxIterCnt, maxErrorThreshold, n):
    count = 0
    bestfit = mat(0)
    consensus_size = 0
    loss = 0
    
    # travese mincnt times
    while count<maxIterCnt: 
    
        # initialize
        tempcon_size = 0
        temp_loss = 0.0
        count += 1
        
        #select randomlsetY the set of points and build the module
        randindesetX = np.random.choice(len(setX), mincnt,replace=False)
        randset_X = [setX[i] for i in randindesetX]
        randset_Y = [setY[i] for i in randindesetX]
        w,NoUse = Least_Squares(randset_X, randset_Y,n)
        
        # count size of consensus point
        for i in range(len(setX)):
            error = abs(f(setX[i],w,n) - setY[i])
            temp_loss += error**2
            if error < maxErrorThreshold:
                tempcon_size += 1
                
        # if the temp_module is superior to cur_module, instead!
        if tempcon_size > consensus_size:
            bestfit = w
            consensus_size = tempcon_size
            loss = temp_loss/len(setX)
            
    return bestfit,loss
    
    
if __name__ == "__main__":
    # tcase = int(input().strip())
    # setX = []
    # setY = []
    # for case in range(tcase):
        # point = list(map(int, input().strip().split()))
        # setX.append(point[0])
        # setY.append(point[1])
    setX,setY = sin_line()
    n=9
    # setX,setY = drawLine(0.6,5,300,4)
    w,loss = Least_Squares(setX,setY,n)
    w1,loss1 = ransacLine(setX,setY,9,100,2,n)
    if w1[1,0]>0:
        b = math.sqrt(1/(1+w1[1,0]**2))
    else:
        b = -math.sqrt(1/(1+w1[1,0]**2))
    a = w1[1,0]*b
    c = w1[0,0]*b
    print("{:.6f} {:.6f} {:.6f}".format(a,b,c))
    x = np.linspace(0,100,100)
    y = [f(i,w,n) for i in x]
    _y = [f(i,w1,n) for i in x]
    plt.scatter(setX,setY)
    plt.plot(x, y, 'g')
    plt.plot(x, _y, 'r')
    plt.show()
    