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
    for i in range(len(_x)/5):
        _y.append(randint(0,100))
        _x.append(randint(0,100))
    return _x,_y

# A function return w0 + w1*x + w2*w^2 +....+ wn-1*w^n-1
def f(x, w, n):
    result = 0
    for i in range(n):
        result += w[i,0]*(x**i)
    return result
    
# function of Least_Squares x: featrue data  y: target data n: power times of featrue
def Least_Squares(x,y,n): 
    
    # initialize
    A = []
    Y = []
    for i in range(len(x)):
        A.append([x[i]**power for power in range(n)])
        Y.append([y[i]])
    A = mat(A)
    Y = mat(Y)
    
    # mat function to solute the Least_Squares question
    return (A.T*A).I*A.T*Y 
    
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
        w = Least_Squares(randset_X, randset_Y,n)
        
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
            loss = temp_loss
            
    return bestfit,loss
    
    
if __name__ == "__main__":
    n=2
    setX,setY = drawLine(0.6,5,300,4)
    w = Least_Squares(setX,setY,n)
    w1,loss = ransacLine(setX,setY,4,100,5,n)
    print(w1)
    print(loss)
    x = np.linspace(0,100,100)
    print(w)
    y = [f(i,w,n) for i in x]
    _y = [f(i,w1,n) for i in x]
    plt.scatter(setX,setY)
    plt.plot(x, y, 'g')
    plt.plot(x, _y, 'r')
    plt.show()