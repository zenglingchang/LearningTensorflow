import pdb
import math
from numpy import *
from matplotlib import pyplot as plt

def drawCircle(r, n, bias, label):
    _feature = [[math.cos(2*pi/n*x)*r + random.normal(0,bias), math.sin(2*pi/n*x)*r + random.normal(0,bias)] for x in range(0,n+1)]
    _label = [[label] for i in range(0,n+1)]
    return _feature, _label
    
def drawLine(n):
    _feature = []
    _label = []
    for i in range(n):
        _feature.append([random.uniform(0,10),random.uniform(0,10)])
        _label.append([1 if _feature[i][0]>_feature[i][1] else 0])
    return _feature,_label
        
def sigmoid(X):
    return 1.0/(1+exp(-X))
    
class LR:
    def __init__(self, alpha=0.1):
        self._alpha = alpha
        self._w = mat(0)
        
    # Line_kernel
    def kernel(self, feature):
        return [feature[0], feature[1], feature[0]*feature[1], feature[0]**2, feature[1]**2]
    # n: means feature number
    # MaxTimes: means 
    def gradAscent(self, Data, ClassLabels, n, MaxTimes=500):
        _Data = [self.kernel(e) for e in Data]
        dataMatrix = mat(_Data)
        labelMatrix = mat(ClassLabels)
        weight = random.rand(5,1)
        for times in range(MaxTimes):
            E = sigmoid(dataMatrix*weight)
            loss = labelMatrix - E
            # pdb.set_trace()
            weight = weight + self._alpha*dataMatrix.T*loss
        self._w = weight
        print(weight)
    
    def _Get_region(self, X, Y):
        _region = [[x,y] 
                for x in linspace(X[0],X[1],100)
                for y in linspace(Y[0],Y[1],100)
                ]
        _label = self.classify(_region)
        region = []
        for i in range(len(_label)):
            if _label[i][0]>0.5:
                region.append(_region[i])
        return array(_region)
        
    def classify(self, data):
        _data = [self.kernel(e) for e in data]
        return sigmoid(_data*self._w)
        
    def error_ratio(self, testData, testlabel):
        count = 0
        real_label = mat(testlabel)
        label = self.classify(testData)
        for i in range(len(testData)):
            if (real_label[i,0]-0.5)*(label[i,0]-0.5) < 0:
                count += 1

        return 1.0*count/len(testData)

if __name__ == "__main__":
    # ----------------------Line test Data------------------
    # Data1 = []
    # Data2 = []
    # tarining_data,tarining_label = drawLine(100)
    # print(tarining_label)
    # for i in range(len(tarining_data)):
        # if tarining_label[i][0] == 1:
            # Data1.append(tarining_data[i])
        # else:
            # Data2.append(tarining_data[i])
            
    # ---------------------Circle test Data------------------
    Data1,Label1 = drawCircle(4,50,0.3,1)
    Data2,Label2 = drawCircle(3,50,0.3,1)
    tarining_data = Data1+Data2
    tarining_label = Label1+Label2
    lr = LR()
    lr.gradAscent(tarining_data,tarining_label,2)
    
    # ----------------------Line test Data------------------
    # testdata,testlabel = drawLine(100)
    # ---------------------Circle test Data------------------
    _Data1,_Label1 = drawCircle(4,50,0.3,1)
    _Data2,_Label2 = drawCircle(3,50,0.3,1)
    testdata = Data1+Data2
    testlabel = Label1+Label2
    error = lr.error_ratio(testdata,testlabel)
    print(error)
    Data1 = array(Data1)
    Data2 = array(Data2)
    region = lr._Get_region([-5,5],[-5,5])
    plt.scatter(Data1[:,0],Data1[:,1],color='r')
    plt.scatter(Data2[:,0],Data2[:,1],color='b')
    plt.scatter(region[:,0],region[:,1],color='g')
    plt.show()