import csv
from random import random
style = {'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
def load_file(file_name):
    csv_file = csv.reader(open(file_name,'r'))
    training_data = []
    testing_data = []
    for e in csv_file:
        if random()>0.4:
            training_data.append([float(e[0]),float(e[1]),float(e[2]),float(e[3]),style[e[4]]])
        else:
            testing_data.append([float(e[0]),float(e[1]),float(e[2]),float(e[3]),style[e[4]]])
    return training_data,testing_data
    
class SVM:
    def __init__(self):
    

if __name__ == '__main__':
    training_data,testing_data = load_file('iris.csv');
    print(len(training_data))
    print(training_data)
    print(len(testing_data))
    print(testing_data)