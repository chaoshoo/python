'''
Created on 2016年7月19日

@author: Hu Chao
'''

import random;
import matplotlib.pyplot as plt;
import numpy as np;
import copy
import com.chaos.machineL.Helper as Helper
from com.chaos.machineL import GradientDescent
        
def initTheta(exampleXs):
    theta = [];
    for exampleX in exampleXs:
        while len(exampleX) > len(theta):
            theta.append(random.uniform(10, 100));
    return np.mat(theta).T
    
class LogisticRegression(Helper.Helper):
    def __init__(self, start, end, param):
        Helper.Helper.__init__(self, [[x / 1000, self.__hypothesis([x / 1000, 1], [param, 0])]for x in range(start , end)], initTheta)
        self.__gradient = GradientDescent.GradientDescent(self.getExampleXs(), self.getExampleYs(), self.getTheta())  
          
    def __hypothesis(self, exampleX, theta):
        htheta = 0;
        for j, value in enumerate(exampleX):
            htheta =  htheta + (theta[j] * value);
        return 1 / (1 + np.exp(-1 * htheta + random.uniform(-0.8 * htheta, 0.8* htheta)));  
        
    def hypothesis(self, exampleX, theta):
        htheta = theta.T.dot(exampleX);
        return 1 / (1 + np.exp(-1 * htheta));

    def __stochasticGradient(self, exampleY, exampleX, theta):
        return (exampleX).dot(exampleY - self.hypothesis(exampleX, theta))
    
    def __batchGradient(self, theta):
        row, column = self.getExampleXs().shape
        result = np.mat([0 for x in range(0, row)]).T
        index = 0
        while index < column :
            delta = self.__stochasticGradient(self.getExampleYs()[index], self.getExampleXs().T[index].T, theta)
            result = result + delta
            index = index + 1
        return result        
     
    def __hessian(self, theta):
        row, column = theta.shape
        hessionA = np.zeros((row,row))  
        for i in range(0, row):
            for j in range(0, row):
                hessionA[i][j] = self.__hessionElement(i, j, theta)
        return np.mat(hessionA)
        
    def __hessionElement(self, i, j, theta):
        result = 0
        h = self.hypothesis(self.getExampleXs(), theta)
        h = h - np.power(h, 2)        
        row, column = h.shape
        index = 0
        while index < column :
#             print(self.getExampleXs().T[index].getA1()[i])
#             print(self.getExampleXs().T[index].getA1()[j])
#             print(h)
            result = result - self.getExampleXs().T[index].getA1()[i] * self.getExampleXs().T[index].getA1()[j] * h.getA1()[index]
            index = index + 1        
        return result;
    
    def newton(self):
        theta = copy.deepcopy(self.getTheta())
        count = 10
        try:
            while count > 0:
                hession = self.__hessian(theta)
                hessionI = hession.I
                gradient = self.__batchGradient(theta)
                theta = theta - hessionI.dot(gradient)
                count = count - 1
        finally:
            print(count)
            return theta
            
    def stochasticGradient(self, step):
        return self.__gradient.stochasticGradient(self.__stochasticGradient, step)
    
    def batchGradient(self, step, divisor):
        return self.__gradient.batchGradient(self.__batchGradient, step, divisor)
        

if __name__ == '__main__':
    logisticRe = LogisticRegression(-50000, 50000, 3)
    originPointX = [value[0] for value in logisticRe.getExamples()];
    originPointY = [value[1] for value in logisticRe.getExamples()];
    x = np.linspace(-50, 50, 100000);
    plt.plot(originPointX, originPointY, 'ro');
    stochastic = logisticRe.stochasticGradient(0.0001);
    print(stochastic);
    stochasticY = [logisticRe.hypothesis(np.mat([value,1]).T, stochastic).A[0] for value in x]
    plt.plot(x, stochasticY, 'g');
    batch = logisticRe.batchGradient(0.0001, 0.00001);
    print(batch);  
    batchY = [logisticRe.hypothesis(np.mat([value,1]).T, batch).A[0] for value in x];
    plt.plot(x, batchY, 'b');
    newton = logisticRe.newton().T.getA1()
    print(newton)
    newtonY = [logisticRe.hypothesis(np.mat([value,1]).T, newton).A[0] for value in x];
    plt.plot(x, newtonY, 'r');
    plt.show();

