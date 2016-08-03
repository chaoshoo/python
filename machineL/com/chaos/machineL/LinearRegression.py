'''
Created on 2016年7月19日

@author: Hu Chao
'''
import random
import matplotlib.pyplot as plt
import numpy as np
import com.chaos.machineL.Helper as Helper
import com.chaos.machineL.GradientDescent as GradientDescent

class LinearRegression(Helper.Helper, GradientDescent.GradientDescent):
    
    def __init__(self, start, end, param):
        Helper.Helper.__init__(self, [[x, self.__hypothesis([x, 1], [param, 0])]for x in range(start , end)])
        self.__gradient = GradientDescent.GradientDescent(self.getExampleXs(), self.getExampleYs(), self.getTheta())       
          
    def __hypothesis(self, exampleX, theta):
        htheta = 0
        for j, value in enumerate(exampleX):
            htheta = htheta + (theta[j] * value)
        return htheta + random.uniform(-0.5 * htheta, 0.5 * htheta)
    
    def hypothesis(self, exampleX, theta):
        htheta = exampleX.dot(theta.T)
        return htheta

    def __stochasticGradient(self, exampleY, exampleX, theta):
        return (exampleY - self.hypothesis(exampleX, theta)).dot(exampleX)
    
    def __batchGradient(self, theta):
        column, row = self.getExampleXs().shape
        result = np.mat([0 for x in range(0, row)])
        index = 0
        while index < column :
            delta = self.__stochasticGradient(self.getExampleYs()[index], self.getExampleXs()[index], theta)
            result = result + delta
            index = index + 1
        return result   

    def matrix(self):
        return self.__exampleXs.T.dot(self.__exampleXs).I.dot(self.__exampleXs.T).dot(self.__exampleYs).T
    
    def stochasticGradient(self, step):
        return self.__gradient.stochasticGradient(self.__stochasticGradient, step)
    
    def batchGradient(self, step, divisor):
        return self.__gradient.batchGradient(self.__batchGradient, step, divisor)

if __name__ == '__main__':
    linearRe = LinearRegression(1, 200, 5.5)
    originPointX = [value[0] for value in linearRe.getExamples()]
    originPointY = [value[1] for value in linearRe.getExamples()]
    plt.plot(originPointX, originPointY, 'ro')
    x = np.linspace(0, 200, 1000)
    stochastic = linearRe.stochasticGradient(0.000005)
    print(stochastic)
    stochasticY = [linearRe.hypothesis(np.mat([value,1]), stochastic).A[0] for value in x]
    plt.plot(x, stochasticY, 'r')
    batch = linearRe.batchGradient(0.0000005, 10)
    print(batch)  
    batchY = [linearRe.hypothesis([value,1], batch) for value in x]
    plt.plot(x, batchY, 'g')
    matrix = linearRe.matrix()
    print(matrix)
    matrixY = [linearRe.hypothesis(np.mat([value,1]), matrix).A[0] for value in x]
    plt.plot(x, matrixY, 'b')
    plt.show()
