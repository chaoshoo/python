'''
Created on 2016年7月19日

@author: Hu Chao
'''
from com.chaos.machineL.GradientDescent import GradientDescent
import random
import matplotlib.pyplot as plt
import numpy as np
import com.chaos.machineL.Helper as Helper

class LinearRegression(Helper.Helper):
    
    def __init__(self, start, end, param):
        Helper.Helper.__init__(self, [[x, self.hypothesis([x, 1], [param, 0], True)]for x in range(start , end)])
        self.__exampleXs = self.getExampleXs()
        self.__exampleYs = self.getExampleYs()
        self.__theta = self.getTheta()
    
    def hypothesis(self, exampleX, theta, confuse=False):
        htheta = 0
        for j, value in enumerate(exampleX):
            htheta = htheta + (theta[j] * value)
        if not confuse:
            return htheta
        return htheta + random.uniform(-0.5 * htheta, 0.5 * htheta)

    def matrix(self):
        xMatrix = np.mat(self.__exampleXs)
        yMatrix = np.mat(self.__exampleYs).T
        xMatrixT = xMatrix.T
        resultMatrix = np.dot(np.dot(np.dot(xMatrixT, xMatrix).I, xMatrixT), yMatrix)
        return [value for value in resultMatrix.T.A[0]]
    
    def delta(self, exampleY, htheta):
        return exampleY - htheta
    
    def distance(self, src, dest):
        result = 0
        maxLen = len(dest)
        if len(src) > len(dest):
            maxLen = len(src)
        for index in range(0, maxLen):
            if index >= len(src):
                result = result + np.power(dest[index], 2)
            elif index >= len(dest):
                result = result + np.power(src[index], 2)
            else:
                result = result + np.power((dest[index] - src[index]), 2)
        return np.sqrt(result)

if __name__ == '__main__':
    linearRe = LinearRegression(1, 200, 5.5)
    gd = GradientDescent(linearRe)
    originPointX = [value[0] for value in linearRe.getExamples()]
    originPointY = [value[1] for value in linearRe.getExamples()]
    plt.plot(originPointX, originPointY, 'ro')
    x = np.linspace(0, 200, 1000)
    stochastic = gd.stochasticGradientDescent(0.000005)
    print(stochastic)
    stochasticY = [linearRe.hypothesis([value,1], stochastic) for value in x]
    plt.plot(x, stochasticY, 'r')
    batch = gd.batchGradientDescent(0.000005, 1000000)
    print(batch)  
    batchY = [linearRe.hypothesis([value,1], batch) for value in x]
    plt.plot(x, batchY, 'g')
    matrix = linearRe.matrix()
    print(matrix)
    matrixY = [linearRe.hypothesis([value,1], matrix) for value in x]
    plt.plot(x, matrixY, 'b')
    plt.show()
