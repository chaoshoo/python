'''
Created on 2016年7月19日

@author: Hu Chao
'''

from com.chaos.machineL.GradientDescent import GradientDescent;
import random;
import matplotlib.pyplot as plt;
import numpy as np;
import com.chaos.machineL.Helper as Helper

class LogisticRegression(Helper.Helper):
    def __init__(self, start, end, param):
        Helper.Helper.__init__(self, [[x / 1000, self.hypothesis([x / 1000,0], [param,0], True)]for x in range(start , end)])
        self.__exampleXs = self.getExampleXs()
        self.__exampleYs = self.getExampleYs()
        self.__theta = self.getTheta()        
        
    def hypothesis(self, exampleX, theta, confuse=False):
        htheta = 0;
        for j, value in enumerate(exampleX):
            htheta =  htheta + (theta[j] * value);
        if not confuse:
            return 1 / (1 + np.exp(-1 * htheta));
        return 1 / (1 + np.exp(-1 * htheta + random.uniform(-0.8 * htheta, 0.8 * htheta)));

    def delta(self, exampleY, htheta):
        return exampleY - htheta;
    
    def distance(self, src, dest):
        result = 0
        maxLen = len(dest)
        if len(src) > len(dest):
            maxLen = len(src)
        for index in range(0, maxLen):
            if index >= len(src):
                result = result + np.abs(dest[index])
            elif index >= len(dest):
                result = result + np.abs(src[index])
            else:
                result = result + np.abs((dest[index] - src[index]))
        return np.sqrt(result)
    
    def hessian(self, theta):
        hessionA = []
        for i, value in enumerate(theta):
            t = []
            for j, value in enumerate(theta):
                t.append(self.hessionElement(i, j, theta))
            hessionA.append(t)
        return np.mat(hessionA)
        
    def hessionElement(self, i, j, theta):
        result = 0
        for exampleX in self.__exampleXs:
            matrixX = np.mat(exampleX).T
            h = self.hypothesisM(theta, matrixX)
            result = result + -1 * exampleX[i] * exampleX[j] * h * (1 - h)
        return result;
    
    def gradient(self, theta):
        g = []
        for tIndex in range(0, theta.size):
            value = 0
            for index, exampleX in enumerate(self.__exampleXs):
                matrixX = np.mat(exampleX).T
                h = self.hypothesisM(theta, matrixX)
                value = value + (self.__exampleYs[index] - h) * exampleX[tIndex]
            g.append([value,])  
        return np.mat(g)
        
    def hypothesisM(self, theta, matrixX):
        return 1 / (1 + np.exp(-1 * np.dot(np.mat(theta).T, np.mat(matrixX)).getA1()[0]))
    
    def newton(self):
        theta = np.mat(self.getTheta()).T
        count = 10
        try:
            while count > 0:
                hessionT = self.hessian(theta)
                hessionM = hessionT.getI()
                gradientM = self.gradient(theta)
                temp = hessionM.dot(gradientM)
                theta = theta-temp
                count = count - 1
        finally:
            print(count)
            return theta
        

if __name__ == '__main__':
    logisticRe = LogisticRegression(-50000, 50000, 3)
    originPointX = [value[0] for value in logisticRe.getExamples()];
    originPointY = [value[1] for value in logisticRe.getExamples()];
    x = np.linspace(-50, 50, 100000);
    plt.plot(originPointX, originPointY, 'ro');
    gd = GradientDescent(logisticRe);
    stochastic = gd.stochasticGradientDescent(0.0001);
    print(stochastic);
    stochasticY = [logisticRe.hypothesis([value,1], stochastic) for value in x];
    plt.plot(x, stochasticY, 'g');
    batch = gd.batchGradientDescent(0.0001, 0.000001);
    print(batch);  
    batchY = [logisticRe.hypothesis([value,1], batch) for value in x];
    plt.plot(x, batchY, 'b');
    newton = logisticRe.newton().T.getA1()
    print(newton)
    newtonY = [logisticRe.hypothesis([value,1], newton) for value in x];
    plt.plot(x, newtonY, 'r');
    plt.show();

