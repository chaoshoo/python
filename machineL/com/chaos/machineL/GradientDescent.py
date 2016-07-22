'''
Created on 2016年7月13日

@author: Hu Chao
'''
import math
import copy
import numpy
class GradientDescent(object):
    '''
    classdocs
    '''


    def __init__(self, regression):
        '''
        Constructor
        '''
        self.__regression = regression
        
    def stochasticGradientDescent(self, step):
        theta = copy.deepcopy(self.__regression.getTheta())
        for index, exampleX in enumerate(self.__regression.getExampleXs()):
            exampleY = self.__regression.getExampleYs()[index]
            delta = self.__regression.delta(exampleY, self.__regression.hypothesis(exampleX, theta))
            for xIndex, xValue in enumerate(exampleX):
                theta[xIndex] = theta[xIndex] + step * delta * xValue
        return theta
    
    def batchGradientDescent(self, step, divisor):
        theta = copy.deepcopy(self.__regression.getTheta())
        oldTheta = copy.deepcopy(self.__regression.getTheta())
        count = 0
        while step > 0:
            while count < 10000:
                deltas = []
                for index, exampleX in enumerate(self.__regression.getExampleXs()):
                    exampleY = self.__regression.getExampleYs()[index]
                    delta = self.__regression.delta(exampleY, self.__regression.hypothesis(exampleX, theta))
                    deltas.append(delta) 
                for thetaIndex, thetaValue in enumerate(theta):
                    for deltaIndex, deltaValue in enumerate(deltas):
                        theta[thetaIndex] = thetaValue + step * deltaValue * self.__regression.getExampleXs()[deltaIndex][thetaIndex]
                if self.__regression.distance(theta, oldTheta) < (step / divisor):
                    return theta
                count = count + 1  
                oldTheta = copy.deepcopy(theta)
            step = step / 10  
            count = 0 
        return []
        
        
    