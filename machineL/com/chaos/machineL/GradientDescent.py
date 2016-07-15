'''
Created on 2016年7月13日

@author: Hu Chao
'''
import math;
import random;
import numpy;
import copy;
class GradientDescent(object):
    '''
    classdocs
    '''


    def __init__(self, examples):
        '''
        Constructor
        '''
        self.examples = examples;
        
    def matrix(self):
        xMatrixArray = [];
        yMatrixArray = [];
        for example in self.examples:
            exampleX, exampleY = self.__createExampleXY(example);
            xMatrixArray.append(exampleX);
            yMatrixArray.append([exampleY,]);
        xMatrix = numpy.mat(xMatrixArray);
        yMatrix = numpy.mat(yMatrixArray);
        xMatrixT = xMatrix.T;
        thetaMatrix = (xMatrixT * xMatrix).I * xMatrixT * yMatrix;
        return thetaMatrix;
        
    def stochasticGradientDescent(self, step):
        theta = [];
        for example in self.examples:
            delta = self.__delta(example, theta);
            for j, value in enumerate(example[:-1]):
                theta[j] = theta[j] + step * delta * value;
        return theta;
    
    def batchGradientDescent(self, step):
        theta = [];
        oldTheta = [];
        count = 0;
        while count < 10000:
            deltas = [];
            for exampleValue in self.examples:
                delta = self.__delta(exampleValue, theta);
                deltas.append(delta); 
            for thetaIndex, thetaValue in enumerate(theta):
                for deltaIndex, deltaValue in enumerate(deltas):
                    theta[thetaIndex] = thetaValue + step * deltaValue * self.examples[deltaIndex][thetaIndex];
            if self.__distance(theta, oldTheta) < (step / 100000):
                return theta;
            count = count + 1;  
            oldTheta = theta.copy();   
        return [];
    
    def __htheta(self, exampleX, theta):
        while len(exampleX) > len(theta):
            theta.append(random.uniform(0, 10));
        htheta = 0;
        for j, value in enumerate(exampleX):
            htheta = htheta + (theta[j] * value);
        return htheta;
    
    def __createExampleXY(self, example):
        exampleY = copy.deepcopy(example[-1]);
        exampleX = [copy.deepcopy(value) for value in example];
        exampleX[-1] = 1;
        return exampleX, exampleY;
    
    def __delta(self, example, theta):
        exampleX, exampleY = self.__createExampleXY(example);
        htheta = self.__htheta(exampleX, theta);
        delta = exampleY - htheta;
        return delta;
    
    def __distance(self, src, dest):
        result = 0;
        maxLen = len(dest);
        if len(src) > len(dest):
            maxLen = len(src);
        for index in range(0, maxLen):
            if index >= len(src):
                result = result + math.pow(dest[index], 2);
            elif index >= len(dest):
                result = result + math.pow(src[index], 2);
            else:
                result = result + math.pow(dest[index] - src[index], 2);
        return math.sqrt(result);
        
        
    