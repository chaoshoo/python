'''
Created on 2016年7月13日

@author: Hu Chao
'''
import copy
import numpy as np;
class GradientDescent(object):
    '''
    classdocs
    '''


    def __init__(self, exampleXs, exampleYs, theta):
        '''
        Constructor
        '''
        self.__exampleXs = exampleXs
        self.__exampleYs = exampleYs
        self.__theta = theta    
          
        
    def stochasticGradient(self, gradient, step):
        theta = copy.deepcopy(self.__theta)
        row, column = self.__exampleXs.shape
        index = 0
        while index < column :
            delta = gradient(self.__exampleYs[index], self.__exampleXs.T[index].T, theta)
            theta = theta + delta * step
            index = index + 1
        return theta
    
    def batchGradient(self, gradient, step, divisor):
        theta = copy.deepcopy(self.__theta)
        oldTheta = copy.deepcopy(self.__theta)
        count = 0
        while step > 0:
            while count < 10000:
                delta = gradient(theta)
                theta = theta + delta * step
                if self.__distance(theta, oldTheta) < (step / divisor):
                    return theta
                count = count + 1  
                oldTheta = copy.deepcopy(theta)
            step = step / 10  
            count = 0 
        return []
    
    
    def __distance(self, src, dest):
        result = 0
        src = src.getA1()
        dest = dest.getA1()
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
        
        
    