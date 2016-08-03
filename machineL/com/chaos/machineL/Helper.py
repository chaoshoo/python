'''
Created on 2016年7月21日

@author: Hu Chao
'''

import copy
import random
from bigfloat.core import BigFloat
import numpy as np
class Helper(object):
    '''
    classdocs
    '''


    def __init__(self, examples):
        '''
        Constructor
        '''
        self.__examples = examples;  
        self.__exampleXs, self.__exampleYs = self.__createExampleXYs(self.__examples)
        self.__theta = self.__initTheta()  
        self.__exampleXs = np.mat(self.__exampleXs)
        self.__exampleYs = np.mat(self.__exampleYs).T
        self.__theta = np.mat(self.__theta)
        
    def __initTheta(self):
        theta = [];
        for exampleX in self.__exampleXs:
            while len(exampleX) > len(theta):
                theta.append(random.uniform(10, 100));
        return theta;
        
    def __createExampleXY(self, example):
        exampleY = self.__copy(example[-1])
        exampleX = [self.__copy(value) for value in example]
        exampleX[-1] = 1;
        return exampleX, exampleY;
    
    def __copy(self, value):
        if type(value) == BigFloat:
            return value.copy()
        else:
            return copy.deepcopy(value)
    
    def __createExampleXYs(self, examples):
        exampleXs = []
        exampleYs = []
        for example in examples:
            exampleX, exampleY = self.__createExampleXY(example)
            exampleXs.append(exampleX)
            exampleYs.append(exampleY)
        return exampleXs, exampleYs

    def getExamples(self):
        return self.__examples
    
    def getExampleXs(self):
        return self.__exampleXs
    
    def getExampleYs(self):
        return self.__exampleYs
    
    def getTheta(self):
        return self.__theta