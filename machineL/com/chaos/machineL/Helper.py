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


    def __init__(self, examples, initTheta):
        '''
        Constructor
        '''
        self.__examples = examples;  
        self.__exampleXs, self.__exampleYs = self.__createExampleXYs(self.__examples)
        self.__theta = initTheta(self.__exampleXs)  
        self.__exampleXs = np.mat(self.__exampleXs).T
        self.__exampleYs = np.mat(self.__exampleYs).T
        
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