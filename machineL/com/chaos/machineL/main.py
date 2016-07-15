'''
Created on 2016年7月13日

@author: Hu Chao
'''
from com.chaos.machineL.GradientDescent import GradientDescent;
import random;

if __name__ == '__main__':
    data1 = [[x, y, z, 3*x+4*y+5*z+random.uniform(-10 * x, 10 * x)]for x in range(1 , 10) for y in range(1, 10) for z in range(1, 10)];
    gd = GradientDescent(data1);
    print(gd.stochasticGradientDescent(0.005));
    print(gd.batchGradientDescent(0.00005));
    print(gd.matrix());