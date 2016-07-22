'''
Created on 2016年7月13日

@author: Hu Chao
'''

#coding=utf-8
#文件开头加上、上面的注释。不然中文注释报错

#第一个自己学的机器学习算法、我目前只给出自己写的代码、注释较多。关于logistic regression和牛顿方法的概念，这里就不给出了。
from numpy import *
from math import *
import operator
import matplotlib
import matplotlib.pyplot as plt
#logistic regression ＋ 牛顿方法
def file2matrix():#完成了文件读取、迭代运算及绘图
#     fr1 = open(filename1)#打开一个文件
#     arrayOflines1 = fr1.readlines()#返回一个行数组
#     numberOfLines1 = len(arrayOflines1)#计算行数
#     matrix = zeros((numberOfLines1,3))#生成一个全零二维数组，numberOflines1行 3列。其实数据只有两列，一列全是1.
#     row = 0
#     for line in arrayOflines1:
#         line = line.strip()#声明：s为字符串，rm为要删除的字符序列 s.strip(rm)删除s字符串中开头、结尾处，位于 rm删除序列的字符
#                             #s.lstrip(rm)       删除s字符串中开头处，位于 rm删除序列的字符
#                             #s.rstrip(rm)      删除s字符串中结尾处，位于 rm删除序列的字符
#                             #注意：#1. 当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
#         listFromLine = line.split('  ')#将一行按（）中的参数符号分开放入一个list中
#         listFromLine[0:0] = ['1']#在list中最前面插入1，上面说了有一列全为一
#         for index,item in enumerate(listFromLine):#将list中的字符串形式的，全转换为对应的数值型。
#             listFromLine[index] = eval(item)
#         matrix[row,:] = listFromLine[:]#每个list赋给对应的二维数组的对应行
#         row+=1
    ArrayX = [[random.uniform(0,10) for x in range(0, 4)] for y in range(0, 4)]
    matrix = mat(ArrayX)#将数组转换为矩阵
#     fr1.close()
#     fr2 = open(filename2)
#     arrayOflines2 = fr2.readlines()
#     numberOfLines2 = len(arrayOflines2)
    matrixy = zeros((4,1))
    row = 0;
    while row < 4:
        matrixy[row,:] = row
        row+=1
    matrixy = mat(matrixy)
#     fr2.close()
    tempxxt = dot(matrix.T,matrix).I#这一部分乘上下面的denominator()既为H.I（Hessian矩阵的逆）
    theta=mat(zeros((4,1)))#初始θ参数，全零
    for i in range(0,2000):#迭代2000次得到了比较好的结果。我采取的可能是全向量的形式计算，感觉迭代次数有点偏多
        temphypo=Hypothesis(theta,matrix,row)
        tempdenominator=denominator(temphypo,row)
        tempnumerator=numerator(temphypo,matrixy,matrix)
        theta = theta+dot(tempxxt,tempnumerator)/tempdenominator
    temparray = ravel(Hypothesis(theta,matrix,row))
    temptheta = ravel(theta)
    for i in range(0,row):#根据hypothesis函数的值进行标记,方便绘图
        if(temparray[i]>=0.5):
            temparray[i]=1
        else:
            temparray[i]=0;
    fig = plt.figure()#生成了一个图像窗口
    ax = fig.add_subplot(111)#刚才窗口中的一个子图
    ax.scatter(ravel(matrix[:,1]),ravel(matrix[:,2]),200,20*temparray)#生成离散点，参数分别为点的x坐标数组、y坐标数组、点的大小数组、点的颜色数组
    x=linspace(-1,10,100)#起点为－1，终止10，100个元素的等差数组x
    ax.plot(x,-(temptheta[0]+temptheta[1]*x)/temptheta[2])#绘制x为自变量的函数曲线
    plt.show()

def Hypothesis(theta,x,row):#假设函数、是一个向量形式
    hypo = zeros((row,1))
    for i in range(0,row):
        temp = exp(-dot(theta.T,x[i].T))
        hypo[i,:] = [1/(1+temp)]
    return hypo

def denominator(hypo,row):#分母部分
    temp=zeros((row,1))
    temp.fill(1)
    temp=temp-hypo
    temp=dot(hypo.T,temp)
    return temp

def numerator(hypo,y,x):#牛顿方法的分子，我们要做的就是迭代使这一部分接近零
    temp = y-hypo
    temp = dot(temp.T,x)
    return temp.T

if __name__ == '__main__':
    file2matrix()
