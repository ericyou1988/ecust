# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:33:35 2018
@author: Liang Qingyuan
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import math


# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, pN, dim, max_iter):
        self.w = 1  # 惯性权重
        self.c1 = 1  # pbest影响权重
        self.c2 = 1  # gbest影响权重
        self.r1 = random.uniform(0, 1)  # 粒子的个体学习因子
        self.r2 = random.uniform(0, 1)  # 社会的学习因子
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值

    # ---------------------目标函数------------------------------------
    def function(self, x):
        sum = 0
        length = len(x)
        x = x ** 2
        for i in range(length):
            sum += x[i]
        return math.sqrt(sum)

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(-10, 10)
                # self.V[i][j] = random.uniform(0,1)
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if (tmp < self.fit):
                self.fit = tmp
                self.gbest = self.X[i]

            # --------------------粒子迭代过程----------------------------------

    def iterator(self):
        fitness = []
        f_gbest = []
        # 开始迭代
        for t in range(self.max_iter):
            # 更新速度和位置
            for i in range(self.pN):
                for j in range(self.dim):
                    self.V[i][j] = self.w * self.V[i][j] + self.c1 * self.r1 * (self.pbest[i][j] - self.X[i][j]) + \
                                   self.c2 * self.r2 * (self.gbest[j] - self.X[i][j])

                    # 更新下一次迭代时的位置
                self.X[i] = self.X[i] + self.V[i]

                # 寻找最优解
                temp = self.function(self.X[i])
                if (temp < self.p_fit[i]):  # 更新个体最优#
                    self.p_fit[i] = temp  # 个体最优结果
                    self.pbest[i] = self.X[i]  # 个体最优位置
                    if (self.p_fit[i] < self.fit):  # 更新全局最优#
                        self.gbest = self.X[i]  # 全局最优位置
                        self.fit = self.p_fit[i]  # 全局最优结果

            # 确定目前最优值
            fitness.append(self.fit)
            f_gbest.append(self.gbest)
            print("#####fit#####", self.fit)  # 输出最优值
            print("#####f-gbest#####", self.gbest)  # 输出最优值位置

        return fitness, f_gbest
    # ----------------------程序执行-----------------------


my_pso = PSO(pN=10, dim=5, max_iter=10000)
my_pso.init_Population()
fitness, f_gbest = my_pso.iterator()
# -----------------------效果展示--------------------------
plt.figure(1)
plt.title("Figure")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, 10000)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=1)
plt.show() 