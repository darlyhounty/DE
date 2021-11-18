import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import timeit


# Differential Evolution Algorithm 差分进化算法


import random

draw_de = []
draw_de1=[]
draw_de2=[]

def f(v):
    return -20 * np.exp(-0.2 * np.sqrt((v[0] ** 2 + v[1] ** 2) / 2)) - np.exp(
        0.5 * (np.cos(2 * np.pi * v[0]) + np.cos(2 * np.pi * v[1]))) + 20 + np.e


class Population(object):
    def __init__(self, min_range, max_range, dim, factor, rounds, size, object_func, CR):
        self.min_range = min_range
        self.max_range = max_range
        self.dimension = dim
        self.factor = factor
        self.rounds = rounds
        self.size = size
        self.cur_round = 1
        self.CR = CR
        self.get_object_function_value = object_func
        # 初始化种群
        self.individuality = [np.array([random.uniform(self.min_range, self.max_range) for s in range(self.dimension)])
                              for tmp in range(size)]
        self.object_function_values = [self.get_object_function_value(v) for v in self.individuality]
        self.mutant = None

    def mutate(self):
        self.mutant = []
        for i in range(self.size):
            select_range = [x for x in range(self.size)]
            select_range.remove(i)
            r0, r1, r2 = np.random.choice(select_range, 3, replace=False)
            tmp = self.individuality[r0] + (self.individuality[r1] - self.individuality[r2]) * self.factor
            for t in range(self.dimension):
                if tmp[t] > self.max_range or tmp[t] < self.min_range:
                    tmp[t] = random.uniform(self.min_range, self.max_range)
            self.mutant.append(tmp)

    def crossover_and_select(self):
        for i in range(self.size):
            Jrand = random.randint(0, self.dimension)
            for j in range(self.dimension):
                if random.random() > self.CR and j != Jrand:
                    self.mutant[i][j] = self.individuality[i][j]
                tmp = self.get_object_function_value(self.mutant[i])
                if tmp < self.object_function_values[i]:
                    self.individuality[i] = self.mutant[i]
                    self.object_function_values[i] = tmp

    def print_best(self):
        m = min(self.object_function_values)
        i = self.object_function_values.index(m)
        if self.factor == 0.1: # self.factor = 0.3 # self.CR == 0.25
            draw_de.append(abs(0 - m))#оптимальное значение точка
        if self.factor == 0.5:
            draw_de1.append(abs(0 - m))
        if self.factor == 0.7:
            draw_de2.append(abs(0 - m))
        # print("точка：" + str(self.individuality[i]))
        # print("оптимальное значение：" + str(m))
        # print("轮数：" + str(self.cur_round))
        # print("最佳个体：" + str(self.individuality[i]))
        # print("目标函数值：" + str(m))

    def evolution(self):
        while self.cur_round < self.rounds:
            self.mutate()
            self.crossover_and_select()
            self.print_best()
            self.cur_round = self.cur_round + 1


def f(v):
    return -20 * np.exp(-0.2 * np.sqrt((v[0] ** 2 + v[1] ** 2) / 2)) - np.exp(
        0.5 * (np.cos(2 * np.pi * v[0]) + np.cos(2 * np.pi * v[1]))) + 20 + np.e


def draw_pic(X1, X2, Z, z_max, title, z_min=0):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    ax.set_zlim(z_min, z_max)
    ax.set_title(title)
    plt.show()


def get_X_AND_Y(X1_min, X1_max, X2_min, X2_max):
    X1 = np.arange(X1_min, X1_max, 0.1)
    X2 = np.arange(X2_min, X2_max, 0.1)
    X1, X2 = np.meshgrid(X1, X2)  # 传参数
    return X1, X2


# Ackley测试函数
def Ackley(X1_min=-5, X1_max=5, X2_min=-5, X2_max=5):
    X1, X2 = get_X_AND_Y(X1_min, X1_max, X2_min, X2_max)
    Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X1 ** 2 + X2 ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * X1) + np.cos(2 * np.pi * X2))) + np.e + 20
    return X1, X2, Z, 15, "Ackley function"




# test
if __name__ == "__main__":

    #  #绘图d=2 3d图像
    # z_min = None
    # X1, X2, Z, z_max, title = Ackley()
    #
    # draw_pic(X1, X2, Z, z_max, title, z_min)

    #绘制近似图
    rounds = 300
    start = timeit.default_timer()
    p = Population(min_range=-5, max_range=5, dim=2, factor=0.1, rounds=300, size=100, object_func=f, CR=0.9)
    p.evolution()
    #p1 = Population(min_range=-5, max_range=5, dim=2, factor=0.5, rounds=300, size=100, object_func=f, CR=0.5)
    #p1.evolution()
    #p2 = Population(min_range=-5, max_range=5, dim=2, factor=0.7, rounds=300, size=100, object_func=f, CR=0.5)
    #p2.evolution()
    end = timeit.default_timer()
    print('Run time : %s s' % (end - start))

    oreder_de = list(range(1, rounds))
    plt.plot(oreder_de, np.log(draw_de),label ="F = 0.1")
    #plt.plot(oreder_de, np.log(draw_de1),label = "F = 0.5")
    #plt.plot(oreder_de, np.log(draw_de2),label = "F = 0.7")
    plt.xlabel("номер поколения")
    plt.ylabel("текущее оптимальное значение погрешности(log)")
    #plt.legend()
    plt.show()
