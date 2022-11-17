
##创建时间：2022-10-26
##1：导入所需模块

import numpy as np
import scipy.optimize as opt


## 线性搜索
from scipy.optimize import newton


class LineSearch:

    @staticmethod
    def armijo(func, gunc, x0, d0, f0, g0):

        """利用armijo搜索计算步长: f(xk+1)<= f(xk)+sigma1 * alpha * gk * dk"""

        # 初始步：参数确定
        sigma1 = 0.2
        pho = 0.5  # 压缩因子

        itermax = 20  # 允许的最大迭代次数

        alpha = 1  ##初始单位步
        i = 0
        nf = 0
        ng = 0

        gd = np.dot(g0, d0)

        while True:

            f1 = func(x0 + alpha * d0)
            nf += 1

            if f1 <= f0 + sigma1 * alpha * gd:
                return alpha, nf, ng
            else:
                alpha *= pho * alpha
                i += 1

            if i > itermax:
                # raise IndexError('线性搜索迭代次数溢出')
                print('线性搜索迭代次数溢出')
                return alpha, nf, ng

    @staticmethod
    def wolfe(func, gunc, x0, d0, f0, g0):

        """计算满足wolfe搜索条件的步长（伪代码见教材）
        编写时间: 2021-03
        输入说明：
        输出说明：
        参数说明："""

        # 初始步：参数确定
        sigma1 = 0.01
        sigma2 = 0.9
        pho = 0.5  # 压缩因子
        pho1 = 0.1

        itermax = 20  # 允许的最大迭代次数

        alpha = 1  ##初始单位步
        i = 0
        nf = 0
        ng = 0

        gd0 = np.dot(g0, d0)

        ##首先检查单位步
        f1 = func(x0 + alpha * d0)
        nf += 1

        g1 = gunc(x0 + alpha * d0)
        ng += 1
        gd1 = np.dot(g1, d0)

        if (f1 <= f0 + sigma1 * alpha * gd0) and (gd1 >= sigma2 * gd0):
            return alpha, nf, ng

        """单位步长不成立"""
        a = 0
        b = 10 * alpha
        alpha = b

        while True:

            f1 = func(x0 + alpha * d0)
            nf += 1

            """检查wolfe搜索的第一个条件"""
            if f1 > f0 + sigma1 * alpha * gd0:
                alpha = pho * alpha  # 不满足，步长太大，缩小步长
                continue
            else:
                """此时，alpha满足第一个条件"""
                a = alpha
                b = alpha / pho
                g1 = gunc(x0 + alpha * d0)
                ng += 1
                gd1 = np.dot(g1, d0)

                """检查wolfe搜索的第二个条件"""
                if gd1 >= sigma2 * gd0:
                    return alpha, nf, ng
                else:
                    alpha = a + pho1 * (b - a)  # 不满足，步长太小，放大步长

            i += 1
            if i > itermax:
                print('线性搜索迭代次数溢出')
                return alpha, nf, ng

    @staticmethod
    def wolfe1(func, gunc, x0, d0, f0, g0):

        """计算满足wolfe搜索条件的步长（伪代码见教材）
        编写时间: 2021-03
        输入说明：
        输出说明：
        参数说明："""

        # 初始步：参数确定
        sigma1 = 0.01
        sigma2 = 0.9
        pho = 0.5  # 压缩因子

        itermax = 20  # 允许的最大迭代次数

        alpha = 1  ##初始单位步
        i = 0
        nf = 0
        ng = 0

        a = 0
        b = 10 * alpha  # float('inf')

        gd0 = np.dot(g0, d0)

        while True:

            f1 = func(x0 + alpha * d0)
            nf += 1

            """检查wolfe搜索的第一个条件"""
            if f1 > f0 + sigma1 * alpha * gd0:
                b = alpha
                alpha = a + pho * (b - a)  # 不满足，步长太大，缩小步长
                continue

            else:
                g1 = gunc(x0 + alpha * d0)
                ng += 1
                gd1 = np.dot(g1, d0)

                """检查wolfe搜索的第二个条件"""
                if gd1 >= sigma2 * gd0:
                    return alpha, nf, ng

                else:
                    a = alpha  # 不满足，步长太小，放大步长
                    alpha = a + pho * (b - a)

            i += 1
            if i > itermax:
                print('线性搜索迭代次数溢出')
                return alpha, nf, ng


## 最速下降算法

def steep(func, gunc, x0, linesearch):
    epsilon = 1e-8
    itermax = len(x0) * 200
    k = 0

    f0 = func(x0)
    g0 = gunc(x0)
    nf = 1
    ng = 1

    d0 = -g0

    while k < itermax:

        if np.dot(g0, g0) >= epsilon ** 2:

            alpha, nf0, ng0 = linesearch(func, gunc, x0, d0, f0, g0)

            nf += nf0
            ng += ng0
            k += 1

            x0 = x0 + alpha * d0
            f0 = func(x0)
            g0 = gunc(x0)
            d0 = -g0
            nf += 1
            ng += 1
        else:
            return x0, f0, nf, ng, k
    else:

        print('最速下降算法迭代溢出')
        return x0, f0, nf, ng, k


##测试问题-二次函数

class Quad:

    def __init__(self, Q, q, c=0):
        """f = 0.5 * x* Q* x + q*x +c"""

        Q = np.asarray(Q)
        self.Q = Q
        q = np.asarray(q)
        self.q = q
        self.c = c

    def grad(self, x):
        x = np.asarray(x).flatten()
        return np.dot(self.Q, x) + self.q

    def hess(self, x):
        return self.Q

    def func(self, x):
        x = np.asarray(x).flatten()
        Qx = np.dot(self.Q, x)
        xQx = 0.5 * np.dot(x, Qx)
        qx = np.dot(self.q, x)
        return xQx + qx + self.c


########################################################

if __name__ == "__main__":
    Q = np.array([[21, 4], [4, 15]])
    q = np.array([2, 3])
    c = 10
    x0 = np.array([-30, 100])

    problem = Quad(Q, q, c)
    func = problem.func
    gunc = problem.grad
    hess = problem.hess

    #     print('最速下降算法')
    #     linesearch = LineSearch.armijo
    #     x1, f0,nf,ng ,k = steep(func, gunc, x0, linesearch)
    #     print(x1, f0,nf,ng ,k)

    #     print('-----wolfe-----')
    #     linesearch = LineSearch.wolfe
    #     x1, f0,nf,ng ,k = steep(func, gunc, x0, linesearch)
    #     print(x1, f0,nf,ng ,k)

    print('newton算法')
    linesearch = LineSearch.armijo

    x1, f0, nf, ng, k = newton(func, gunc, hess, x0, linesearch)
    print(x1, f0, nf, ng, k)

    print('-----wolfe-----')
    linesearch = LineSearch.wolfe
    x1, f0, nf, ng, k = newton(func, gunc, hess, x0, linesearch)
    print(x1, f0, nf, ng, k)
