import numpy as np
import scipy

def dampnm_armijo(fun, gfun, hessian, x0):  # 使用Armijo准则来求步长因子的阻尼牛顿法
    maxk = 100
    #rho = .55
    sigma = .4
    k = 0 #循环次数
    epsilon = 1e-5
    nf=0 #函数调用次数
    ng=0 #梯度调用次数
    while k < maxk:
        gk = gfun(x0)
        ng += 1
        Gk = hessian(x0)
        dk = -np.linalg.inv(Gk) @ gk #Gk矩阵的逆乘gk
        if np.linalg.norm(gk) < epsilon: #norm表示范数，默认为二范数
            break
        alpha=1
        b = 0.8  # alpha的压缩因子
        nf += 1
        if fun(x0 + alpha * dk) <= fun(x0) + sigma * alpha * gk.T @ dk:

            break
        else:
            alpha *= b

        x0 = x0 + alpha * dk
        k += 1
    x = x0
    val = fun(x)
    return x, val, k,nf,ng


def dampnm_wolfe(fun, gfun, hessian, x0):  # 使用Wolfe准则来求步长因子的阻尼牛顿法
    maxk = 1000
    k = 0
    ng=0
    nf=0
    epsilon = 1e-5
    while k < maxk:
        gk = gfun(x0)
        ng +=1
        Gk = hessian(x0)
        dk = -np.linalg.inv(Gk) @ gk
        if np.linalg.norm(gk) < epsilon:
            break
        # m = 0
        rho = 0.4
        sigma = 0.5
        a = 0
        b = np.inf
        alpha = 1
        # j = 0
        while True:
            nf += 1
            if not ((fun(x0) - fun(x0 + alpha * dk)) >= (-rho * alpha * gfun(x0).T @ dk)):
                # j+=1
                b = alpha
                alpha = (a + alpha) / 2
                continue
            if not (gfun(x0 + alpha * dk).T @ dk >= sigma * gfun(x0).T @ dk):
                a = alpha
                alpha = np.min([2 * alpha, (alpha + b) / 2])
                continue
            break

        x0 = x0 + alpha * dk
        k += 1
    x = x0
    val = fun(x)
    return x, val, k,nf,ng

class rosenbrock:
    def __init__(self, n=2, x=[], *args, **kwargs):
        self.n = n  ##问题的维数
        if x == []:
            for i in range(n):
                if i % 2 == 0:
                    x.append(-1.2)
                else:
                    x.append(1)
        self.x = np.asarray(x).flatten()  ##当前点扁平化
        self.xopt = np.ones(n)  ##问题的最优解
        # self.problem = 1  ##问题的编号

    def fvec(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        n = self.n
        m = n
        fvec = np.zeros((m,))

        for i in range(int(n/2)):
            fvec[2 * i] = 10 * (x[2 * i + 1] - x[2 * i] ** 2)
            fvec[2 * i + 1] = 1 - x[2 * i]
        return np.array(fvec)

    def jac(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        n = self.n
        m =  n
        jac = np.zeros((m, n))  # 预设空间，尽量降低时间空间复杂度

        for i in range(int(n/2)):
            jac[2 * i, 2 * i + 1] = 10
            jac[2 * i, 2 * i] = -20 * x[2 * i]
            jac[2 * i + 1, 2 * i + 1] = 0
            jac[2 * i + 1, 2 * i] = -1

        return np.array(jac)

    def fun(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        fvec = self.fvec(x)
        return np.linalg.norm(fvec) ** 2

    def hessian(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        n = self.n
        G = np.zeros((n, n))
        for i in range(int(n/2)):
            G[2 * i, 2 * i] = -400 * x[2 * i + 1] + 1200 * x[2 * i] ** 2 + 2
            G[2 * i, 2 * i + 1] = -400 * x[2 * i]
            G[2 * i + 1, 2 * i] = -400 * x[2 * i]
            G[2 * i + 1, 2 * i + 1] = 200
        return G

    def gfun(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        jac = self.jac(x)
        fvec = self.fvec(x)
        return 2 * np.dot(jac.T, fvec)

if __name__ == '__main__':
    x0 = np.array([[-1.2], [1]])
    test1=rosenbrock(n=2,x=[])
    fun=test1.fun
    gfun=test1.gfun
    hessian=test1.hessian
    x, val, k,nf , ng = dampnm_armijo(fun, gfun, hessian, x0)  # Armijo准则
    print('近似最优点：{}\n最优值：{}\n迭代次数：{}\n函数调用次数：{}\n梯度调用次数：{}'.format(x, val.item(), k, nf , ng))
    x, val, k ,nf , ng= dampnm_wolfe(fun, gfun, hessian, x0)  # wolfe准则
    print('近似最优点：{}\n最优值：{}\n迭代次数：{}\n函数调用次数：{}\n梯度调用次数：{}'.format(x, val.item(), k,nf,ng))
