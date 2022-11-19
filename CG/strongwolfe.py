#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
---------------------------------------
@ProjectName: optimizor
@FileName  :strongwolfe.py
@Time      :2022/11/18 22:59
@Author    :zhangjw 
---------------------------------------
"""
from warnings import warn

import numpy as np

###############################################
###线性搜索需要的参数默认值，
ls_options = {'method': 1,  # 线性搜索算法，默认值：
              # 1： armijo搜索，2：wolfe搜索，3：强wolfe搜索，4：scipy的强wolfe搜索。
            'alpha0': 3,  # 初始步长选择，默认
              #'alpha0': 1，alpha0 =1,
              #'alpha0': 2, alpha0 = 2 * (f0 - f0_old)/derphi0
              #'alpha0': 3, alpha0 = min(1,1.01 * alpha_2)
            'interp': True,  # 插值选择参数，默认利用插值方法计算步长
            'c1': 1e-4, 'c2': 0.9,  ##线性搜索搜索参数
            'pho': 0.2,  # 无插值方法时，压缩因子
            'alpha_max': 1e2,  # 最大步长
            'steptol': 1e-9,  ##允许的最小步长
            'maxiter': 20  # 线性搜索最大迭代次数
                  }
##########################################################


class SLineSearch:
    #####线性搜索参数说明参见
    ls_options = {'method': 1, 'alpha0': 3,
                  'interp': 0, 'c1': 1e-4, 'c2': 0.9, 'pho': 0.2, 'alpha_max': 1e2,
                  'steptol': 1e-9, 'maxiter': 20}

    def __init__(self, fun, x0, d0, f0, g0, grad=None, f0_old=None, derphi0=None,
                 alpha=1, isdebug=False, options={}, scale=False, doPlot=False, **kwargs):

        ##检查输入数据的合理性
        # fun, x0, d0, f0, g0, grad = Check.check(fun, x0, d0, f0, g0, grad)

        res = self.dolinesearch(fun=fun, x0=x0, d0=d0, f0=f0, g0=g0,
                                grad=grad, f0_old=f0_old, derphi0=derphi0,
                                alpha=alpha, isdebug=isdebug, options=options,
                                scale=scale, **kwargs)
        if doPlot:
            self.plot(fun=fun, x0=x0, d0=d0, f0=f0, g0=g0, grad=grad,
                      derphi0=derphi0, res=res, **kwargs)
        self.result = res

    def dolinesearch(self, fun, x0, d0, f0, g0,
                     grad=None, f0_old=None, derphi0=None,
                     alpha=1, isdebug=False, options={},
                     scale=False, **kwargs):


        """
        Find alpha that satisfies strong Wolfe conditions.

           alpha > 0 is assumed to be a descent direction.

           输入参数说明
           ----------
           fun : callable, n维目标函数：R^n -> R.
           grad : callable， 目标函数的梯度函数，R^n -> R^n
           x0 : array,n维向量，当前迭代点
           d0 : array,n维向量，下降的搜索方向
           f0 : float, 当前点的目标函数值
           g0 : array,n维向量，当前点的梯度函数值
           derphi0：float, f 在x0处沿搜索方向d0的方向导数 = g0'*d0


           返回结果说明
           -------
           返回一个字典对象lineresult,包括下面键值对：
           lineresult['alpha']=alpha_star,  满足强wolfe条件的步长
           lineresult['f1']=phi_star,  满足强wolfe条件时计算的下一个迭代点的函数值
           lineresult['g1']=phi_star,  满足强wolfe条件时计算的下一个迭代点的梯度值
           lineresult['fcalls']=fcalls[0],  目标函数调用次数
           lineresult['gcalls']=gcalls[0],  梯度函数调用次数
           lineresult['nit']=i,  线性搜索的迭代次数
           lineresult['success']=(flag == 0),  线性搜索执行状态
           lineresult['message']=msg,  线性搜索执行信息提示


           注
           -----
           1. 该线性算法的详细说明参见See Wright and Nocedal, 'Numerical Optimization',
           1999, pp. 59-61.

           2. 该算法的编程借用了scipy.optimize部分内容。

        """
        ##更新线性搜索的参数
        SLineSearch.checkoption(options, self.ls_options)
        amax = self.ls_options['alpha_max']
        c1 = self.ls_options['c1']
        c2 = self.ls_options['c2']
        maxiter = self.ls_options['maxiter']
        init = self.ls_options['alpha0']
        # interp = self.ls_options['interp']
        # steptol = self.ls_options['steptol']



        ###装饰目标函数和梯度函数，其作用是每调用一次函数值或梯度值计算，
        ###函数调用次数fcalls, 和梯度调用次数自动增加一次。
        fcalls, f = SLineSearch.wrap_function(fun, **kwargs)
        gcalls, grad = SLineSearch.wrap_function(grad, **kwargs)

        def phi(alpha, **kwargs):
            return f(x0 + alpha * d0, **kwargs)

        def derphi(alpha, **kwargs):
            g = grad(x0 + alpha * d0, **kwargs)
            return np.dot(g, d0)

        ######选择初始步长
        if derphi0 is None:
            derphi0 = np.dot(g0, d0)  # 计算一元函数phi在0处的导数：g0*p0
        if f0_old is None:
            f0_old = f0 + np.linalg.norm(g0) / 2
        ###计算初始步长,设置初始步长搜索区间[alpha0, alpha1]

        alpha0 = 0
        phi0 = f0
        phi_a0 = f0
        derphi_a0 = derphi0

        ####计算初始步长
        alpha1 = self.compute_init(init=init, f0=f0, f0_old=f0_old, derphi0=derphi0)


        if amax is not None:
            alpha1 = min(alpha1, amax)
        phi_a1 = phi(alpha1)
        # derphi_a1 = derphi(alpha1) #evaluated below
        trace = []



        """强wolfe搜索的实施过程说明：
        1：初始步长的选择：
        利用f0, f0_old, f'0 = np.dot(g0, d0),建立目标函数的二次近似，其最小值为初始步长
        alpha1 = 2*(f0 - f0_old)/f'0

        2:算法包括两个阶段。第一阶段，给定试探步长alpha1, 不断增加，直到：
        i)找到满足搜索条件的步长alpha_star,
        ii)或者包括满足条件的步长区间(alpha0, alpha1)

        3: 在满足条件的步长区间(alpha0, alpha1)里，利用zoom函数不断缩短区间长度，
        直到找到满足搜索条件的步长alpha_star。

        4：序列{alpha_i}是单调递增

        5：在下面三种情况下，区间(alpha0, alpha1)包括满足条件的步长
        i): alpha1步满足armijo条件
        ii): phi(alpha1) > phi(alpha0)
        上面两种情况的含解区间(alpha0, alpha1)
        iii): derphi(alpha1) >= 0
        此时含解区间(alpha1, alpha0)，此时利用插值多项式外插，或者简单alpha2 = c * alpha1,
        计算下一个试探步长alpha2.

        无论何种情况，试探步长应在有限次迭代中快速逼近最大步长alphamax.

        6: 关于zoom(alo,ahi)的说明
        i): (alo,ahi)是含解区间
        ii): alo是目前为止所有的计算中满足armijo条件，且函数值最小的步长
        iii): 选择ahi满足derphi(alo)(ahi -alo)<0

        zoom(alo, ahi):
        1: 在区间(alo, ahi)里插值计算步长alpha
        if phi(alpha) > phi0 + c1 * alpha * derphi0 or phi(alpha) >= phi(alo)
            ahi = alpha
        else；
            计算derphi(alpha)
            if np.abs(derphi(alpha)) <= np.abs(c2 * derphi0)
                 alpha_star = alpha 
                 return
            if derphi(alpha)(ahi -alo) >= 0
                 ahi = alo
            alo = alpha


        """

        flag = 0
        unitstep = False
        for i in range(maxiter):
            ##case1:  试探步长趋于0，算法失败
            if alpha1 == 0 or (amax is not None and alpha0 == amax):
                # alpha1 == 0: This shouldn't happen. Perhaps the increment has
                # slipped below machine precision?
                alpha_star = None
                phi_star = phi0
                phi0 = f0_old
                derphi_star = None
                flag = -1

                if alpha1 == 0:
                    msg = '满足强wolfe搜索的步长趋于0，线性搜索失败'
                else:
                    msg = "满足强wolfe搜索的步长大于指定的最大步长" + \
                          "线性搜索失败: %s" % amax

                warn(msg, LineSearchWarning)
                break

            ##case2: 不满足armijo条件，或者 phi_a1 >= phi_a0 且 非第一次迭代时，
            # 此时，区间(alpha0, alpha1)包含满足wolfe搜索条件的步长。
            ##进行zoom，找到满足wolfe条件的步长为止。
            not_first_iteration = i > 0
            if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
                    ((phi_a1 >= phi_a0) and not_first_iteration):
                alpha_star, phi_star, derphi_star = \
                    self._zoom(alpha0, alpha1, phi_a0,
                               phi_a1, derphi_a0, phi, derphi,
                               phi0, derphi0, c1, c2)
                break

            ##此时步长alpha1满足armijo条件
            derphi_a1 = derphi(alpha1)
            # case3: 找到满足wolfe条件的步长
            if (abs(derphi_a1) <= -c2 * derphi0):
                ##如果曲率条件满足，返回符合搜索 步长
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                if alpha_star == 1:
                    unitstep = True
                break
            ###case4：此时，步长满足armijo条件，但不满足曲率条件
            # 此时，区间(alpha1, alpha0)包含满足wolfe搜索条件的步长。
            ##进行zoom，找到满足wolfe条件的步长为止。
            if (derphi_a1 >= 0):  ###步长太小
                alpha_star, phi_star, derphi_star = \
                    self._zoom(alpha1, alpha0, phi_a1,
                               phi_a0, derphi_a1, phi, derphi,
                               phi0, derphi0, c1, c2)
                break
            ###此时，满足armijo条件，但不满足曲率条件且 derphi_a1 < 0
            alpha2 = 2 * alpha1  # increase by factor of two on each iteration
            if amax is not None:
                alpha2 = min(alpha2, amax)
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            phi_a1 = phi(alpha1)
            derphi_a0 = derphi_a1

        else:
            # stopping test maxiter reached
            flag = -2
            alpha_star = None
            phi_star = phi_a1
            derphi_star = None
            warn('The line search algorithm did not converge', LineSearchWarning)

        if flag == 0:
            msg = f'满足强wolfe搜索的步长{alpha_star}'

        lineresult = OptimizeResult(f1=phi_star, g1=derphi_star, fcalls=fcalls[0],
                                    gcalls=gcalls[0], status=flag,
                                    success=(flag == 0), message=msg, alpha=alpha_star,
                                    nit=i, unitstep=unitstep)

        if __name__ == "__main__":
            print("%s%s" % ("计算结果: " if flag >= 0 else "警告: ", msg))
            print(f'线性搜索参数c1: {c1}')
            print(f'线性搜索参数c2: {c2}')
            print("         步长: %f" % alpha_star)
            print("         迭代次数: %d" % i)
            print("         函数值计算次数: %d" % fcalls[0])
            print("         函数值计算次数: %d" % gcalls[0])

        return lineresult

        # return alpha_star, phi_star, derphi_star, f0_old, fcalls, gcalls, flag
    #############################################################
    @staticmethod
    def checkoption(options, defoptions):
        for key in options.keys():
            if key in defoptions.keys():
                defoptions[key] = options[key]
    #################################################################
    @staticmethod
    def wrap_function(function, *args):
        ncalls = [0]
        if function is None:
            return ncalls, None

        def function_wrapper(*wrapper_args):
            ncalls[0] += 1
            return function(*(wrapper_args + args))

        return ncalls, function_wrapper
    #####################################################################
    @staticmethod
    def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
              phi, derphi, phi0, derphi0, c1, c2):
        """Zoom stage of approximate linesearch satisfying strong Wolfe conditions.

       Notes
        -----
        Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
        'Numerical Optimization', 1999, pp. 61.

        """

        maxiter = 10
        i = 0
        delta1 = 0.2  # cubic interpolant check
        delta2 = 0.1  # quadratic interpolant check
        phi_rec = phi0  ##当前点函数值
        a_rec = 0  ##当前步长
        while True:
            # interpolate to find a trial step length between a_lo and
            # a_hi Need to choose interpolation here. Use cubic
            # interpolation and then if the result is within delta *
            # dalpha or outside of the interval bounded by a_lo or a_hi
            # then use quadratic interpolation, if the result is still too
            # close, then use bisection

            dalpha = a_hi - a_lo  ## 搜索区间
            if dalpha < 0:
                a, b = a_hi, a_lo
            else:
                a, b = a_lo, a_hi

            # minimizer of cubic interpolant
            # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
            #
            # if the result is too close to the end points (or out of the
            # interval), then use quadratic interpolation with phi_lo,
            # derphi_lo and phi_hi if the result is still too close to the
            # end points (or out of the interval) then use bisection

            if (i > 0):
                cchk = delta1 * dalpha
                a_j = SLineSearch._cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                                            a_rec, phi_rec)
            if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
                qchk = delta2 * dalpha
                a_j = SLineSearch._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                    a_j = a_lo + 0.5 * dalpha

            # Check new value of a_j

            phi_aj = phi(a_j)
            if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj
            else:
                derphi_aj = derphi(a_j)
                if abs(derphi_aj) <= -c2 * derphi0:
                    ##计算得到满足条件的步长
                    a_star = a_j
                    val_star = phi_aj
                    valprime_star = derphi_aj
                    break
                if derphi_aj * (a_hi - a_lo) >= 0:
                    phi_rec = phi_hi
                    a_rec = a_hi
                    a_hi = a_lo
                    phi_hi = phi_lo
                else:
                    phi_rec = phi_lo
                    a_rec = a_lo
                a_lo = a_j
                phi_lo = phi_aj
                derphi_lo = derphi_aj
            i += 1
            if (i > maxiter):
                # Failed to find a conforming step size
                a_star = None
                val_star = None
                valprime_star = None
                break
        return a_star, val_star, valprime_star

    @staticmethod
    def compute_init(init, f0, f0_old, derphi0):
        """计算初始步长：计算策略见文档说明
        init = 0: 初始步长 = 1
        init  = 1, alpha = 2 * (f0 - f0_old) / derphi0
        followed Nocedal: Numerical Optimization pp58
        init = 2, alpha = min(1, 1.01 * alpha)
        """
        if init == 0 or f0_old is None:
            alpha = 1

        else:
            alpha = 2 * (f0 - f0_old) / derphi0

            if init == 2:
                alpha = min(1, 1.01 * alpha)

            if alpha < 0:
                alpha = 1
        return alpha

    @staticmethod
    def _cubicmin(a, fa, fpa, b, fb, c, fc):
        """
        Finds the minimizer for a cubic polynomial that goes through the
        points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

        If no minimizer can be found, return None.

        """
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

        with np.errstate(divide='raise', over='raise', invalid='raise'):
            try:
                C = fpa
                db = b - a
                dc = c - a
                denom = (db * dc) ** 2 * (db - dc)
                d1 = np.empty((2, 2))
                d1[0, 0] = dc ** 2
                d1[0, 1] = -db ** 2
                d1[1, 0] = -dc ** 3
                d1[1, 1] = db ** 3
                [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                                fc - fa - C * dc]).flatten())
                A /= denom
                B /= denom
                radical = B * B - 3 * A * C
                xmin = a + (-B + np.sqrt(radical)) / (3 * A)
            except ArithmeticError:
                return None
        if not np.isfinite(xmin):
            return None
        return xmin

    @staticmethod
    def _quadmin(a, fa, fpa, b, fb):
        """
        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at a of fpa.

        """
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        with np.errstate(divide='raise', over='raise', invalid='raise'):
            try:
                D = fa
                C = fpa
                db = b - a * 1.0
                B = (fb - D - C * db) / (db * db)
                xmin = a - C / (2.0 * B)
            except ArithmeticError:
                return None
        if not np.isfinite(xmin):
            return None
        return xmin

    @staticmethod
    def plot(fun, x0, d0, f0, g0, grad, derphi0, res, **kwargs):
        pass


##########################################
class LineSearchWarning(RuntimeWarning):
    pass
#########################################
class OptimizeResult(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

###################################################
####测试问题

class Rosen:
    """f = sigma(100.0*(x[1] - x[0]**2.0)**2.0 + (1 - x[0])**2.0)
         = sigma (f1 ** 2+ f2 ** 2)
    % Problem no. 1
    % Dimensions -> n=2, m=2
    % Standard starting point -> x=(-1,-1)
    % Minima -> f=0 at (1,1)
    % Revised on 03/22/2021 by zhangjw
    """

    def __init__(self, *args):
        x = [-1, -1]
        self.x = np.asarray(x).flatten()
        self.xopt = [1, 1]

    def fvec(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        fvec = []
        fvec.append(10 * (x[1] - x[0] ** 2))
        fvec.append(1 - x[0])
        return np.array(fvec)

    def jac(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()

        jac = []
        jac.append([-20 * x[0], 10])
        jac.append([-1, 0])
        return np.array(jac)

    def f(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        fvec = self.fvec(x)
        fvec = np.asarray(fvec)
        return np.linalg.norm(fvec) ** 2

    def g(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        jac = self.jac(x)
        fvec = self.fvec(x)
        return 2 * np.dot(jac.T, fvec)

    def G(self, x=None):
        if x is None:
            x = self.x
        else:
            x = np.asarray(x).flatten()
        x, y = x.tolist()
        return np.array([[2 * (600 * x ** 2 - 200 * y + 1), -400 * x],
                         [-400 * x, 200]])

if __name__ == "__main__":
    x0 = np.array([-1, -1])
    fun = Rosen().f
    grad = Rosen().g

    f0 = fun(x0)

    g0 = grad(x0)
    d0 = -g0

    print('----------nocedal--strong -wolfe')
    res = SLineSearch(fun, x0, d0, f0, g0, grad=grad, alpha=1, isdebug=False, options={'interp': 0})
    print(res.result)
    print()
