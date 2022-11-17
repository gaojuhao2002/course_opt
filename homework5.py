#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/10/31

import numpy as np
import diagonal_4
import extented_rosenbrock
import scipy.optimize as opt
import os
import scipy
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#线性搜索
class LineSearch:
    @staticmethod
    def armijo(func, gunc, x0, d0, f0, g0,i = 0, nf = 0,ng = 0):
        sigma1 = 0.2
        pho = 0.5  # 压缩因子
        itermax = 20  # 允许的最大迭代次数
        alpha = 1  ##初始单位步
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
                print('armijo搜索迭代次数溢出')
                return alpha, nf, ng
    @staticmethod
    def wolfe(func, gunc, x0, d0, f0, g0,i=0,nf=0,ng=0,sigma1=0.01,sigma2=0.9):
        pho = 0.5  # 压缩因子
        pho1 = 0.1
        itermax = 20  # 允许的最大迭代次数
        alpha = 1  ##初始单位步
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
                print('wolf搜索迭代次数溢出')
                return alpha, nf, ng

#阻尼牛顿
def Damp_Newton(getf,getg,getG,x0,linesearch,itermax=1000):
    epsilon = 1e-8
    x=np.asarray(x0).flatten()
    k = 0
    #计算初始值
    f = getf(x)
    g = getg(x)
    nf = 1
    ng = 1

    for k in range(itermax):

        #如果小于满足精度条件结束迭代
        if np.linalg.norm(g) < epsilon:
            return x, f, g, nf, ng,k

        #不满足精度条件
        dk = -np.dot(np.linalg.inv(getG(x)), g.T)#计算牛顿方向
        alphak,nfk,ngk=linesearch(getf,getg,x,dk,f,g)#（func, gunc, x0, d0, f0, g0）线性搜索计算步长等
        nf,ng=nf+nfk,ng+ngk#更新函数值和梯度的计算次数
        x=x+alphak*dk#更新迭代点
        f,g=getf(x),getg(x)#更新函数值和梯度值
        nf, ng = nf + 1, ng + 1

        #溢出
        if k==itermax-1:
            print('阻尼牛顿迭代溢出')
            break
    return x, f, g, nf, ng, k

#BFGS
def Quasi_Newton(getf,getg,x0,linesearch,eps=1e-8,method='BFGS',itermax=1000):
    '''
    参数说明
    :获取函数值 getf:
    :获取梯度 getg:
    :初始点 x0:
    :线性搜索方法 linesearch:
    :如果是armijo为了保证B正定确定修正的精度 eps:
    :指定方法是BFGS还是DFS,由于只是差异于B的公式上所以统一在此 method（缺省值BFGS）:
    :return:
    x, f, g, nf, ng,k=(迭代点，函数值，梯度，函数值计算次数，梯度计算次数，迭代次数)
    '''
    epsilon = 1e-8
    len_x=len(x0)

    k = 0
    #计算初始值
    x=np.asarray(x0).flatten()
    B0=np.eye(len_x)#初始对称正定矩阵B采用单位阵
    f = getf(x)
    g = getg(x)
    f=np.asarray(f).flatten()
    g=np.asarray(g).flatten()
    nf = 1
    ng = 1
    for k in range(itermax):
        # 如果小于满足精度条件结束迭代
        if np.linalg.norm(g) < epsilon:
            return x, f, g, nf, ng,k
        # 不满足精度条件
        if k==0:
            B=B0
        else:
            yy = yk.reshape((len_x,1))*yk
            ys = np.dot(yk, sk)
            if method == 'BFGS':
                Bs = np.dot(B, sk.T)
                sBs = np.dot(np.dot(sk,B),sk)
                # 如果采用armijo，ys可能小于0（实际做用一个精度），此时用bfgs修正
                B = B - ( 1.0*Bs.reshape((len_x,1))*Bs/ sBs) + 1.0*(yy / ys) if ys > eps else B
            elif method == 'DFP':
                yst_chu_ys=(yk.reshape((len_x, 1))*sk)/ys
                one_jian_yst_chu_ys=np.eye(len_x)-yst_chu_ys
                # 如果采用armijo，ys可能小于0（实际做用一个精度），此时用dfp修正
                B=np.dot(np.dot(one_jian_yst_chu_ys,B),one_jian_yst_chu_ys.T)+yy/ys if ys>eps else B
        try:
            dk=scipy.linalg.solve(B,-g.T)
        except:
            print('方向计算复杂度过高')
            break
        # dk = -np.dot(np.linalg.inv(B),g.T) # 计算牛顿方向
        alphak,nfk,ngk=linesearch(getf,getg,x,dk,f,g)#线性搜索计算步长等
        nf,ng=nf+nfk,ng+ngk#更新函数值和梯度的计算次数
        xnew=x+alphak*dk#计算迭代点
        f, gnew = getf(xnew), getg(xnew) # 计算函数值和梯度值
        nf,ng=nf+1,ng+1
        sk=xnew-x
        yk=gnew-g
        #更新迭代点
        x,g=xnew,gnew
        #溢出信息，全部封装到print_infor函数
        # if k==itermax-1:
        #     print('BFGS算法迭代溢出') if method=='BFGS' else print('DFP算法迭代溢出')
        #     break
    return x, f, g, nf, ng, k

def print_infor(name,linsearch,method,n,k,f, nf, ng,resultx,itermax):
    if k==itermax-1:
        print('[n={}]{}_{}_{}:\t经过{}次迭代,溢出'.format(n,name, linsearch, method,k))
    else:
        print('[n={}]{}_{}_{}:'.format(n,name,linsearch,method),end='\t')
        if n<10:
          print('k={}\tf={}\tnf={}\tng={}\tx={}'.format(k, np.around(f,1), nf, ng,resultx))
        else:
          print('k={}\tf={}\tnf={}\tng={}'.format(k, np.around(f,1), nf, ng))
          print('x={}'.format(np.around(resultx,1)))

if __name__ == '__main__':

    # #分别创建测试的对象（n为2，20，200）

    # #初始点（-1.2，1，...，-1.2，1）
    target_rosenbrock_2=extented_rosenbrock.Extend_Rosenbrock(2,x=[])
    target_rosenbrock_20 = extented_rosenbrock.Extend_Rosenbrock(20,x=[])
    target_rosenbrock_200 = extented_rosenbrock.Extend_Rosenbrock(200,x=[])
    # #初始点（1，1，...1，1）
    target_diagonal4_2=diagonal_4.Diagonal_4(2,x=[])
    target_diagonal4_20 = diagonal_4.Diagonal_4(20,x=[])
    target_diagonal4_200 = diagonal_4.Diagonal_4(200,x=[])

    #保存到列表
    target_list=[target_rosenbrock_2,target_rosenbrock_20,target_rosenbrock_200,
                 target_diagonal4_2,target_diagonal4_20,target_diagonal4_200]
    linsearch_list=[LineSearch.armijo,LineSearch.wolfe]
    method_list = ['BFGS','DFP']

    for target in target_list:
        itermax = target.n * 2000
        # itermax=40000
        for linesearch in linsearch_list:
            #阻尼牛顿
            x, f, g, nf, ng, k = Damp_Newton(target.f, target.g, target.G,
            target.x,linesearch=linesearch,itermax=itermax)
            print_infor(target.name,linesearch.__name__,'DampNewton',target.n,k,f,nf,ng,resultx=x,itermax=itermax)
            #拟牛顿（bfgs+dfp）
            for method in method_list:
                x, f, g, nf, ng, k = Quasi_Newton(target.f, target.g, target.x,
                method=method,linesearch=linesearch,itermax=itermax)
                print_infor(target.name,linesearch.__name__,method,target.n,
                            k,f,nf,ng,resultx=x,itermax=itermax)
            print('')#每次换一个线性搜索就换行



    print('-------------------------------------------scipy------------------------------------------------')
    for target in target_list:
        res=opt.minimize(fun=target.f,x0=target.x.reshape(target.n,),jac=target.g,
                         method='BFGS',options={'maxiter':1000,'disp':False})
        k,f,nf,ng=res.nit,res.fun,res.nfev,res.njev
        print('[n={}]BFGS_{}_scipy:\tk={}\tf={}\tnf={}\tng={}'.format(target.n,target.name,
                                                              k,f,nf,ng))
        print('')
    print('-----------------------------------------NewtonCG----------------------------------------------')
    for target in target_list:
        res = opt.minimize(fun=target.f, x0=target.x.reshape(target.n, ), jac=target.g,
                           method='Newton-CG', options={'maxiter': 1000, 'disp': False})
        k, f, nf, ng = res.nit, res.fun, res.nfev, res.njev
        print('[n={}]Newton_{}:\tk={}\tf={}\tnf={}\tng={}'.format(target.n, target.name,
                                                                      k, f, nf, ng))
        print('')




