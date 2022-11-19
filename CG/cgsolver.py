#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2022/11/18
import numpy as np

import convex1
import convex2
import penalty
import strongwolfe as sw


def getbeta(method,gk,gk_prev,dk_prev=None):
    '''function for beta'''
    if method == "HS":
        yk_prev=gk-gk_prev
        beta=np.dot(gk,yk_prev)/np.dot(dk_prev,yk_prev)
    elif method == "FR":
        beta=(np.linalg.norm(gk)/np.linalg.norm(gk_prev))**2
    elif method == "PRP":
        yk_prev = gk - gk_prev
        beta=np.dot(gk,yk_prev)/(np.linalg.norm(gk_prev)**2)
    elif method == "CD":
        beta=-np.linalg.norm(gk)**2/np.dot(dk_prev,gk_prev)
    elif method == "DY":
        yk_prev = gk - gk_prev
        beta=np.linalg.norm(gk)**2/np.dot(dk_prev,yk_prev)
    elif method == "PRP+":
        yk_prev = gk - gk_prev
        beta=np.dot(gk,yk_prev)/(np.linalg.norm(gk_prev)**2)
        beta=max(beta,0)
    return beta
def swolf_alpha(fun,grad,x0,d0,f0,g0):
    ''':return alpha,fcalls,gcalls'''
    res = sw.SLineSearch(fun, x0, d0, f0, g0, grad=grad, alpha=1, isdebug=False, options={'interp': 0})
    result=res.result
    return result.alpha,result.fcalls,result.gcalls
def CG(problem,beta_method,itermax,linesearch=swolf_alpha):
    '''return nf,ng,k,x,fk,flag
        flag:   0   算法成功
        flag:   1   强wolf失败
        flag:   2   CG迭代溢出
    '''
#--------------------------------------init----------------------------------------#
    funf,fung,x=problem.f,problem.g,problem.x               #get funf,fung,x0 from problem
    fk,gk=funf(x),fung(x)                                   #f0,g0
    nf,ng=1,1                                               #函数值，梯度计算次数
    dk=-gk                                                  #d0=-g0
    epsilon=1e-5                                           #精度
    flag=0                                                  #意味着成功与否
#--------------------------------------CG------------------------------------------#
    for k in range(itermax):
        if np.linalg.norm(gk)<epsilon:                      #if ||gk||<epsilon: return
            return nf,ng,k,x,fk,flag
        alphak,fk,gk=linesearch(funf,fung,x,dk,fk,gk)       #calculate alphak
        nf,ng=nf+fk,ng+gk                                   #update nf,ng
        flag=1 if alphak is None else 0                     #alphak计算失败改变flag表示，强wolf失败
        flag=2 if k==itermax-1 and flag!=1 else flag        #判断是否迭代溢出，改变flag
        if flag == 0:                                       #如果flag 没有变成失败状态才继续做
            x=x+alphak*dk                                   #update x
            fk,nf=funf(x),nf+1                              #calculate new fk meanwhile upadate nf
            gk_prev=gk                                      #save previous gk in order to calculate beta
            gk,ng=fung(x),ng+1                              #calculate new gk meanwhile upadate ng
            dk=-gk+getbeta(beta_method,gk,gk_prev,dk)       #updata dk by eq:dk=-gk+beta*dk_prev
        else:
            return nf,ng,k,x,fk,flag                                #结束，返回信息
def print_infor(nf, ng, k, x, fk,flag,problem,beta_method):
    name=type(problem).__name__
    n=problem.n

    if flag == 2:
        print('[n={}]{}_{}:\t经过{}次迭代,溢出'.format(n,name,beta_method,k))
    elif flag == 0:
        print('[n={}]{}_{}_{}:'.format(n,name,beta_method),end='\t')
        print('k={}\tf={}\tnf={}\tng={}'.format(k, np.around(fk,1), nf, ng))
    elif flag == 1:
        print('[n={}]{}_{}:\t经过{}次迭代,强wolf失败'.format(n, name, beta_method, k))

if __name__ == '__main__':
#--------------------------preparation---------------------#
    beta_meathod_list=['PRP','FR','PRP+','CD','DY','HS']
    n_list=[1e2,1e3,1e4]
    problem_class_list=[convex1.Convex1,convex2.Convex2,penalty.Penalty]
#-----------------------create_problems-------------------#
    problem_list=[]
    for n in n_list:
        for problem in problem_class_list:
            problem_list.append(problem(int(n),x=[]))
#----------------------CGsolver---------------------------#
    for beta_method in beta_meathod_list:
        for problem in problem_list:
            itermax=problem.n*20
            nf, ng, k, x, fk,flag=CG(problem,beta_method,itermax=itermax)
            print_infor(nf, ng, k, x, fk,flag,problem,beta_method)



