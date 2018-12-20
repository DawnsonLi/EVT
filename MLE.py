#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#Maximum likelihood estimation
import numpy as np
from scipy.optimize import root
class CALError(Exception):
    pass
def Gamma(x, Yi):
    vx = 0
    Nt = len(Yi)
    for i in range(1, Nt+1):
        try:
            if 1 + x*Yi[i-1] <=0:
                raise CALError()
            vx = vx + np.log(1 + x*Yi[i-1])
        except CALError:
            pass
    vx = vx/Nt
    return vx

def Delta(x, gamma):
    try:
        if x < 0.00000001:
            raise CALError()
        return gamma/x
    except CALError:
            pass
def zq(q, gamma,delta,n, Nt,t):
    try:
        if gamma < 0.00000001:
            raise CALError()
    except CALError:
            pass
    zq = t
    tmp = (q*n/Nt)**(-gamma)-1
    zq += tmp*(delta/gamma)
    return zq
def f(x, Yi):
    x = float(x[0])
    Nt = len(Yi)
    ux = 0
    for i in range(1, Nt+1):
        if (1+ x*Yi[i-1]) < 0.00000001:
            return 1
        ux = ux + 1.0/(1+ x*Yi[i-1])
    ux = ux/Nt
    vx = 0
    for i in range(1, Nt+1):
        if 1 + x*Yi[i-1] <= 0:
            return 1
        vx = vx + np.log(1 + x*Yi[i-1])
    vx = vx/Nt
    vx = vx + 1
    return [
        ux * vx -1
        ]

def choose_zq(Yi, t, zq_lst):
    l = [i + t for i in Yi]
    m = np.median(l)
    minus = 1000000
    ans = zq_lst[0]
    for zq in zq_lst:
        tmp = abs( zq - m)
        if tmp < minus:
            minus = tmp
            ans = zq
    return ans

def MLE_get_zq(Yi, q, n, t):
    Nt = len(Yi)
    Ym = np.max(Yi)
    low = -1.0/Ym
    ym = np.min(Yi)+0.1
    high = 2*((np.mean(Yi) - ym)/ym**2)
    #print [low, high]
    zq_lst = []
    for i in range(5):
        guess = low + i*(high-low)/5 
        guess = float(guess)
        #print "guess", guess
        sol = root(f, guess, Yi)
        #print sol
        if sol['success'] == True:
            x =  sol['x'][0]
            try:
                gamma =  Gamma(x, Yi)
                delta = Delta(x, gamma)
                #print gamma, delta
                zq_lst.append( zq(q, gamma,delta,n, Nt,t) )
            except:
                pass
        #print i, " ********** is over"
    return choose_zq(Yi, t, zq_lst) 

def MOM_get_zq(Yi, q, n, t):
    avg = np.mean(Yi)
    var = np.var(Yi)
    Nt = len(Yi)
    gamma = 0.5*(avg**2/var + 1)
    delta = 0.5*avg*(avg**2/var +1)
    return zq(q, gamma,delta,n, Nt,t)

Yi = [5,10,2,4,8,100,102,3,4,8,100,102,3]
q = 0.96
n = 1000
t = 100 #t是Y序列的q分位数
print MLE_get_zq(Yi, q, n, t)
print MOM_get_zq(Yi, q, n, t)