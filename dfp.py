"""
the conjugate gradient algorithm

author:  fengyanlin@pku.edu.cn
"""

import numpy as np
import time
#import matplotlib.pyplot as plt
import argparse


def func(x):
    return x[0]**4 / 4 + x[1]**2 / 2 - x[0] * x[1] + x[0] - x[1]


def grad(x):
    return np.array([x[0]**3 - x[1] + 1, x[1] - x[0] - 1])


def backtrack(x, dx, a, b):
    f = func(x)
    df = grad(x)
    while func(x + dx) > f + a * dx.dot(df):
        dx = dx * b
    return dx


def dfp(x0, H0, func, grad, eps, a, b):
    i = 0
    history = []
    xk = x0
    gk = grad(xk)
    Hk = H0
    while True:
        if np.sqrt((gk**2).sum()) < eps:
            break
        else:
            i += 1
            f = func(xk)
            print('iter = {}   f = {:.6f}'.format(i, f))
            history.append(f)

        dk = -Hk.dot(gk)
        dx = backtrack(xk, dk, a, b)
        xk = xk + dx
        gk_1 = grad(xk)
        dg = gk_1 - gk
        Hk = Hk + np.outer(dx, dx) / dx.dot(dg) - np.outer(Hk.dot(dg), Hk.dot(dg)) / dg.dot(Hk.dot(dg))
        gk = gk_1
    return xk, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-b', '--beta', type=float, default=0.8)
    parser.add_argument('-eta', '--eta', type=float, default=1e-5)
    args = parser.parse_args()

    a, b, eta = args.alpha, args.beta, args.eta

    #fig = plt.figure()

    for j, x0 in enumerate([np.array([0., 0.]), np.array([1.5, 1.])]):
        x, history = dfp(x0, np.identity(2), func, grad, eta, a, b)
        print('x = {}'.format(x))
        #ax = fig.add_subplot('12{}'.format(j + 1))
        #ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    #fig.set_size_inches(9, 4)
    #fig.savefig('dfp.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
