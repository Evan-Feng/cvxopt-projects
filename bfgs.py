"""
the BFGS algorithm

author:  fengyanlin@pku.edu.cn
"""

import numpy as np
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse


def func(x):
    return (3 - x[0])**2 + 7 * (x[1] - x[0]**2)**2


def grad(x):
    return np.array([2 * (x[0] - 3) - 28 * x[0] * (x[1] - x[0]**2), 14 * (x[1] - x[0]**2)])


def backtrack(x, dx, a, b):
    f = func(x)
    df = grad(x)
    while func(x + dx) > f + a * dx.dot(df):
        dx = dx * b
    return dx


def bfgs(x0, H0, func, grad, eps, a, b):
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
        Hk = Hk + (1 + dg.dot(Hk.dot(dg)) / dg.dot(dx)) * (np.outer(dx, dx)) / dx.dot(dg) - \
            (np.outer(Hk.dot(dg), dx) + np.outer(dx, Hk.dot(dg))) / dg.dot(dx)
        gk = gk_1
    return xk, history


def dfp(x0, H0, func, grad, eps, a, b):
    i = 0
    history = []
    xk = x0
    gk = grad(xk)
    Hk = H0
    while True:
        if np.sqrt((gk**2).sum()) < eps or i > 50:
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
        Hk = Hk + np.outer(dx, dx) / dx.dot(dg) - np.outer(Hk.dot(dg), Hk.dot(dg)) / (dg.dot(Hk.dot(dg)) + 1e-20)
        gk = gk_1
    return xk, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-b', '--beta', type=float, default=0.8)
    parser.add_argument('-eta', '--eta', type=float, default=1e-5)
    args = parser.parse_args()

    a, b, eta = args.alpha, args.beta, args.eta

    fig = plt.figure()

    x0 = np.array([0, 0])

    # DFP
    x, history = dfp(x0, np.identity(2), func, grad, eta, a, b)
    ax = fig.add_subplot('121')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    # BFGS
    x, history = bfgs(x0, np.identity(2), func, grad, eta, a, b)
    ax = fig.add_subplot('122')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    fig.set_size_inches(9, 4)
    fig.savefig('bfgs.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
