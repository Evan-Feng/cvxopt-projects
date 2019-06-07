"""
the conjugate gradient algorithm

author:  fengyanlin@pku.edu.cn
"""

import numpy as np
from scipy.linalg import null_space
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse


def func(x):
    return (x * np.log(x)).sum()


def grad(x):
    return np.log(x) + 1


def backtrack(x, dx, a, b):
    f = func(x)
    df = grad(x)
    while ((x + dx) <= 0).any():
        dx = dx * b
    while func(x + dx) > f + a * dx.dot(df):
        dx = dx * b
    return dx


def proj_grad(x0, A, B, func, grad, eps, a, b):
    """
    Projected gradient method.
    """
    i = 0
    history = []
    xk = x0
    gk = grad(xk)
    P = np.identity(x0.shape[0]) - A.T.dot(np.linalg.pinv(A.dot(A.T))).dot(A)
    while True:
        if np.sqrt(((P.dot(gk))**2).sum()) < eps:
            break
        else:
            i += 1
            f = func(xk)
            print('[PG] iter = {}   f = {:.6f}'.format(i, f))
            history.append(f)

        dk = P.dot(-gk)
        dx = backtrack(xk, dk, a, b)
        xk = xk + dx
        gk = grad(xk)
    return xk, history


def elim_eq(x0, A, B, func, grad, eps, a, b):
    """
    Eliminate equality contraint algorithm.
    """
    def backtrack_1(x, dx, a, b):
        f = func_1(x)
        df = grad_1(x)
        while ((F.dot(x + dx) + x0) <= 0).any():
            dx = dx * b
        while func_1(x + dx) > f + a * dx.dot(df):
            dx = dx * b
        return dx

    i = 0
    history = []
    P = np.identity(x0.shape[0]) - A.T.dot(np.linalg.pinv(A.dot(A.T))).dot(A)
    F = null_space(A)
    zk = np.zeros(F.shape[1])
    func_1 = lambda z: func(F.dot(z) + x0)
    grad_1 = lambda z: F.T.dot(grad(F.dot(z) + x0))
    gk = grad_1(zk)
    while True:
        if np.sqrt((gk**2).sum()) < eps:
            break
        else:
            i += 1
            f = func_1(zk)
            print('[EE] iter = {}   f = {:.6f}'.format(i, f))
            history.append(f)

        dk = -gk
        dx = backtrack_1(zk, dk, a, b)
        zk = zk + dx
        gk = grad_1(zk)
    xk = F.dot(zk) + x0
    return xk, history


def dual_approach(x0, A, B, func, grad, eps, a, b):
    """
    Dual approach.
    """
    def backtrack_1(x, dx, a, b):
        f = func_1(x)
        df = grad_1(x)
        while func_1(x + dx) > f + a * dx.dot(df):
            dx = dx * b
        return dx

    i = 0
    history = []
    P = np.identity(x0.shape[0]) - A.T.dot(np.linalg.pinv(A.dot(A.T))).dot(A)
    F = null_space(A)
    zk = np.random.randn(A.shape[0])
    func_1 = lambda z: z.dot(B) + np.exp(-A.T.dot(z) - 1).sum()
    grad_1 = lambda z: B - A.dot(np.exp(-A.T.dot(z) - 1))
    gk = grad_1(zk)
    while True:
        if np.sqrt((gk**2).sum()) < eps:
            break
        else:
            i += 1
            f = func_1(zk)
            print('[DU1] iter = {}   f = {:.6f}'.format(i, f))
            history.append(f)

        dk = -gk
        dx = backtrack_1(zk, dk, a, b)
        zk = zk + dx
        gk = grad_1(zk)
    d_opt = -f
    history_dual = [-x for x in history]

    def backtrack_2(x, dx, a, b):
        f = func_1(x)
        df = grad_1(x)
        while ((x + dx) <= 0).any():
            dx = dx * b
        while func_1(x + dx) > f + a * dx.dot(df):
            dx = dx * b
        return dx

    func_1 = lambda x: func(x) + zk.dot(A.dot(x) - B)
    grad_1 = lambda x: grad(x) + A.T.dot(zk)
    xk = x0
    gk = grad_1(xk)
    history = []
    while True:
        if np.sqrt((gk**2).sum()) < eps:
            break
        else:
            i += 1
            f = func_1(xk)
            print('[DU2] iter = {}   f = {:.6f}'.format(i, f))
            history.append(f)

        dk = -gk
        dx = backtrack_2(xk, dk, a, b)
        xk = xk + dx
        gk = grad_1(xk)

    return xk, history_dual, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n', type=int, default=500)
    parser.add_argument('-p', '--p', type=int, default=100)
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-b', '--beta', type=float, default=0.8)
    parser.add_argument('-eta', '--eta', type=float, default=1e-5)
    args = parser.parse_args()

    a, b, eta = args.alpha, args.beta, args.eta

    A = np.random.randn(args.p, args.n)
    B = np.random.randn(args.p)
    x0 = np.random.uniform(.5, 1, (args.n))
    B = A.dot(x0)

    fig = plt.figure()

    print()
    print('Projected Gradient')
    print('------------------')
    x, history = proj_grad(x0, A, B, func, grad, eta, a, b)
    ax = fig.add_subplot('221')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    print()
    print('Eliminate Equations')
    print('-------------------')
    x, history = elim_eq(x0, A, B, func, grad, eta, a, b)
    ax = fig.add_subplot('222')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    print()
    print('Dual Approach')
    print('-------------------')
    x, history_dual, history = dual_approach(x0, A, B, func, grad, eta, a, b)
    ax = fig.add_subplot('223')
    ax.plot(np.arange(len(history_dual)), history_dual, marker='D', color='#3972ad')
    ax = fig.add_subplot('224')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    fig.set_size_inches(9, 9)
    fig.savefig('eqcontraned.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
