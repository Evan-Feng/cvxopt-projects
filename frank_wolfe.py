"""
the penalty method

author:  fengyanlin@pku.edu.cn
"""

import numpy as np
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse


def backtrack(x, dx, func, grad, a, b):
    f = func(x)
    df = grad(x)
    while func(x + dx) > f + a * dx.dot(df):
        dx = dx * b
    return dx


def proj_grad(x0, func, grad, proj, eps, a, b):
    """
    Projected gradient method.
    """
    i = 0
    history = []
    xk = x0
    gk = grad(xk)
    last_updated = 0
    while True:
        if i - last_updated > 50:
            break
        else:
            if i > 0 and func(xk) < f - eps:
                last_updated = i
            f = func(xk)
            print('[Proj_Grad] iter = {}   f = {:.6f}'.format(i, f))
            i += 1
            history.append(f)
        dx = backtrack(xk, -gk, func, grad, a, b)
        xk = proj(xk + dx)
        gk = grad(xk)
    return xk, history


def frank_wolfe(x0, func, grad, linearize_solver, eps):
    """
    Projected gradient method.
    """
    i = 0
    history = []
    xk = x0
    gk = grad(xk)
    last_updated = 0
    while True:
        if i - last_updated > 50:
            break
        else:
            if i > 0 and func(xk) < f - eps:
                last_updated = i
            f = func(xk)
            print('[Frank_Wolfe] iter = {}   f = {:.6f}'.format(i, f))
            i += 1
            history.append(f)
        g = linearize_solver(gk)
        gamma = 2 / (i + 2)
        xk = (1 - gamma) * xk + gamma * g
        gk = grad(xk)
    return xk, history


def proj_l1_ball(x, a=1):
    if np.abs(x).sum() < a:
        return x
    lambd = 0
    while True:
        m = np.abs(x[np.abs(x) > lambd]).min()
        if np.maximum(0, np.abs(x) - m).sum() <= a:
            lambd = lambd + (np.maximum(0, np.abs(x) - lambd).sum() - a) / (np.abs(x) > lambd).sum()
            break
        else:
            lambd = m
    return np.maximum(0, np.abs(x) - lambd) * np.sign(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-b', '--beta', type=float, default=0.8)
    parser.add_argument('-eta', '--eta', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    a, b, eta = args.alpha, args.beta, args.eta
    np.random.seed(args.seed)

    A = np.random.uniform(-1, 1, (200, 300))
    B = np.random.uniform(-1, 1, (200,))
    x0 = np.random.uniform(-1, 1, (300,))

    func = lambda x: ((A.dot(x) - B)**2).sum()
    grad = lambda x: A.T.dot(A.dot(x) - B)

    fig = plt.figure()

    ###################################################
    # l-1 norm                                        #
    ###################################################

    proj = proj_l1_ball
    x, history = proj_grad(x0, func, grad, proj, eta, a, b)
    ax = fig.add_subplot('221')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    solver = lambda g: (np.abs(g) == np.abs(g).max()).astype(float) * (g < 0).astype(float)
    x, history = frank_wolfe(x0, func, grad, solver, eta)
    ax = fig.add_subplot('222')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    ###################################################
    # l-inf norm                                      #
    ###################################################

    proj = lambda x: np.minimum(1, np.maximum(-1, x))
    x, history = proj_grad(x0, func, grad, proj, eta, a, b)
    ax = fig.add_subplot('223')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    solver = lambda g: (g < 0).astype(float)
    x, history = frank_wolfe(x0, func, grad, solver, eta)
    ax = fig.add_subplot('224')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    fig.set_size_inches(9, 9)
    fig.savefig('frank_wolfe.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
