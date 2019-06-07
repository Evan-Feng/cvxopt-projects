"""
the conjugate gradient algorithm

author:  fengyanlin@pku.edu.cn
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import argparse


def func(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def grad(x):
    return np.array([-400 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])


def backtrack(x, dx, a, b):
    f = func(x)
    df = grad(x)
    while func(x + dx) > f + a * dx.dot(df):
        dx = dx * b
    return dx


def update_beta(gk, gk_1, dk, formula='hs'):
    if formula == 'hs':
        return gk_1.dot(gk_1 - gk) / dk.dot(gk_1 - gk)
    elif formula == 'pr':
        return gk_1.dot(gk_1 - gk) / gk.dot(gk)
    elif formula == 'fr':
        return gk_1.dot(gk_1) / gk.dot(gk)


def conj_grad(x0, func, grad, formula, eps, a, b):
    xk = x0
    gk = grad(xk)
    dk = -gk
    history = []
    i = 0
    while True:
        if np.sqrt((gk**2).sum()) < eps:
            break
        else:
            i += 1
            f = func(xk)
            print('iter = {}   f = {:.6f}'.format(i, f))
            history.append(np.log(f))

        dx = backtrack(xk, dk, a, b)
        xk = xk + dx
        gk_1 = grad(xk)
        bk = update_beta(gk, gk_1, dk, formula)
        dk = -gk_1 + bk * dk
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

    for j, formula in enumerate(['hs', 'pr', 'fr']):
        x, history = conj_grad(np.array([-2., 2.]), func, grad, formula, eta, a, b)
        ax = fig.add_subplot('13{}'.format(j + 1))
        ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    fig.set_size_inches(14, 4)
    fig.savefig('conj_grad.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
