"""
the L-BFGS algorithm

author:  fengyanlin@pku.edu.cn
"""

import numpy as np
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse


A = 10


def func(x):
    return A * ((x[1::2] - x[::2]**2)**2).sum() + ((1 - x[::2])**2).sum()


def grad(x):
    g1 = -4 * A * (x[1::2] - x[::2]**2) * x[::2] - 2 * (1 - x[::2])
    g2 = 2 * A * (x[1::2] - x[::2]**2)
    return np.stack([g1, g2], 1).reshape(-1)


def backtrack(x, dx, a, b):
    f = func(x)
    df = grad(x)
    while func(x + dx) > f + a * dx.dot(df):
        dx = dx * b
    return dx


def get_lbfgs_dir(sk, yk, H0, q0):
    q = q0
    a_set = []
    for s, y in zip(sk[::-1], yk[::-1]):
        p = 1 / (y.dot(s))
        a = p * s.dot(q)
        q = q - a * y
        a_set = [a] + a_set
    p = H0.dot(q)
    for s, y, a in zip(sk, yk, a_set):
        pk = 1 / (y.dot(s))
        b = pk * y.dot(p)
        p = p + (a - b) * s
    return p


def lbfgs(x0, m, func, grad, eps, a, b):
    i = 0
    history = []
    xk = x0
    gk = grad(xk)
    sk = []
    yk = []
    while True:
        if np.sqrt((gk**2).sum()) < eps:
            break
        else:
            i += 1
            f = func(xk)
            print('iter = {}   f = {:.6f}'.format(i, f))
            history.append(f)

        H0 = ((sk[-1].dot(yk[-1]) / yk[-1].dot(yk[-1])) if i > 1 else 1) * np.identity(x0.shape[0])
        dk = -get_lbfgs_dir(sk, yk, H0, grad(xk))
        dx = backtrack(xk, dk, a, b)
        xk = xk + dx
        gk_1 = grad(xk)
        sk.append(dx)
        yk.append(gk_1 - gk)
        if len(sk) > m:
            sk.pop(0)
            yk.pop(0)
        gk = gk_1

    return xk, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--m', type=int, default=10)
    parser.add_argument('-n', '--n', type=int, default=10)
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-b', '--beta', type=float, default=0.8)
    parser.add_argument('-eta', '--eta', type=float, default=1e-5)
    args = parser.parse_args()

    a, b, eta = args.alpha, args.beta, args.eta

    fig = plt.figure()

    x0 = np.array([-1] * args.n)

    for i, m in enumerate([1, 2, 4, 8, 16]):
        x, history = lbfgs(x0, m, func, grad, eta, a, b)
        ax = fig.add_subplot(f'15{i+1}')
        ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    fig.set_size_inches(22, 4)
    fig.savefig('lbfgs.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
