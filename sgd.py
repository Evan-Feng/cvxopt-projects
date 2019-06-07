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


def sgd(w0, X, y, a, b, eps):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    func = lambda w: np.log(1 + np.exp(-y * X.dot(w))).mean()
    grad = lambda w: ((sigmoid(-y * X.dot(w)) * (-y))[:, np.newaxis] * X).mean(0)
    w = w0
    step = 0
    history = []
    while True:
        g = grad(w)
        if np.linalg.norm(g) < eps:
            break
        print('step = {}   f = {:.6f}   g = {:.6f}'.format(step, func(w), np.sqrt((g**2).sum())))
        history.append(func(w))
        dw = backtrack(w, -g, func, grad, a, b)
        w = w + dw
        step += 1

    return w, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-b', '--beta', type=float, default=0.8)
    parser.add_argument('-eta', '--eta', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    a, b, eta = args.alpha, args.beta, args.eta

    np.random.seed(args.seed)

    X = np.random.randn(100, 20)
    y = np.sign(np.random.randn(100))
    w0 = np.random.randn(20)

    fig = plt.figure()
    w, history = sgd(w0, X, y, a, b, eta)
    ax = fig.add_subplot('111')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')
    # ax = fig.add_subplot('122')
    # ax.plot(np.arange(len(err)), err, marker='D', color='#3972ad')

    fig.set_size_inches(4, 4)
    fig.savefig('sgd.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
