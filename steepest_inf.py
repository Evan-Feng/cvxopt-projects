"""
Computing the analytic center of a set of linear 
inequalitied using steepest descent on l-oo norm


author:  fengyanlin@pku.edu.cn
"""

import numpy as np
import matplotlib.pyplot as plt

import argparse


def func(x, A):
    return -np.log(1 - A.dot(x)).sum() - np.log(1 - x**2).sum()


def backtrack(x, A, a, b):
    f = func(x, A)
    df = A.T.dot(1 / (1 - A.dot(x))) + 2 * (x / (1 - x**2))
    dx = -np.abs(df).sum() * np.sign(df)
    while (1 - A.dot(x + dx) <= 0).any() or ((x + dx)**2 >= 1).any():
        dx = dx * b
    while func(x + dx, A) > f + a * dx.dot(df):
        dx = dx * b
    return f, df, dx


def run_backtrack(x, A, a, b, eta):
    print()
    print('running backtrack with:  a = {:.4f}  b = {:.4f}'.format(a, b))
    history = []
    niter = 0
    while True:
        f, df, dx = backtrack(x, A, a, b)
        history.append(f)
        print('iter = {}   f = {:.4f}    |df| = {:.4f}'.format(niter, f, np.sqrt(np.sum(df**2))))

        if np.sum(df**2) <= eta**2:
            break
        else:
            x = x + dx
            niter += 1
    return history, f, x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--m', type=int, default=50)
    parser.add_argument('-n', '--n', type=int, default=100)
    parser.add_argument('-a', '--alpha', type=float, nargs='+', default=[0.01, 0.03, 0.1, 0.3])
    parser.add_argument('-b', '--beta', type=float, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument('-eta', '--eta', type=float, default=1e-1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    m, n, eta = args.m, args.n, args.eta

    np.random.seed(args.seed)

    A = np.random.randn(m, n)
    x = np.zeros(n)

    fig = plt.figure()
    nrow = len(args.alpha)
    ncol = len(args.beta)
    for i, a in enumerate(args.alpha):
        for j, b in enumerate(args.beta):
            history, f, _ = run_backtrack(x, A, a, b, eta)
            ax = fig.add_subplot(nrow, ncol, i * nrow + j + 1)
            ax.plot(np.arange(len(history)), history - f, marker='D', color='#3972ad')
            ax.set_title('a = {:.4f}  b = {:.4f}'.format(a, b))

    fig.set_size_inches(4 * ncol, 4 * nrow)
    fig.savefig('steepest_inf.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
