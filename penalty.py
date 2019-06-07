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


def solve_penalty(A, B, eps):
    i = 0
    history = []
    gamma = 1
    I = np.identity(A.shape[1])
    while True:
        i += 1
        x = np.linalg.pinv(A.T.dot(A) + 1 / (2 * gamma) * I).dot(A.T.dot(B))
        gamma *= 2
        f = (x**2).sum() / 2
        print('iter = {}   f = {:.6f}'.format(i, f))
        history.append(f)
        if np.sqrt(((A.dot(x) - B)**2).sum()) < eps:
            break
    return x, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-b', '--beta', type=float, default=0.8)
    parser.add_argument('-eta', '--eta', type=float, default=1e-5)
    args = parser.parse_args()

    a, b, eta = args.alpha, args.beta, args.eta

    A = np.random.uniform(-1, 1, (200, 300))
    B = np.random.uniform(-1, 1, (200,))

    fig = plt.figure()
    x, history = solve_penalty(A, B, eta)
    ax = fig.add_subplot('111')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    fig.set_size_inches(4, 4)
    fig.savefig('penalty.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
