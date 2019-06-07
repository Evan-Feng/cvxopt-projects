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


def ladmpsap(Z0, E0, L0, D, lambd=.1, bmax=1e2, rho=1, eps=1e-5):
    def prox_nuc(X, c):
        u, s, vt = np.linalg.svd(X)
        s = np.maximum(s - c, 0)
        return u.dot(np.diag(s)).dot(vt)

    def prox_21(X, c):
        norms = np.sqrt((X ** 2).sum(0))
        return X * np.maximum((norms - c) / norms, 0)

    Z, E, L = Z0, E0, L0
    I = np.identity(D.shape[0])
    ones = np.ones((1, D.shape[1]))
    zeros = np.zeros((1, D.shape[0]))
    A = np.concatenate([D, ones], 0)
    B = np.concatenate([I, zeros], 0)
    C = np.concatenate([D, ones], 0)
    A, B, C = D, I, D
    norm_nuc = lambda X: np.linalg.norm(X, 'nuc')
    norm_21 = lambda X: np.sqrt((X**2).sum(0)).sum()
    func = lambda Z, E: norm_nuc(Z) + lambd * norm_21(E)
    step = 0
    b = 1
    ea = np.sqrt((A**2).sum()) * 1.02
    eb = np.sqrt((B**2).sum()) * 1.02
    history, err = [], []
    while True:
        if np.linalg.norm(A.dot(Z) + B.dot(E) - C) / np.linalg.norm(C) < eps:
            break
        print('step = {}   f = {:.6f}   err = {:.6f}'.format(step, func(Z, E), np.sqrt(((A.dot(Z) + B.dot(E) - C)**2).sum())))
        history.append(func(Z, E))
        err.append(np.sqrt(((A.dot(Z) + B.dot(E) - C)**2).sum()))
        Zx = Z - A.T.dot(L + b * (A.dot(Z) + B.dot(E) - C)) / (b * ea)
        Z = prox_nuc(Zx, 1 / (b * ea))
        Ex = E - B.T.dot(L + b * (A.dot(Z) + B.dot(E) - C)) / (b * eb)
        E = prox_21(Ex, lambd / (b * ea))
        L = L + b * (A.dot(Z) + B.dot(E) - C)
        b = min(rho * b, bmax)
        step += 1
    return (Z, E), history, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-b', '--beta', type=float, default=0.8)
    parser.add_argument('-eta', '--eta', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    a, b, eta = args.alpha, args.beta, args.eta

    np.random.seed(args.seed)

    D = np.random.uniform(-0.1, 0.1, (200, 300))
    Z0 = np.random.uniform(-0.1, 0.1, (300, 300))
    E0 = np.random.uniform(-0.1, 0.1, (200, 300))
    L0 = np.random.uniform(-0.1, 0.1, (200, 300))

    fig = plt.figure()
    (Z, E), history, err = ladmpsap(Z0, E0, L0, D)
    ax = fig.add_subplot('121')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')
    ax = fig.add_subplot('122')
    ax.plot(np.arange(len(err)), err, marker='D', color='#3972ad')

    fig.set_size_inches(9, 4)
    fig.savefig('ladmpsap.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
