"""
the Majorant Minimization algorithm

author:  fengyanlin@pku.edu.cn
"""

import numpy as np
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse


def backtrack(x, dx, a, b):
    f = func(x)
    df = grad(x)
    while func(x + dx) > f + a * dx.dot(df):
        dx = dx * b
    return dx


def bfgs(x0, H0, func, grad, eps, a, b, max_iter=float('inf')):
    i = 0
    history = []
    xk = x0
    gk = grad(xk)
    Hk = H0
    while True:
        if np.sqrt((gk**2).sum()) < eps or i > max_iter:
            break
        else:
            i += 1
            f = func(xk)
            history.append(f)

        dk = -Hk.dot(gk)
        # dx = backtrack(xk, dk, a, b)
        dx = dk
        xk = xk + dx
        gk_1 = grad(xk)
        dg = gk_1 - gk
        Hk = Hk + (1 + dg.dot(Hk.dot(dg)) / dg.dot(dx)) * (np.outer(dx, dx)) / dx.dot(dg) - \
            (np.outer(Hk.dot(dg), dx) + np.outer(dx, Hk.dot(dg))) / dg.dot(dx)
        gk = gk_1
    return xk, history


def mm_mse_grdmj(x0, A, B, lambd, eps, a, b):
    """
    Solve MSE loss using lipschitz gradient method.
    """
    func = lambda x: 0.5 * ((A.dot(x) - B) ** 2).sum()
    grad = lambda x: A.T.dot(A.dot(x) - B)
    l1norm = lambda x: lambd * np.abs(x).sum()
    _, s, _ = np.linalg.svd(A)
    L = s[0] ** 2

    i = 0
    xk = x0
    history = []
    while True:
        fk = func(xk)
        gk = grad(xk)
        gk_1 = gk + lambd * ((xk > 0) * 2 - 1)
        if np.sqrt((gk_1**2).sum()) < eps:
            break
        else:
            i += 1
            print('iter = {}   f = {}'.format(i, fk + l1norm(xk)))
            history.append(fk + l1norm(xk))
        g_func = lambda x: fk + gk.dot(x - xk) + (L / 2) * ((x - xk)**2).sum() + lambd * np.abs(x).sum()
        g_grad = lambda x: gk + L * (x - xk) + lambd * ((x > 0) * 2 - 1)
        xk, _ = bfgs(xk, np.identity(xk.shape[0]), g_func, g_grad, eps, a, b)

    return xk, history


def mm_mse_vrmj(x0, A, B, lambd, eps, a, b):
    """
    Solve MSE loss using variational majorant method.
    """
    func = lambda x: 0.5 * ((A.dot(x) - B) ** 2).sum()
    grad = lambda x: A.T.dot(A.dot(x) - B)
    l1norm = lambda x: lambd * np.abs(x).sum()

    i = 0
    xk = x0
    dk = np.random.randn(x0.shape[0])
    I = np.identity(x0.shape[0])
    one = np.ones(x0.shape[0])
    history = []
    while True:
        fk = func(xk)
        gk = grad(xk)
        gk_1 = gk + lambd * ((xk > 0) * 2 - 1)
        if np.sqrt((gk_1**2).sum()) < eps or i > 100:
            break
        else:
            i += 1
            print('iter = {}   f = {}'.format(i, fk + l1norm(xk)))
            history.append(fk + l1norm(xk))

        h_func = lambda d: lambd * (0.5 * (xk.dot(d * xk) + (1 / d).sum())) + 0.5 * ((A.dot(xk) - B) ** 2).sum()
        h_grad = lambda d: lambd * (xk**2 - 2 / d**2)
        dk, _ = bfgs(dk, I, h_func, h_grad, eps, a, b, 2)
        g_func = lambda x: lambd * (0.5 * (x.dot(dk * x) + (1 / dk).sum())) + 0.5 * ((A.dot(x) - B) ** 2).sum()
        g_grad = lambda x: lambd * (dk * x) + A.T.dot(A.dot(x) - B)
        xk, _ = bfgs(xk, I, g_func, g_grad, eps, a, b)

    return xk, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-b', '--beta', type=float, default=0.8)
    parser.add_argument('-lambd', '--lambd', type=float, default=1e-5)
    parser.add_argument('-eta', '--eta', type=float, default=1e-5)
    args = parser.parse_args()

    a, b, eta = args.alpha, args.beta, args.eta

    fig = plt.figure()

    x0 = np.random.randn(10)
    A = np.random.randn(10, 10) / 10
    B = np.random.randn(10) / 10

    x, history = mm_mse_grdmj(x0, A, B, args.lambd, eta, a, b)
    ax = fig.add_subplot(f'121')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    x, history = mm_mse_vrmj(x0, A, B, args.lambd, eta, a, b)
    ax = fig.add_subplot('122')
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')

    fig.set_size_inches(9, 4)
    fig.savefig('mm.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
