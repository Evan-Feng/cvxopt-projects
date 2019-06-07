"""
Minimizing Rosenbrock's function using steepest descent,
Gauss-Newton method and Damped Newton method


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


def hessian(x):
    return np.array([[1200 * x[0]**2 - 400 * x[1], -400 * x[0]],
                     [-400 * x[0], 200]])


def jacob(x):
    return np.array([[-2 * np.sqrt(100) * x[0], np.sqrt(100)],
                     [-1, 0]])


def backtrack(x, dx, a, b):
    f = func(x)
    df = grad(x)
    while func(x + dx) > f + a * dx.dot(df):
        dx = dx * b
    return dx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.1)
    parser.add_argument('-b', '--beta', type=float, default=0.8)
    parser.add_argument('-eta', '--eta', type=float, default=1e-5)
    args = parser.parse_args()

    a, b, eta = args.alpha, args.beta, args.eta

    fig = plt.figure()

    run_time = np.zeros(3)
    n_epoch = np.ones(3)

    # steepest descent
    print()
    t0 = time.time()
    x = np.array([-2, 2])
    history = []
    niter = 0
    while True:
        f = func(x)
        P = hessian(x)
        df = grad(x)
        print('iter = {}   f = {:.8f}'.format(niter, f))
        history.append(f)
        if np.sqrt(np.sum(df**2)) <= eta:
            break
        else:
            dx = -np.linalg.pinv(P).dot(df)
            dx = backtrack(x, dx, a, b)
            x = x + dx
            niter += 1
    # print('finished in {:.4f} seconds'.format(time.time() - t0))
    run_time[0] = time.time() - t0
    n_epoch[0] = len(history)
    ax = fig.add_subplot(131)
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')
    ax.set_title('steepest descent')

    # gauss-newton method
    print()
    t0 = time.time()
    x = np.array([-2, 2])
    history = []
    niter = 0
    while True:
        f, J, df = func(x), jacob(x), grad(x)
        r = np.array([np.sqrt(100) * (x[1] - x[0]**2), 1 - x[0]])
        print('iter = {}   f = {:.8f}'.format(niter, f))
        history.append(f)
        if np.sqrt(np.sum(df**2)) <= eta or niter > 10:
            break
        else:
            dx = -np.linalg.pinv(J.T.dot(J)).dot(J.T.dot(r))
            x = x + dx
            niter += 1
    # print('finished in {:.4f} seconds'.format(time.time() - t0))
    run_time[1] = time.time() - t0
    n_epoch[1] = len(history)
    ax = fig.add_subplot(132)
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')
    ax.set_title('gauss-newton method')

    # damped newton method
    print()
    t0 = time.time()
    x = np.array([-2, 2])
    history = []
    niter = 0
    while True:
        f, P, df = func(x), hessian(x), grad(x)
        print('iter = {}   f = {:.8f}'.format(niter, f))
        history.append(f)
        df_norm = np.sqrt(np.sum(df**2))
        if df_norm <= eta:
            break
        else:
            dx = -np.linalg.pinv(P).dot(df)
            if df_norm > 10 * eta:
                dx = backtrack(x, dx, a, b)
            x = x + dx
            niter += 1
    # print('finished in {:.4f} seconds'.format(time.time() - t0))
    run_time[2] = time.time() - t0
    n_epoch[2] = len(history)
    ax = fig.add_subplot(133)
    ax.plot(np.arange(len(history)), history, marker='D', color='#3972ad')
    ax.set_title('damped newton method')

    print('run time (s): {}'.format(run_time))
    print('run time per epoch (ms): {}'.format(run_time / n_epoch))

    fig.set_size_inches(14, 4)
    fig.savefig('hessian.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
