"""
ResNet implementation in numpy

author:  fengyanlin@pku.edu.cn
"""

import numpy as np


n_layers = 3
n_epochs = 100
iw = 400
ih = 200
fsize = 3
n_labels = 10
lr = 0.000001


def resnet_step(X, y, W, U, b):
    # forward pass
    conv_in = []
    conv_out = []
    relu_out = []
    layer_out = []
    X_in = X
    for l in range(n_layers):
        conv_in.append(X_in)
        X_conv = np.zeros((iw, ih))
        for i in range(fsize):
            for j in range(fsize):
                X_conv[:-fsize + 1, :-fsize + 1] += W[l, i, j] * X_in[i:i + iw - fsize + 1, j:j + ih - fsize + 1]
        conv_out.append(X_conv)

        # ReLU activation
        X_relu = np.maximum(X_conv, 0)
        relu_out.append(X_relu)

        # residual connection
        X_out = X_relu + X_in
        layer_out.append(X_out)
        X_in = X_out

    # FC layer
    Xr = X_in.reshape(-1)
    Z = U.T.dot(Xr) + b
    Z = Z - Z.max()
    log_probs = Z - np.log(np.exp(Z).sum())

    # compute loss
    pred = log_probs.argmax()
    loss = -log_probs[y]

    # backward pass
    dZ = np.exp(log_probs) - np.eye(n_labels)[y]
    db = dZ
    dU = np.outer(Xr, dZ)

    dXr = U.dot(dZ)
    dX_out = dXr.reshape(iw, ih)
    dW = np.zeros((n_layers, fsize, fsize))
    for l in range(n_layers - 1, -1, -1):
        dX_relu = dX_out
        dX_conv = (conv_out[l] >= 0) * dX_relu
        dX_in = np.zeros((iw, ih))
        for i in range(fsize):
            for j in range(fsize):
                dW[l, i, j] = (dX_conv[:-fsize + 1, :-fsize + 1] * conv_in[l][i:i + iw - fsize + 1, j:j + ih - fsize + 1]).sum()
                dX_in[i:i + iw - fsize + 1, j:j + ih - fsize + 1] += W[l, i, j] * dX_conv[:-fsize + 1, :-fsize + 1]
        dX_out = dX_in = dX_out + dX_in

    return pred, loss, dW, dU, db


def main():
    # initialization
    W = np.random.randn(n_layers, fsize, fsize)  # filters
    U = np.random.randn(iw * ih, n_labels)  # FC layer - weight
    b = np.random.randn(n_labels)  # FC layer - bias
    X = np.random.randn(iw, ih)
    y = np.random.randint(0, n_labels)

    for epoch in range(n_epochs):
        pred, loss, dW, dU, db = resnet_step(X, y, W, U, b)

        # verify gradients
        eps = 1e0
        t = 1e-10
        v = np.random.randn(*W.shape)
        _, loss_t, _, _, _ = resnet_step(X, y, W + t * v, U, b)
        assert ((loss_t - loss) / t - (dW * v).sum()) < eps
        v = np.random.randn(*U.shape)
        _, loss_t, _, _, _ = resnet_step(X, y, W, U + t * v, b)
        assert ((loss_t - loss) / t - (dU * v).sum()) < eps
        v = np.random.randn(*b.shape)
        _, loss_t, _, _, _ = resnet_step(X, y, W, U, b + t * v)
        assert ((loss_t - loss) / t - (db * v).sum()) < eps

        # take a SGD step
        W -= lr * dW
        U -= lr * dU
        b -= lr * db

        print('epoch {:3}  loss {:10.4f}   pred {}   gold {}  [grad verified]'.format(epoch, loss, pred, y))


if __name__ == '__main__':
    main()
