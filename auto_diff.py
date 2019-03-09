"""
a program for automatic differentiation

author: fengyanlin@pku.edu.cn
"""

import numpy as np


compute_graph = []


class Scalar(object):

    def __init__(self, value):
        self.grad = 0
        self.value = value
        self.graph_id = None

    def backward(self):
        global compute_graph
        compute_graph = compute_graph[:self.graph_id + 1]
        self.grad = 1
        while len(compute_graph) > 0:
            inputs, op, out = compute_graph.pop()
            grads = op.backward(out.grad)
            for x, dx in zip(inputs, grads):
                x.grad += dx

    def register_id(self, graph_id):
        self.graph_id = graph_id


class Function(object):
    """
    Function base class
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Add(Function):
    """
    ADD operator: a, b -> a + b
    """

    def forward(self, a, b):
        return Scalar(a.value + b.value)

    def backward(self, grad_out):
        return [grad_out, grad_out]


class Cos(Function):
    """
    COS operator: a -> cos(a)
    """

    def forward(self, a):
        self.forward_values = a.value
        return Scalar(np.cos(a.value))

    def backward(self, grad_out):
        a_val = self.forward_values
        return [grad_out * -np.sin(a_val)]


class Sin(Function):
    """
    SIN operator: a -> sin(a)
    """

    def forward(self, a):
        self.forward_values = a.value
        return Scalar(np.sin(a.value))

    def backward(self, grad_out):
        a_val = self.forward_values
        return [grad_out * np.cos(a_val)]


class Log(Function):
    """
    LOG operator: a -> log(a)
    """

    def forward(self, a):
        self.forward_values = a.value
        return Scalar(np.log(a.value))

    def backward(self, grad_out):
        a_val = self.forward_values
        return [grad_out / a_val]


class Tan(Function):
    """
    TAN operator: a -> tan(a)
    """

    def forward(self, a):
        self.forward_values = a.value
        return Scalar(np.tan(a.value))

    def backward(self, grad_out):
        a_val = self.forward_values
        return [grad_out / np.cos(a_val)**2]


class Exp(Function):
    """
    exp operator: a -> e^(a)
    """

    def forward(self, a):
        self.forward_values = a.value
        return Scalar(np.exp(a.value))

    def backward(self, grad_out):
        a_val = self.forward_values
        return [grad_out * np.exp(a_val)]


class Mul(Function):
    """
    MUL operator: a, b -> a * b
    """

    def forward(self, a, b):
        self.forward_values = [a.value, b.value]
        return Scalar(a.value * b.value)

    def backward(self, grad_out):
        a_val, b_val = self.forward_values
        return [b_val * grad_out, a_val * grad_out]


def register_op(inputs, op, output):
    graph_id = len(compute_graph)
    compute_graph.append([inputs, op, output])
    return graph_id


def add(a, b):
    op = Add()
    out = op(a, b)
    graph_id = register_op([a, b], op, out)
    out.register_id(graph_id)
    return out


def mul(a, b):
    op = Mul()
    out = op(a, b)
    graph_id = register_op([a, b], op, out)
    out.register_id(graph_id)
    return out


def cos(a):
    op = Cos()
    out = op(a)
    graph_id = register_op([a], op, out)
    out.register_id(graph_id)
    return out


def sin(a):
    op = Sin()
    out = op(a)
    graph_id = register_op([a], op, out)
    out.register_id(graph_id)
    return out


def tan(a):
    op = Tan()
    out = op(a)
    graph_id = register_op([a], op, out)
    out.register_id(graph_id)
    return out


def log(a):
    op = Log()
    out = op(a)
    graph_id = register_op([a], op, out)
    out.register_id(graph_id)
    return out


def exp(a):
    op = Exp()
    out = op(a)
    graph_id = register_op([a], op, out)
    out.register_id(graph_id)
    return out


def zero_grad(*args):
    for x in args:
        x.grad = 0


def main():
    dag = lambda x1, x2, x3: (np.sin(x1 + 1) + np.cos(2 * x2)) * np.tan(np.log(x3)) + \
        (np.sin(x2 + 1) + np.cos(2 * x1)) * np.exp(np.sin(x3) + 1)

    for i in range(100):
        values = np.random.rand(3) + 1e-1
        v1, v2, v3 = values
        x1, x2, x3 = Scalar(v1), Scalar(v2), Scalar(v3)
        one = Scalar(1)
        two = Scalar(2)
        f = add(mul(add(sin(add(x1, one)), cos(mul(two, x2))), tan(log(x3))),
                mul(add(sin(add(x2, one)), cos(mul(two, x1))), exp(add(one, sin(x3)))))
        f.backward()

        # verify gradients
        t = 1e-10
        eps = 1e-2
        t1, t2, t3 = np.random.rand(3)
        gold = (dag(v1 + t * t1, v2 + t * t2, v3 + t * t3) - dag(v1, v2, v3)) / t
        estim = x1.grad * t1 + x2.grad * t2 + x3.grad * t3
        assert np.abs(((gold - estim) / gold)) < eps
        print('test {} passed'.format(i))


if __name__ == '__main__':
    main()
