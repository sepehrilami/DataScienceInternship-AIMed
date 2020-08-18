import numpy as np
import math


# 0-1 loss function
def zero_one_loss_function(true_value, estimate):
    return true_value != estimate


# absolute-loss
def absolute_loss(true_value, estimate):
    return abs(true_value - estimate)


def linear_loss_function(true_value, estimate):
    # we set 'a' as a number bigger than zero
    a = 0.7
    return a * abs(true_value - estimate)


def linear_loss_function_2(true_value, estimate):
    # we set 'a' and 'b' as numbers bigger than zero
    a = 0.7
    b = 0.6
    if true_value >= estimate:
        return a * abs(true_value - estimate)
    return b * abs(true_value - estimate)


# squared-error loss
def squared_error_loss(true_value, estimate):
    return abs(true_value - estimate) ** 2


# asymmetric squared-error loss function
def asymmetric_squared_error_loss_function(true_value, estimate):
    if estimate < true_value:
        return abs(true_value - estimate) ** 2
    # we set 'c' as a number between 0 and 1. For example:
    c = 0.5
    return c * abs(true_value - estimate) ** 2


# non-linear loss function
def non_linear_loss(true_value, estimate):
    if estimate * true_value > 0:
        return abs(estimate - true_value)
    else:
        return abs(estimate) * (estimate - true_value) ** 2


# log loss
def log_loss(true_label, estimate, eps=1e-15):
    p = np.clip(estimate, eps, 1 - eps)
    if true_label == 1:
        return -math.log(p)
    else:
        return -math.log(1 - p)


# similar with:
def cross_entropy(yHat, y):
    if y == 1:
        return -math.log(yHat)
    else:
        return -math.log(1 - yHat)


# another implementation for above
def log_loss_2(true_value, estimate):
    first_part = - estimate * math.log(true_value)
    second_part = (1 - estimate) * math.log(1 - true_value)
    return first_part - second_part


def loss(true_table, estimate):
    return abs(true_table - estimate) / (true_table * (1 - estimate))


# exp loss
def exp_loss(true_table, estimate):
    power = -((true_table - estimate) ** 2)
    return 1 - math.exp(power)


def Huber(estimate, y, delta=1.):
    return np.where(np.abs(y - estimate) < delta, .5 * (y - estimate) ** 2,
                    delta * (np.abs(y - estimate) - 0.5 * delta))


# Kullback-leibler divergence
def KLDivergence(yHat, y):
    # param yHat:
    # param y:
    # return: KLDiv(yHat or y)
    return np.sum(yHat * np.log((yHat / y)))


def L1(yHat, y):
    return np.sum(np.absolute(yHat - y))


def MSE(yHat, y):
    return np.sum((yHat - y) ** 2) / y.size


# showcase loss
def showcase_loss(guess, true_price, risk=50000):
    # risk is a parameter that defines how bad it is if your guess is over the true price.
    # I’ve arbitrarily picked 10,000.
    # A lower risk means that you are more comfortable with the idea of going over.
    # we set 'dif' as the acceptable difference between 'true_price' and 'estimate' .
    dif = 250
    if true_price < guess:
        return risk
    elif abs(true_price - guess) <= dif:
        return -2 * np.abs(true_price)
    else:
        return np.abs(true_price - guess - dif)

