# -*- coding: utf-8 -*-
#
#  Activate Function Model
#
#  @auth whuang022ai
#

import math
import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from enum import Enum


class Activate_Function_Type(Enum):

    sigmoid = 'sigmoid'
    tanh = 'tanh'
    relu = 'relu'
    leaky_relu = 'leaky-relu'


class Activate_Function(metaclass=ABCMeta):

    @abstractmethod
    def f(self, x):
        pass

    @abstractmethod
    def df(self, y):
        pass

    def plot_fx_dfx(self, min, max, inter, title):

        x = np.arange(min, max, inter)
        f_vect = np.vectorize(self.f)
        df_vect = np.vectorize(self.df)
        plt.figure(title)
        plt.subplot(211)
        plt.ylabel('f(x)')
        plt.plot(x, f_vect(x))
        plt.grid(True)
        plt.subplot(212)
        plt.ylabel('d f(x)')
        plt.plot(x, df_vect(f_vect(x)), color='red',
                 linewidth=1.0, linestyle='--')
        plt.grid(True)
        plt.show()

    @abstractmethod
    def show_plot(self):
        pass


class Activate_Function_Sigmoid(Activate_Function):

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def df(self, y):
        return y*(1-y)

    def show_plot(self):
        self.plot_fx_dfx(-5.0, 5.0, 0.1, 'The Sigmoid Function')


class Activate_Function_Tanh(Activate_Function):

    def f(self, x):
        return np.tanh(x)

    def df(self, y):
        return 1-y**2

    def show_plot(self):
        self.plot_fx_dfx(-5.0, 5.0, 0.1, 'The Tanh Function')


class Activate_Function_Relu(Activate_Function):

    def __init__(self):
        self.lrelu = Activate_Function_LeakyRelu()
        self.lrelu.alpha = 0.0

    def f(self, x):
        return self.lrelu.f(x)

    def df(self, y):
        return self.lrelu.df(y)

    def show_plot(self):
        self.plot_fx_dfx(-5.0, 5.0, 0.01, 'The ReLU Function')


class Activate_Function_LeakyRelu(Activate_Function):

    def __init__(self):
        self.alpha = 0.1

    def f(self, x):
        if x > 0:
            return x
        else:
            return (self.alpha*x)

    def df(self, y):
        if y > 0:
            return 1.0
        else:
            return self.alpha

    def show_plot(self):
        self.plot_fx_dfx(-5.0, 5.0, 0.01, 'The Leaky ReLU Function')


class Activate_Function_Generator():

    def __init__(self, name):

        if isinstance(name, Activate_Function_Type):
            name = str(name.value)
            self.__genfromname(name)
        else:
            self.__genfromname(name)

    def __genfromname(self, name):

        if name == 'sigmoid':
            self.get = Activate_Function_Sigmoid()
        elif name == 'tanh':
            self.get = Activate_Function_Tanh()
        elif name == 'relu':
            self.get = Activate_Function_Relu()
        elif name == 'leaky-relu':
            self.get = Activate_Function_LeakyRelu()
