# -*- coding: utf-8 -*-
#
#  @about A 3-layer-sigmoid-MLP Neural Network using Numpy .
#  This is a version that processes the data by full batch which is more easier than the older version
#  https://github.com/whuang022ai/AI-Tutorial-Multilayer-Perceptron-Numpy-Version-
#  @auth whuang022ai
#

import numpy as np
import matplotlib.pyplot as plt
from actf import Activate_Function_Generator
from actf import Activate_Function_Type


class MLP():

    def __init__(self, input_size, batch_size, hidden_size, output_size):

        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.WIH = np.random.rand(self.input_size+1, self.hidden_size)
        self.WHO = np.random.rand(self.hidden_size+1, self.output_size)

    def colum_add_ones(self, x):
        return np.hstack((x, np.ones((x.shape[0], 1), dtype=x.dtype)))

    def forward(self, X):
        self.X = self.colum_add_ones(X)
        self.sumIH = np.dot(self.X, self.WIH)  # sum = XW
        act = Activate_Function_Generator(Activate_Function_Type.sigmoid).get
        activate_function = np.vectorize(act.f)
        self.H = activate_function(self.sumIH)  # H = sigmoid(sum)
        self.H = self.colum_add_ones(self.H)
        self.sumHO = np.dot(self.H, self.WHO)  # sum = HW
        self.O = activate_function(self.sumHO)  # O = sigmoid(sum)
        return self.O

    def meansure_error_mse(self, D):
        mse = 0.0
        for i in range(len(D)):
            mse += (D[i]-self.O[i])**2
        mse /= len(D)
        return mse

    def backward(self, D):
        act = Activate_Function_Generator(Activate_Function_Type.sigmoid).get
        activate_function_d = np.vectorize(act.df)
        deltaO = -1*(D-self.O)
        self.dO = (deltaO)*activate_function_d(self.O)
        dtmpH = np.dot(self.dO, np.transpose(self.WHO))
        dtmpH = dtmpH[:, 1:]
        H = self.H[:, 1:]
        self.dH = dtmpH*activate_function_d(H)
        return deltaO

    def update_value_calculation(self):
        self.dWHO = np.dot(np.transpose(self.H), self.dO)
        self.dWIH = np.dot(np.transpose(self.X), self.dH)
        return

    def update_fullbatch(self, lr):
        self.WHO = self.WHO-lr*self.dWHO
        self.WIH = self.WIH-lr*self.dWIH
        return

    def test_forward(self):
        while (True):
            # get input
            input_data = np.arange(self.input_size+1)
            for x in range(self.input_size):
                input_data[x] = float(input('Enter the feature: '))
            input_data[self.input_size] = 1
            # same process as forward
            self.sumIH = np.dot(input_data, self.WIH)  # sum = XW
            act = Activate_Function_Generator(
                Activate_Function_Type.sigmoid).get
            # act.show_plot()
            activate_function = np.vectorize(act.f)
            self.H = activate_function(self.sumIH)  # H = sigmoid(sum)
            self.H = np.append(self.H, [[1.0]])
            self.sumHO = np.dot(self.H, self.WHO)  # sum = HW
            self.O = activate_function(self.sumHO)  # O = sigmoid(sum)
            print(self.O)

    def save_model(self, file_name):
        network_setting = np.zeros(4)
        network_setting[0] = self.input_size
        network_setting[1] = self.batch_size
        network_setting[2] = self.hidden_size
        network_setting[3] = self.output_size
        np.savetxt(file_name+'_config.txt',  network_setting)
        np.savetxt(file_name+'_WIH.txt', self.WIH)
        np.savetxt(file_name+'_WHO.txt', self.WHO)
        return

    def load_model(self, file_name):
        network_setting = np.loadtxt(file_name+'_config.txt')
        self.input_size = int(network_setting[0])
        self.batch_size = int(network_setting[1])
        self.hidden_size = int(network_setting[2])
        self.output_size = int(network_setting[3])
        self.WIH = np.loadtxt(file_name+'_WIH.txt')
        self.WHO = np.loadtxt(file_name+'_WHO.txt')
        return


if __name__ == "__main__":

    # leaky-relu recommand setting for xor : epoh=200 , lr =0.07 , act.alpha=0.2
    draw_mse = True  # display mse realtime
    mlp = MLP(2, 4, 5, 1)
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]
    )

    D = np.array(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0]
        ]
    )
    epoh = 6000
    if(draw_mse):
        plt.figure('Neural Network MSE Error Monitor')
        plt.axis([0, epoh, 0, 0.0001])
        plt.draw()
        plt.ion()
        plt.autoscale(enable=True, axis='both')
    for i in range(epoh):
        mlp.forward(X)
        mse = mlp.meansure_error_mse(D)
        mlp.backward(D)
        mlp.update_value_calculation()
        mlp.update_fullbatch(0.5)
        if(i % 100 == 0):
            print(mse)
        if(i > epoh*0.1 and i % 10 == 0 and draw_mse):
            plt.plot(i, mse, 'b*-', label="MSE")
            plt.pause(0.01)
        if(i > epoh*0.01 and mse < 0.01):
            break
    if(draw_mse):
        plt.ioff()
        plt.show(block=True)
    mlp.save_model('xor')
    mlp.test_forward()
