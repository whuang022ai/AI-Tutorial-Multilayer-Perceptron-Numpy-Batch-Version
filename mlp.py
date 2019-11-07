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
        #self.WIH = np.random.rand(self.input_size+1, self.hidden_size)
        #self.WHO = np.random.rand(self.hidden_size+1, self.output_size)
        # init by Xavier init x sigmoid
        a = pow(6/(self.input_size+1+self.hidden_size), 1/4)
        self.WIH = np.random.uniform(-1*a, a,
                                     (self.input_size+1, self.hidden_size))
        a = pow(6/(self.hidden_size+1+self.output_size), 1/4)
        self.WHO = np.random.uniform(-1*a, a,
                                     (self.hidden_size+1, self.output_size))

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
            mse += ((D[i]-self.O[i])).mean()**2
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
            activate_function = np.vectorize(act.f)
            self.H = activate_function(self.sumIH)  # H = sigmoid(sum)
            self.H = np.append(self.H, [[1.0]])
            self.sumHO = np.dot(self.H, self.WHO)  # sum = HW
            self.O = activate_function(self.sumHO)  # O = sigmoid(sum)
            print(self.O)

    def fit(self, X, D, epochs, learing_rate, draw_mse=False, early_stopping=False):
        if(draw_mse):
            plt.figure('Neural Network MSE Error Monitor')
            plt.axis([0, epochs, 0, 0.0001])
            plt.draw()
            plt.ion()
            plt.autoscale(enable=True, axis='both')
        for i in range(epochs):
            self.forward(X)
            mse = self.meansure_error_mse(D)
            self.backward(D)
            self.update_value_calculation()
            self.update_fullbatch(learing_rate)
            if(i % 100 == 0):
                print(mse)
            if(i > epochs*0.1 and i % 10 == 0 and draw_mse):
                plt.plot(i, mse, 'b*-', label="MSE")
                plt.pause(0.01)
            if(i > epochs*0.01 and mse < 0.0001 and early_stopping):
                break
        if(draw_mse):
            plt.ioff()
            plt.show(block=True)

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

    # leaky-relu recommand setting for xor : epochs=200 , lr =0.07 , act.alpha=0.2
    
    # xor problem setting
    input_size = 2
    sample_size = 4
    output_size = 1
    hidden_size = 5
    epochs = 6000
    learing_rate = 0.8
    draw_mse=True
    early_stopping=True

    # iris problem setting
    # input_size = 4
    # sample_size = 4
    # output_size = 1
    # hidden_size = 5
    # epochs = 5000
    # learing_rate = 0.08

    # iris 2 class one hot problem setting
    # input_size = 4
    # sample_size = 4
    # output_size = 2
    # hidden_size = 5
    # epochs = 500
    # learing_rate = 0.08
    # draw_mse=True
    # early_stopping=False

    mlp = MLP(input_size, sample_size, hidden_size, output_size)
    F = np.genfromtxt('xor_dataset.csv', delimiter=',')
    spilt_colindex = input_size
    X = F[:, :spilt_colindex]
    D = F[:, spilt_colindex:]
    mlp.fit(X, D, epochs, learing_rate,  draw_mse, early_stopping)
    mlp.save_model('xor')
    mlp.test_forward()
