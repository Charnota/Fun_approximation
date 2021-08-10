import numpy as np
import math #mathematics
import matplotlib.pyplot as plt #GRAPHIKI

#
class Layer:
    def __init__(self, input_num, neuron_num, o_fun, o_fun_point):
        self.input_num = input_num
        self.neuron_num = neuron_num
        self.weights = np.random.rand(input_num, neuron_num)
        self.s = np.array(np.zeros([neuron_num]), ndmin=2)
        self.o = np.array(np.zeros([neuron_num]), ndmin=2)
        self.bra_delta = np.array(np.zeros([neuron_num]),ndmin=2)
        self.activation_fun = o_fun
        self.activation_fun_point = o_fun_point #proizvodnaya ot f activazii
        pass
    def weighted_sum(self, in_x):
        self.s = in_x.dot(self.weights)
        pass
    def activate_s(self):
        self.o = self.activation_fun(self.s)
        pass
    def activate_s_point(self):
        return self.activation_fun_point(self.o)
    
    def calculate_out(self, in_x):
        self.weighted_sum(in_x)
        self.activate_s()
        pass
############

#
class Input_layer(Layer):
    def __init__(self, input_num):
        #
        def inp_layer_f(x):
            return x
        #
        def inp_layer_f_point(x):
            return 0
        #
        super().__init__(input_num, input_num, inp_layer_f, inp_layer_f_point)
        self.weights = np.eye(input_num)        
#
class NN:
    #init takes array of neuron nums d[0] = num_of_in_x, d[1] = neuron num of first hidden Layer...!!!
    #razmernost f menee d na 1 i soderzhit f[0] = activ fun nulevogo slouy
    def __init__(self, d, f, f_p):
        self.d = d
        self.layer = [Input_layer(d[0])]
        #making of input layer
        for i in range(len(d)-1):
            self.layer.append(Layer(d[i],d[i+1],f[i], f_p[i]))
            
        pass
    def get_y(self):
        return self.layer[len(self.d)-1].o
        pass
    def get_w(self):
        n = 0
        for i in self.layer:
            print('w[' , n, ']', ' = ', i.weights)
            n+=1
        pass
    def make_calculation(self, bra_x):
        in_x = np.array(bra_x, ndmin = 2)
        for i in self.layer:
            i.calculate_out(in_x)
            in_x = i.o
        return in_x
        pass
    def learn(self, lr, epochs, x_arr, target_bras_arr):
        for e in range(epochs):
            for i in range(x_arr.shape[0]):
                bra_x = np.array(x_arr[i], ndmin = 2)
                E = target_bras_arr[i,:] - self.make_calculation(bra_x)
                m = len(self.layer)
                self.layer[m-1].bra_delta = np.array(E*self.layer[m-1].activate_s_point(),ndmin =2)
                self.layer[m-1].weights += lr*self.layer[m-2].o.transpose().dot(self.layer[m-1].bra_delta)
                for j in range(m-2):
                    ind = m-j-2
                    self.layer[ind].bra_delta = self.layer[ind].activate_s_point()*self.layer[ind+1].bra_delta.dot(self.layer[ind+1].weights.transpose())
                    self.layer[ind].weights += lr*self.layer[ind-1].o.transpose().dot(self.layer[ind].bra_delta)
                    pass
##################################################################################
                ##################################################################
                ##################################################################
                #USING EXAMPLE
                #WITH XOR                  
                
    
# Activation functions and their points (proizvodnie):
def sigmoidal(x):
    return 1/(1+np.exp(-x))
def sigmoidal_point(o):
    return o*(1-o)
def heaviside(x):
    bias = 3
    if x<bias:
        return 0
    else:
        return 1
    pass
def heaviside_point(o):
    return 1

#hiperbolic tg
def th(x):
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)
def th_point(o):
    return 1-o**2
def out_activation_fun(x):
    return x
def out_activation_fun_point(o):
    return 1

#write in file
def WRITE(X, filename_x, Y, filename_y):
    fx = open(filename_x, 'w')
    str_X = str(X)
    str_X = str_X.strip('[]')
    fy = open(filename_y, 'w')
    str_Y = str(Y)
    str_Y = str_Y.strip('[]')
    fx.write(str_X)
    fy.write(str_Y)
    fx.close()
    fy.close()
    pass
#grid graphs
def GRID():
    pass
#MAIN MAIN MAIN
#
# 1) creation of train data:
train_X = []
train_Y = []
x = 0
while x<1:
    train_X += [x]
    train_Y += [math.sin(12*x)+0.5]
    x +=0.025

WRITE(train_X, 'dataset/Data_sin_x.csv', train_Y, 'dataset/Data_sin_y.csv')

# x_train_arr, we need move neuron in input, so second input forewer is 1
x_arr = np.array([train_X, np.ones(len(train_X))]).T
bra_y = np.array(train_Y, ndmin=2).T
#
d = [2,15,1]
f = [th, out_activation_fun]
f_p = [th_point, out_activation_fun_point]
#
nrn = NN(d,f,f_p)
#
lr = 0.05
epochs = 10000
nrn.learn(lr, epochs, x_arr, bra_y)

res_y = []
bra_slogs = np.zeros([nrn.d[1], 40])

for k in range(x_arr.shape[0]):
    bra_x = np.array(x_arr[k], ndmin = 2)
    a = nrn.make_calculation(bra_x)[0,0]
    res_y += [a]
    bra_slogs[:, k] = (nrn.layer[2].weights*nrn.layer[1].o.transpose())[:,0]
# check in graph
plt.title("SIN")
plt.xlabel("X")
plt.ylabel("Y")
#

#
plt.plot(train_X, train_Y, label = "Input", color = 'r')
plt.plot(train_X, res_y, label = "RESULT", color = 'g')

for k in range(bra_slogs.shape[0]):
    plt.plot(train_X, bra_slogs[k,:], label = "slog "+str(k), color = 'c')

plt.legend(loc = 2)
plt.show()

