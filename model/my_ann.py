import numpy as  np

class MyANN:
    def __ini__(self, input_size = 13, hidden_size=16,output_size=3,learning_rate=0.01):
        self.lr = learning_rate

        #Weights
        self.W1 = np.random.rand(hidden_size,input_size)
        self.b1 = np.zeros((hidden_size,1))
        self.W2 = np.random.rand(output_size,hidden_size)
        self.b2 = np.zeros((output_size,1))

    def sigmoid(self,x):# sigmoid fonksiyonu
        return 1 / 1 + np.exp(-x)
    
    def sigmoid_derivative(self,x):#kısmi türev alındı
        return self.sigmoid(x) * (1 - self.sigmoid(z)) 