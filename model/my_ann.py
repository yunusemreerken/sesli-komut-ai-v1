import numpy as  np
import pickle

class MyANN:
    def __init__(self, input_size = 13, hidden_size=32,output_size=3,learning_rate=0.01):
        self.lr = learning_rate

        #Weights
        self.W1 = np.random.rand(hidden_size,input_size)
        self.b1 = np.zeros((hidden_size,1))
        self.W2 = np.random.rand(output_size,hidden_size)
        self.b2 = np.zeros((output_size,1))

    def sigmoid(self,z):# sigmoid fonksiyonu
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self,z):#kısmi türev alındı
        return self.sigmoid(z) * (1 - self.sigmoid(z)) 
    
    def softmax(self,z):
        exp_x = np.exp(z - np.max(z))
        return exp_x / exp_x.sum(axis=0)
    
    def forward(self,x):
        self.Z1 = np.dot(self.W1, x) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.W2,self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def backward(self,x,y):
        y_hat = self.A2
        dZ2 = y_hat - y
        dW2 = np.dot(dZ2, self.A1.T)
        db2 = dZ2

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.sigmoid_derivative(self.Z1)
        dW1 = np.dot(dZ1, x.T)
        db1 = dZ1

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


    def train(self, X, Y, epochs=100):
        for epoch in range(epochs):
            for x,y in zip(X,Y):
                x = x.reshape(-1,1)
                y_one_hot = np.zeros((3,1))
                y_one_hot[y] = 1
                self.forward(x)
                self.backward(x,y_one_hot)
            if(epoch % 10 ==0):
                print(f"Devir {epoch} tamamlandı.")
    
    

    def predict(self,x):
        x = x.reshape(-1,1)
        output = self.forward(x)
        return np.argmax(output)
    
    def save(self, path="model/ann_weights.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.W1, self.b1, self.W2, self.b2), f)

    def load(self, path="model/ann_weights.pkl"):
        with open(path, "rb") as f:
            self.W1, self.b1, self.W2, self.b2 = pickle.load(f)
