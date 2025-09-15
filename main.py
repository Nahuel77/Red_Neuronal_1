import pandas as pd
import numpy as np

train = pd.read_csv('data_set/train.csv')
test = pd.read_csv('data_set/test.csv')

y_train = train['label'].values
x_train = train.drop(columns=['label']).values / 255.0

def one_hot_encode(y, num_classes=10):
    y_encoded = np.zeros((y.shape[0], num_classes))
    y_encoded[np.arange(y.shape[0]), y] = 1
    return y_encoded

y_train_encoded = one_hot_encode(y_train)

input_size = 784
hidden_size = 128
hidden_size2 = 64
output_size = 10

#capa oculta 1
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))

#capa oculta 2
W2 = np.random.rand(hidden_size, hidden_size2) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, hidden_size2))

#capa de salida
W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
b3 = np.zeros((1, output_size))

#def sigmoid(x):
#    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivate(x):
    return (x>0).astype(float)

def softmax(x):
    exp_x = np.exp(x-np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#Forward Pass
#Z1 = np.dot(x_train, W1) + b1 #Matriz suma ponderada de la primera capa por los pesos + b1 (matriz de ajuste)
#A1 = sigmoid(Z1)

#Z2 = np.dot(A1, W2) + b2
#A2 = softmax(Z2)

def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8))/m

#Bucle de entrenamiento

learning_rate = 0.1
epochs = 100
for epoch in range(epochs):
    #forward
    Z1=np.dot(x_train, W1)+b1
    A1=relu(Z1)
    
    Z2=np.dot(A1, W2)+b2
    A2=relu(Z2)
    
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    
    #Loss
    loss = cross_entropy(y_train_encoded, A3)
    #Backprop
    m = y_train.shape[0]
    
    dZ3 = A3 - y_train_encoded
    dW3 = np.dot(A2.T, dZ3)/ m
    db3 = np.sum(dZ3, axis=0, keepdims=True)/ m
    
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivate(Z2)
    dW2 = np.dot(A1.T, dZ2)/m
    db2 = np.sum(dZ2, axis=0, keepdims=True)/m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivate(Z1)
    dW1 = np.dot(x_train.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    # Update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Loss: ", loss)
