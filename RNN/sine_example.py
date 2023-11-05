import numpy as np
import math
import matplotlib.pyplot as plt

# prepare the data (y values of a sine wave)
sin_wave = np.array([math.sin(x) for x in np.arange(200)])

X = []
Y = []

seq_len = 50
num_records = len(sin_wave) - seq_len

for i in range(num_records - 50):
    X.append(sin_wave[i : i + seq_len])
    Y.append(sin_wave[i + seq_len])

X = np.array(X)
X = np.expand_dims(X, axis=2)

Y = np.array(Y)
Y = np.expand_dims(Y, axis=1)

X_val = []
Y_val = []

for i in range(num_records - 50, num_records):
    X_val.append(sin_wave[i : i + seq_len])
    Y_val.append(sin_wave[i + seq_len])

X_val = np.array(X_val)
X_val = np.expand_dims(X_val, axis=2)

Y_val = np.array(Y_val)
Y_val = np.expand_dims(Y_val, axis=1)

# (50,50,1) (50,1)

# RNN architecture
learning_rate = 0.0001
nepoch = 25
T = 50
hidden_dim = 100
output_dim = 1

bptt_truncate = 5
min_clip_value = -10
max_clip_value = 10

# define the weights matrices
U = np.random.uniform(0, 1, (hidden_dim, T))
W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
V = np.random.uniform(0, 1, (output_dim, hidden_dim))


# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# train the model
# 1. check the loss on training data
for epoch in range(nepoch):
    loss = 0

    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]
        prev_s = np.zeros((hidden_dim, 1))

        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulu + mulw
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s
        # calculate error
        loss_per_record = (y - mulv) ** 2 / 2
        loss += loss_per_record
    loss = loss / float(y.shape[0])

# check lost on val
for epoch in range(nepoch):
    loss_val = 0
    for i in range(Y_val.shape[0]):
        # pick sample i
        x, y = X_val[i], Y_val[i]
        prev_s = np.zeros((hidden_dim, 1))
        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulu + mulw
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s
        loss_per_record = (y - mulv) ** 2 / 2
        loss_val += loss_per_record
    loss_val = loss_val / float(y.shape[0])

print("Epoch: ", epoch + 1, ", Loss: ", loss, ", Val Loss: ", loss_val)

# train model
for i in range(Y.shape[0]):
    x, y = X[i], Y[i]

    layers = []
    prev_s = np.zeros((hidden_dim, 1))
    dU = np.zeros(U.shape)
    dV = np.zeros(V.shape)
    dW = np.zeros(W.shape)

    dU_t = np.zeros(U.shape)
    dV_t = np.zeros(V.shape)
    dW_t = np.zeros(W.shape)

    dU_i = np.zeros(U.shape)
    dW_i = np.zeros(W.shape)

    # derivative of pred
    dmulv = mulv - y

    # backward pass
    for t in range(T):
        dV_t = np.dot(dmulv, np.transpose(layers[t]["s"]))
        dsv = np.dot(np.transpose(V), dmulv)

        ds = dsv
        dadd = add * (1 - add) * ds

        dmulw = dadd * np.ones_like(mulw)

        dprev_s = np.dot(np.transpose(W), dmulw)

        for i in range(t - 1, max(-1, t - bptt_truncate - 1), -1):
            ds = dsv + dprev_s
            dadd = add * (1 - add) * ds

            dmulw = dadd * np.ones_like(mulw)
            dmulu = dadd * np.ones_like(mulu)

            dW_i = np.dot(W, layers[t]["prev_s"])
            dprev_s = np.dot(np.transpose(W), dmulw)

            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            dU_i = np.dot(U, new_input)
            dx = np.dot(np.transpose(U), dmulu)

            dU_t += dU_i
            dW_t += dW_i

            dV += dV_t
            dU += dU_t
            dW += dW_t
