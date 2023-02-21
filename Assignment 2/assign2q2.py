from math import floor
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.python.keras as keras
import random

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

evaluation = [[13.72,
               1.43,
               2.5,
               16.7,
               108,
               3.4,
               3.67,
               0.19,
               2.04,
               6.8,
               0.89,
               2.87,
               1285],
              [12.04,
               4.3,
               2.38,
               22,
               80,
               2.1,
               1.75,
               0.42,
               1.35,
               2.6,
               0.79,
               2.57,
               580],
              [14.13,
               4.1,
               2.74,
               24.5,
               96,
               2.05,
               0.76,
               0.56,
               1.35,
               9.2,
               0.61,
               1.6,
               560]]

my_data = np.loadtxt('randomized_data.txt', delimiter=",", unpack=True)

target = my_data.T[:, 0]
input = my_data.T[:, 1:]

target_classes = to_categorical(
    target, dtype="int32")
target_classes = target_classes[:, 1:]
print(target_classes)


indices_1 = [i for i, x in enumerate(target) if x == 1]
indices_2 = [i for i, x in enumerate(target) if x == 2]
indices_3 = [i for i, x in enumerate(target) if x == 3]

test_indices_1 = indices_1[:floor(len(indices_1)*0.75)]
test_indices_2 = indices_2[:floor(len(indices_2)*0.75)]
test_indices_3 = indices_3[:floor(len(indices_3)*0.75)]

train_indices_1 = indices_1[floor(len(indices_1)*0.75):]
train_indices_2 = indices_2[floor(len(indices_2)*0.75):]
train_indices_3 = indices_3[floor(len(indices_3)*0.75):]

test_indeces = np.concatenate([test_indices_1, test_indices_2])
test_indeces = np.concatenate([test_indeces, test_indices_3])
train_indeces = np.concatenate([train_indices_1, train_indices_2])
train_indeces = np.concatenate([train_indeces, train_indices_3])

random.shuffle(test_indeces)
random.shuffle(train_indeces)


def run_model(model):
    callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0,
                                             patience=100,
                                             verbose=0, mode='auto')

    model = Sequential()
    model.add(
        Dense(512, activation='relu', input_dim=13))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(3, activation='softmax'))

    # print(input[test_indeces])
    # print(target[test_indeces])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    h = model.fit(
        input[test_indeces], target_classes[test_indeces], epochs=5000, batch_size=len(input), callbacks=[callback], validation_split=0.25, verbose=0)

    scores = model.evaluate(
        input[train_indeces], target_classes[train_indeces], verbose=0)
    acc = scores[1] * 100
    loss = scores[0]

    print(f"accuracy: {acc}, val_loss: {loss}, loss: {h.history['loss'][-1]}")
    predictions = model.predict(evaluation)
    print(predictions)


print("4 Layers, descending, relu")
model1 = Sequential()
model1.add(
    Dense(512, activation='relu', input_dim=13))
model1.add(Dense(128, activation="relu"))
model1.add(Dense(64, activation="relu"))
model1.add(Dense(3, activation='softmax'))

run_model(model1)

print("5 Layers, desending, relu")
model2 = Sequential()
model2.add(
    Dense(512, activation='relu', input_dim=13))
model2.add(Dense(254, activation="relu"))
model2.add(Dense(128, activation="relu"))
model2.add(Dense(64, activation="relu"))
model2.add(Dense(32, activation="relu"))
model2.add(Dense(3, activation='softmax'))

run_model(model2)

print("5 Layers, desending, linear")
model3 = Sequential()
model3.add(
    Dense(512, activation='relu', input_dim=13))
model3.add(Dense(254, activation="linear"))
model3.add(Dense(128, activation="linear"))
model3.add(Dense(64, activation="linear"))
model3.add(Dense(32, activation="linear"))
model3.add(Dense(3, activation='softmax'))
run_model(model3)

print("5 Layers, 100 nodes, relu")
model4 = Sequential()
model4.add(
    Dense(100, activation='relu', input_dim=13))
model4.add(Dense(100, activation="relu"))
model4.add(Dense(100, activation="relu"))
model4.add(Dense(100, activation="relu"))
model4.add(Dense(100, activation="relu"))
model4.add(Dense(3, activation='softmax'))

run_model(model4)

print("3 Layers, 100 nodes, relu")
model5 = Sequential()
model5.add(
    Dense(100, activation='relu', input_dim=13))
model5.add(Dense(100, activation="relu"))
model5.add(Dense(100, activation="relu"))
model5.add(Dense(3, activation='softmax'))
run_model(model5)

print("1 Layer, 100 nodes, relu")
model6 = Sequential()
model6.add(
    Dense(100, activation='relu', input_dim=13))
model6.add(Dense(3, activation='softmax'))
run_model(model6)

print("1 Layer, 20 nodes, relu")
model6 = Sequential()
model6.add(
    Dense(20, activation='relu', input_dim=13))
model6.add(Dense(3, activation='softmax'))
run_model(model6)
# plt.plot(np.log10(h.history['loss']))
# plt.plot(np.log10(h.history['val_loss']), 'r')
# plt.legend(['train loss', 'val loss'])
# plt.show()
