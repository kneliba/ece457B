# import required packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.python.keras as keras
import time

# make it easier to understand by importing the required libraries within keras
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential

from sklearn.model_selection import KFold

callback = keras.callbacks.EarlyStopping(monitor='loss')
callback2 = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=10,
                                          verbose=0, mode='auto')


def mapping(function):
    i_arr = [10, 40, 80, 200]
    j_arr = [2, 10, 40, 100]

    training_loss_arr = np.zeros((4, 4))
    val_loss_arr = np.zeros((4, 4))

    for i in range(len(i_arr)):
        for j in range(len(j_arr)):

            training_loss = []
            val_loss = []
            for k in range(5):
                if function == "f1":
                    x = np.random.uniform(low=-1, high=1, size=(i_arr[i]))
                    f = x*np.sin(6*np.pi*x)*np.exp(-x**2)
                else:
                    x = np.random.uniform(low=-2, high=2, size=(i_arr[i]))
                    f = np.exp(-x**2)*np.arctan(x)*np.sin(4*np.pi*x)

                n_training = int(0.8*len(x))

                x_training_set = x[:n_training]
                f_training_set = f[:n_training]

                x_test_set = x[n_training:]
                f_test_set = f[n_training:]

                n_folds = n_training if n_training < 10 else 10
                # kfold = KFold(n_splits=n_folds, shuffle=True)

                fold_num = 1
                num_val = n_training // n_folds
                # for train, test in kfold.split(inputs, targets):
                for k in range(n_folds):

                    model = Sequential()
                    model.add(
                        Dense(j_arr[j], activation='relu', input_shape=(1,)))
                    # model.add(Dense(64, activation="relu"))
                    # model.add(Dense(64, activation="relu"))
                    model.add(Dense(1, activation='linear'))

                    model.compile(loss='mean_squared_error',
                                  optimizer='adam')
                    h = model.fit(
                        x_training_set, f_training_set, epochs=5000, batch_size=50, callbacks=[callback], validation_split=0.1, verbose=0)

                    # plt.plot(np.log10(h.history['loss']))
                    # plt.plot(np.log10(h.history['val_loss']), 'r')
                    # plt.legend(['train loss', 'val loss'])

                    scores = model.evaluate(
                        x_test_set, f_test_set, verbose=0)
                    print(
                        f'Score for fold {fold_num}:')

                    print(scores)
                    training_loss.append(h.history['loss'][-1])
                    val_loss.append(h.history['val_loss'][-1])

                    fold_num += 1

                    np.roll(x_training_set, num_val)
                    np.roll(f_training_set, num_val)

            print("i = " + str(i_arr[i]) + " j = " + str(j_arr[j]))
            print(f"Average Training Loss: {np.average(training_loss)}")
            print(f"Average Validation Loss: {np.average(val_loss)}")

            training_loss_arr[i][j] = np.average(training_loss)
            val_loss_arr[i][j] = np.average(val_loss)
    return training_loss_arr, val_loss_arr


def run_best_map(function, i=200, j=40):
    training_loss = []
    val_loss = []
    predict = []

    if function == "f1":
        x = np.random.uniform(low=-1, high=1, size=(i))
        f = x*np.sin(6*np.pi*x)*np.exp(-x**2)
    else:
        x = np.random.uniform(low=-2, high=2, size=(i))
        f = np.exp(-x**2)*np.arctan(x)*np.sin(4*np.pi*x)

    n_training = int(0.8*len(x))

    x_training_set = x[:n_training]
    f_training_set = f[:n_training]

    x_test_set = x[n_training:]
    f_test_set = f[n_training:]

    model = Sequential()
    model.add(
        Dense(j, activation='relu', input_shape=(1,)))
    # model.add(Dense(100, activation="relu"))
    # model.add(Dense(100, activation="relu"))
    # model.add(Dense(100, activation="relu"))
    # model.add(Dense(100, activation="relu"))
    # model.add(Dense(100, activation="relu"))
    # model.add(Dense(100, activation="relu"))
    # model.add(Dense(100, activation="relu"))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    h = model.fit(
        x_training_set, f_training_set, epochs=5000, batch_size=50, callbacks=[callback], validation_split=0.1, verbose=0)

    # plt.plot(np.log10(h.history['loss']))
    # plt.plot(np.log10(h.history['val_loss']), 'r')
    # plt.legend(['train loss', 'val loss'])

    scores = model.evaluate(
        x_test_set, f_test_set, verbose=0)

    print(scores)
    training_loss.append(h.history['loss'][-1])
    val_loss.append(h.history['val_loss'][-1])

    predict = model.predict(x_training_set)
    plt.scatter(x_training_set, f_training_set)
    plt.scatter(x_training_set, predict)
    plt.show()

    print(f"Average Training Loss: {np.average(training_loss)}")
    print(f"Average Validation Loss: {np.average(val_loss)}")

    avg_training_loss = np.average(training_loss)
    avg_val_loss = np.average(val_loss)


# training_loss, val_loss = mapping("f2")

# print(f"Training loss array")
# print(training_loss)
# print(f"Validation loss array")
# print(val_loss)

run_best_map("f1")

# training_loss, val_loss = mapping("f2")

# print(f"Training loss array")
# print(training_loss)
# print(f"Validation loss array")
# print(val_loss)
