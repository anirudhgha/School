import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

"""We need to import several things from Keras."""

# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from keras import losses

"""This was developed using Python 3.6 (Anaconda) and package versions:"""

tf.__version__
tf.keras.__version__
pd.__version__

# df = [[0 for i in range(74502)] for j in range(3)]
df = pd.DataFrame()

# read from file with stock data
stocks_file = 'nasdaq100_padding.csv'
f = open(stocks_file, 'r')
dfall = pd.read_csv(stocks_file)
f.close()

num_cols = 1  # number of columns (stocks) to read
# drop every column after the 3rd one, just to limit input data, i can extend it later when everything works
for i in range(num_cols, len(dfall.columns)):
    dfall.drop(dfall.columns[num_cols], axis=1, inplace=True)  # df.columns is zero-based pd.Index

# test if variable was set correctly, plot 3 of the 4 remaining stock columns
# print(np.shape(dfall))
# df1 = dfall.iloc[:, 0]
# df2 = dfall.iloc[:, 1]
# df3 = dfall.iloc[:, 2]
# df4 = dfall.iloc[:, 3]
# x = np.arange(0, len(df1), 1)
# plt.plot(x, df1)
# plt.plot(x, df2)
# plt.plot(x, df3)
# plt.plot(x, df4)
# plt.title('Input Sequences')
# plt.show(block=True)

# target will be the 4th column, but shifted back 5 steps (predict 5 days into the future)
shift_steps = 5  # number of steps/days to shift for target data
df_target = dfall.shift(-shift_steps)

# print(df_target.tail(shift_steps))    # confirms that i shifted the target data the right way(tail of target should be nan)

x_data = dfall.values[0:-shift_steps]
y_data = df_target.values[:-shift_steps]

num_data = len(x_data)
train_split = .8  # train with 80% of input data

num_train = int(train_split * num_data)
num_test = num_data - num_train

# -----------------------------------------------------------------------------------
x_train = x_data[0:num_train]  # MAY NEED TO INCLUDE A VALIDATION SET
x_test = x_data[num_train:]
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
# -----------------------------------------------------------------------------------

num_x_signals = x_data.shape[1]
num_y_signals = y_data.shape[1]

"""
scale input data to be between -1 and 1 for improved performance
"""
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

"""
Create random batches of input data to train with, instead of using entire 5000+ input values
"""


def batch_generator_train(batch_size, sequence_length):
    while True:
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        for x in range(batch_size):
            idx = np.random.randint(num_train - sequence_length)
            x_batch[x] = x_train_scaled[idx:idx + sequence_length]
            y_batch[x] = y_train_scaled[idx:idx + sequence_length]
        yield (x_batch, y_batch)


def batch_generator_validation(batch_size, sequence_length):
    while True:
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch_val = np.zeros(shape=x_shape, dtype=np.float16)
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch_val = np.zeros(shape=y_shape, dtype=np.float16)

        for x in range(batch_size):
            idx = np.random.randint(num_test - sequence_length)
            x_batch_val[x] = x_test_scaled[idx:idx + sequence_length]
            y_batch_val[x] = y_test_scaled[idx:idx + sequence_length]
        yield (x_batch_val, y_batch_val)


batch_size = 128
sequence_length = 200  # 2 months of data per batch of training
generator_train = batch_generator_train(batch_size=batch_size, sequence_length=sequence_length)
generator_validation = batch_generator_validation(batch_size=batch_size, sequence_length=sequence_length)

# x_batch, y_batch = next(generator_train)
# print('UNDERNEATH BEGINS X BATCH')
# print(x_batch)
# print('UNDERNEATH BEGINS Y BATCH')
# print(y_batch)


# set some validation data so the lowest error checkpoint can be saved
# validation_data = (np.expand_dims(x_test_scaled, axis=0), np.expand_dims(y_test_scaled, axis=0))

"""
Set up the recurrent neural network model, which should be simple in Keras
"""
model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals)))
model.add(Dense(num_y_signals, activation='sigmoid'))

warmup_steps = 10


def loss_mse_warmup(y_true, y_pred):
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean


optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
model.compile(loss=losses.mean_squared_error, optimizer=optimizer)
model.summary()

# save a checkpoint of the best validation accuracy run
path_checkpoint = 'C:/Users/Anirudh Ghantasala/git_repos/School/ECE629_prediction/Checkpoints/12-1-18/best-model.hdf5'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      save_best_only=True,
                                      mode='min',
                                      save_weights_only=True,
                                      verbose=0)
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5,
                                        verbose=0)

callbacks = [callback_early_stopping,
             callback_checkpoint]
# print(validation_data)
model.fit_generator(generator=generator_train,
                    epochs=30,
                    steps_per_epoch=100,
                    validation_data=generator_validation,
                    validation_steps=30,
                    callbacks=callbacks)

print('done!')
