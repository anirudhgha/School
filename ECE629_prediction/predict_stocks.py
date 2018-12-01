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
stocks_file = 'C:/Users/alasg/Downloads/full_non_padding.csv'
f = open(stocks_file, 'r')
dfall = pd.read_csv(stocks_file)
f.close()

print(np.shape(dfall))

# drop every column after the 3rd one, just to limit input data, i can extend it later when everything works
for i in range(10, 105):
    dfall.drop(dfall.columns[10], axis=1, inplace=True)  # df.columns is zero-based pd.Index

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
# plt.title('Input sequence of 4 stocks')
# plt.show(block=True)

# target will be the 4th column, but shifted back 5 steps (predict 5 days into the future)
shift_steps = 2  # number of steps/days to shift for target data
df_target = dfall.shift(-shift_steps)

# print(df_target.tail(shift_steps))    # confirms that i shifted the target data the right way(tail of target should be nan)

x_data = dfall.values[0:-shift_steps]
y_data = df_target.values[:-shift_steps]

num_data = len(x_data)
train_split = .7  # train with 70% of input data

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


def batch_generator(batch_size, sequence_length):
    while True:
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        for i in range(batch_size):
            idx = np.random.randint(num_train - sequence_length)
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]
        yield (x_batch, y_batch)


batch_size = 50
sequence_length = 61 # 2 months of data per batch of training
generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length)


#set some validation data so the lowest error checkpoint can be saved
validation_data = (np.expand_dims(x_test_scaled, axis=0), np.expand_dims(y_test_scaled, axis=0))


"""
Set up the recurrent neural network model, which should be simple in Keras
"""
model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))
model.add(Dense(num_y_signals, activation='sigmoid'))

warmup_steps = 10


def loss_mse_warmup(y_true, y_pred):
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

optimizer = RMSprop(lr=1e-3)
model.compile(loss=losses.mean_squared_error, optimizer=optimizer)
model.summary()

# save a checkpoint of the best validation accuracy run
path_checkpoint = 'best_val_checkpoint'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)
#stop run when validation gets worse for 5 batches
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_reduce_lr]

model.fit_generator(generator=generator,
                    epochs=20,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks)


#if the weights worsened for many batches before stopping, we want to reload the optimal weights
try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

#run the network!
result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))



# Test whether the network learned how to predict the stock market!
def plot_comparison(start_idx, length=100, train=True):

    if train:
        x = x_train_scaled
        y_true = y_train
    else:
        x = x_test_scaled
        y_true = y_test

    end_idx = start_idx + length
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    for signal in range(num_y_signals):
        signal_pred = y_pred_rescaled[:, signal]
        signal_true = y_true[:, signal]
        plt.figure(figsize=(15, 5))
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        plt.ylabel('column ', signal)
        plt.legend()
        plt.show()

plot_comparison(start_idx=100, length=1000, train=True)














