
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from keras import losses




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

# target will be the 4th column, but shifted back 5 steps (predict 5 days into the future)
shift_steps = 5  # number of steps/days to shift for target data
df_target = dfall.shift(-shift_steps)

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

#-----------------------------------------------------------------------------------------
# Load the model

model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals)))
model.add(Dense(num_y_signals, activation='sigmoid'))

model.load_weights('C:/Users/Anirudh Ghantasala/git_repos/School/ECE629_prediction/Checkpoints/12-1-18/best-model.hdf5')
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
model.compile(loss=losses.mean_squared_error, optimizer=optimizer)

model.summary()
# result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
#                         y=np.expand_dims(y_test_scaled, axis=0))

def plot_comparison(start_idx, length=100, train=True):

    if train:
        x = x_train_scaled
        y_true = y_train
    else:
        x = x_test_scaled
        y_true = y_test
    print(x)
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
        # p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        #plt.ylabel('column ', signal)
        plt.legend()
        plt.title('Predicted Vs. Training Stock (AAL)')
        print(signal)
        plt.show(block=True)

plot_comparison(start_idx=100, length=3000, train=False)
























