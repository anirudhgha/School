Using stock market data for 104 companies, this program will predict a company's stock x minutes into the future at any
given time.

The most difficult part of the project will be pre-processing the input data. The data must be split such that finally
there is x_train, x_test, y_train, and y_test. I will use my own data.

Based on Gated Recurrent Unit (GRU) in Keras.

Notes:
    - Splitting the validation set from the training set may be more efficient than having it overlap with the testing
      set. This is something to change later.
    - ran into an issue with loss giving me NaN for every validation check, it turns out the issue was there were
      missing values in the input data. I was using a file called 'full_no_padding.csv', which contained stock closing
      data per day, but left out spots where there was no data. Switching to 'nasdaq100_padding.csv' fixed the error,
      where missing spots have been averaged out between their gaps. This data is decidedly less accurate, but it allows
      training to work.
    - I set up a validation data generator, which i think will be more useful than having the validation check happen
      across all the validation data every time. This generator randomizes the validation data being used at each batch
      and should provide more true-to-life loss values.

Need to do:
    - Modify the checkpoint methodology to make use of keras model saving, rather than tensorflow saving, may be usefull


References
HVASS Laboratories - https://github.com/Hvass-Labs/TensorFlow-Tutorials
https://www.guru99.com/reading-and-writing-files-in-python.html
stack overflow


----------------------------------------------------------------------------------------------------------------------
EXTRA MATERIAL

# x=np.expand_dims(x_train_scaled, axis=0)
# y=np.expand_dims(y_train_scaled, axis=0)

# model.fit(x_batch, y_batch, batch_size=1, epochs=10, callbacks=callbacks, verbose=1, validation_data=validation_data)
