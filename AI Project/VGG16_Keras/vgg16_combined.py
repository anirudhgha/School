# %matplotlib inline
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np

from keras.models import load_model
from keras.optimizers import Adam, SGD
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import cv2


def predict_class(filep):
    img = cv2.imread(filep)
    cv2.imshow('image', img)
    new_shape = (224, 224, 3)
    img_resized = resize(img, new_shape)

    # img_resized = img_resized.reshape((-1, 224, 224, 3))
    # cv2.imshow('image2', img_resized)
    # cv2.waitKey(0)

    img_redim = np.expand_dims(img_resized, axis=0)

    pred1 = model1.predict(img_redim, batch_size=None, verbose=0, steps=1)
    pred2 = model2.predict(img_redim, batch_size=None, verbose=0, steps=1)
    pred3 = model3.predict(img_redim, batch_size=None, verbose=0, steps=1)
    pred4 = model4.predict(img_redim, batch_size=None, verbose=0, steps=1)
    pred5 = model5.predict(img_redim, batch_size=None, verbose=0, steps=1)

    print(pred1)
    print(pred2)
    print(pred3)
    print(pred4)
    print(pred5)

    avg = assign_crit_index(pred1, pred2, pred3, pred4, pred5) * 10
    return avg


def assign_crit_index(pred1, pred2, pred3, pred4, pred5):
    summer = 0
    summer += pred1[0, 0] + pred2[0, 0] + pred3[0, 0] + pred4[0, 0] + pred5[0, 0]
    return summer / 5


"""load the checkpoints that we trained on the medical image data"""
model1 = load_model('model_heavy_critical.hdf5')
model2 = load_model('model_lean_critical.hdf5')
model3 = load_model("model_norm.hdf5")
model4 = load_model("model_lean_non_critical.hdf5")
model5 = load_model("model_heavy_non_critical.hdf5")

# optimizer = Adam(lr=1e-5)
# loss = 'categorical_crossentropy'
# metrics = ['categorical_accuracy']
# model1.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# model2.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# model3.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# model4.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# model5.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print('10 is most critical, 0 is least critical')
print('[nv = not crit, not_nv = crit]')
# image 1
filepath = './final_images/nv1.jpg'
pred = predict_class(filepath)
print('image1 ', filepath, 'critical index: ', pred)

# image 2
filepath = './final_images/nv2.jpg'
pred = predict_class(filepath)
print('image2 ', filepath, 'critical index: ', pred)

# image 3
filepath = './final_images/nv3.jpg'
pred = predict_class(filepath)
print('image3 ', filepath, 'critical index: ', pred)

# image 4
filepath = './final_images/nv4.jpg'
pred = predict_class(filepath)
print('image2 ', filepath, 'critical index: ', pred)

# image 5
filepath = './final_images/nv5.jpg'
pred = predict_class(filepath)
print('image3 ', filepath, 'critical index: ', pred)

# image 6
filepath = './final_images/not_nv1.jpg'
pred = predict_class(filepath)
print('image4 ', filepath, 'critical index: ', pred)

# image 7
filepath = './final_images/not_nv2.jpg'
pred = predict_class(filepath)
print('image5 ', filepath, 'critical index: ', pred)

# image 8
filepath = './final_images/not_nv3.jpg'
pred = predict_class(filepath)
print('image6 ', filepath, 'critical index: ', pred)

# image 9
filepath = './final_images/not_nv4.jpg'
pred = predict_class(filepath)
print('image5 ', filepath, 'critical index: ', pred)

# image 10
filepath = './final_images/not_nv5.jpg'
pred = predict_class(filepath)
print('image6 ', filepath, 'critical index: ', pred)