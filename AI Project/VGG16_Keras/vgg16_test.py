

# %matplotlib inline
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

"""These are the imports from the Keras API. Note the long format which can hopefully be shortened in the future to e.g. `from tf.keras.models import Model`."""

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

# """check if tensorflow is using gpu"""
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())



"""This was developed using Python 3.6 and TensorFlow version:"""

tf.__version__

"""## Helper Functions

### Helper-function for joining a directory and list of filenames.
"""


def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]


"""### Helper-function for plotting images

Function used to plot at most 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
"""


def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


"""### Helper-function for printing confusion matrix"""

# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix


def print_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    print("Confusion matrix:")

    # Print the confusion matrix as text.
    print(cm)

    # Print the class-names for easy reference.
    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))


"""### Helper-function for plotting example errors

Function for plotting examples of images from the test-set that have been mis-classified.
"""


def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != cls_test)

    # Get the file-paths for images that were incorrectly classified.
    image_paths = np.array(image_paths_test)[incorrect]

    # Load the first 9 images.
    images = load_images(image_paths=image_paths[0:9])

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    # Plot the 9 images we have loaded and their corresponding classes.
    # We have only loaded 9 images so there is no need to slice those again.
    plot_images(images=images,
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


"""Function for calculating the predicted classes of the entire test-set and calling the above function to plot a few examples of mis-classified images."""


def example_errors():
    # The Keras data-generator for the test-set must be reset
    # before processing. This is because the generator will loop
    # infinitely and keep an internal index into the dataset.
    # So it might start in the middle of the test-set if we do
    # not reset it first. This makes it impossible to match the
    # predicted classes with the input images.
    # If we reset the generator, then it always starts at the
    # beginning so we know exactly which input-images were used.
    generator_test.reset()

    # Predict the classes for all images in the test-set.
    y_pred = new_model.predict_generator(generator_test,
                                         steps=steps_test)
    # Convert the predicted classes from arrays to integers.
    cls_pred = np.argmax(y_pred, axis=1)

    # Plot examples of mis-classified images.
    plot_example_errors(cls_pred)

    # Print the confusion matrix.
    print_confusion_matrix(cls_pred)


"""### Helper-function for loading images

The data-set is not loaded into memory, instead it has a list of the files for the images in the training-set and another list of the files for the images in the test-set. This helper-function loads some image-files.
"""


def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)


"""### Helper-function for plotting training history

This plots the classification accuracy and loss-values recorded during training with the Keras API.
"""


def plot_training_history(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')

    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show(block=True)


"""The directories where the images are now stored."""

train_dir = "C:/AI 570 PROJECT/Data/train"
test_dir = "C:/AI 570 PROJECT/Data/test"

"""## Pre-Trained Model: VGG16"""

model = VGG16(include_top=True, weights='imagenet')

"""## Input Pipeline"""

input_shape = model.layers[0].output_shape[1:3]
input_shape

datagen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.9, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen_test = ImageDataGenerator(rescale=1. / 255)
batch_size = 20
if True:
    save_to_dir = None
else:
    save_to_dir = 'augmented_images/'

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)

generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

"""Because the data-generators will loop for eternity, we need to specify the number of steps to perform during evaluation and prediction on the test-set. Because our test-set contains 530 images and the batch-size is set to 20, the number of steps is 26.5 for one full processing of the test-set. This is why we need to reset the data-generator's counter in the `example_errors()` function above, so it always starts processing from the beginning of the test-set.

This is another slightly awkward aspect of the Keras API which could perhaps be improved.
"""

steps_test = generator_test.n / batch_size

image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

cls_train = generator_train.classes
cls_test = generator_test.classes

"""Get the class-names for the dataset."""
class_names = list(generator_train.class_indices.keys())
num_classes = generator_train.num_classes


"""### Plot a few images to see if data is correct"""

# Load the first images from the train-set.
images = load_images(image_paths=image_paths_train[0:9])

# Get the true classes for those images.
cls_true = cls_train[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true, smooth=True)

"""### Class Weights

The Knifey-Spoony dataset is quite imbalanced because it has few images of forks, more images of knives, and many more images of spoons. This can cause a problem during training because the neural network will be shown many more examples of spoons than forks, so it might become better at recognizing spoons.

Here we use scikit-learn to calculate weights that will properly balance the dataset. These weights are applied to the gradient for each image in the batch during training, so as to scale their influence on the overall gradient for the batch.
"""

from sklearn.utils.class_weight import compute_class_weight
class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)

"""Note how the weight is about 1.398 for the forky-class and only 0.707 for the spoony-class. This is because there are fewer images for the forky-class so the gradient should be amplified for those images, while the gradient should be lowered for spoony-images."""

print(class_weight)
print(class_names)
print(num_classes)

class_weights = [1, 5.1286504]
#heavy critical = [3.0, 0.7286504]
#lean critical = [2.00000, 0.7286504]
#norm = unchanged
#lean non-critical = [1.0, 0.7286504]
"""## Transfer Learning

The pre-trained VGG16 model was unable to classify images from the Knifey-Spoony dataset. The reason is perhaps that the VGG16 model was trained on the so-called ImageNet dataset which may not have contained many images of cutlery.

The lower layers of a Convolutional Neural Network can recognize many different shapes or features in an image. It is the last few fully-connected layers that combine these featuers into classification of a whole image. So we can try and re-route the output of the last convolutional layer of the VGG16 model to a new fully-connected neural network that we create for doing classification on the Knifey-Spoony dataset.

First we print a summary of the VGG16 model so we can see the names and types of its layers, as well as the shapes of the tensors flowing between the layers. This is one of the major reasons we are using the VGG16 model in this tutorial, because the Inception v3 model has so many layers that it is confusing when printed out.
"""


"""We can see that the last convolutional layer is called 'block5_pool' so we use Keras to get a reference to that layer."""

transfer_layer = model.get_layer('block5_pool')

"""We refer to this layer as the Transfer Layer because its output will be re-routed to our new fully-connected neural network which will do the classification for the Knifey-Spoony dataset.

The output of the transfer layer has the following shape:
"""

transfer_layer.output

"""Using the Keras API it is very simple to create a new model. First we take the part of the VGG16 model from its input-layer to the output of the transfer-layer. We may call this the convolutional model, because it consists of all the convolutional layers from the VGG16 model."""

conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)

"""We can then use Keras to build a new model on top of this."""

# Start a new Keras Sequential model.
new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(conv_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# Add a dropout-layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test-set.
new_model.add(Dropout(0.5))

# Add the final layer for the actual classification.
new_model.add(Dense(num_classes, activation='softmax'))

"""We use the Adam optimizer with a fairly low learning-rate. The learning-rate could perhaps be larger. But if you try and train more layers of the original VGG16 model, then the learning-rate should be quite low otherwise the pre-trained weights of the VGG16 model will be distorted and it will be unable to learn."""

optimizer = Adam(lr=1e-7)

"""We have 3 classes in the Knifey-Spoony dataset so Keras needs to use this loss-function."""

loss = 'categorical_crossentropy'

"""The only performance metric we are interested in is the classification accuracy."""

metrics = ['categorical_accuracy']

"""Helper-function for printing whether a layer in the VGG16 model should be trained."""


def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))


"""By default all the layers of the VGG16 model are trainable."""

print_layer_trainable()

"""In Transfer Learning we are initially only interested in reusing the pre-trained VGG16 model as it is, so we will disable training for all its layers."""

conv_model.trainable = False

for layer in conv_model.layers:
    # Boolean whether this layer is trainable.
    trainable = ('block5' in layer.name or 'block4' in layer.name)

    # Set the layer's bool.
    layer.trainable = trainable
print_layer_trainable()

"""Once we have changed whether the model's layers are trainable, we need to compile the model for the changes to take effect."""

new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

new_model.summary()


"""An epoch normally means one full processing of the training-set. But the data-generator that we created above, will produce batches of training-data for eternity. So we need to define the number of steps we want to run for each "epoch" and this number gets multiplied by the batch-size defined above. In this case we have 100 steps per epoch and a batch-size of 20, so the "epoch" consists of 2000 random images from the training-set. We run 20 such "epochs".

The reason these particular numbers were chosen, was because they seemed to be sufficient for training with this particular model and dataset, and it didn't take too much time, and resulted in 20 data-points (one for each "epoch") which can be plotted afterwards.
"""

epochs = 10
steps_per_epoch = 70


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

"""Training the new model is just a single function call in the Keras API. This takes about 6-7 minutes on a GTX 1070 GPU."""
steps_test = 50

# tf_config = tf.ConfigProto(allow_soft_placement=False)
# tf_config.gpu_options.allow_growth = True
# s = tf.Session(config=tf_config)
# K.set_session(s)
#
# with tf.device('/gpu:0'):
history = new_model.fit_generator(generator=generator_train,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              class_weight=class_weight,
                              validation_data=generator_test,
                              validation_steps=steps_test
                              )

"""Keras records the performance metrics at the end of each "epoch" so they can be plotted later. This shows that the loss-value for the training-set generally decreased during training, but the loss-values for the test-set were a bit more erratic. Similarly, the classification accuracy generally improved on the training-set while it was a bit more erratic on the test-set."""

plot_training_history(history)

"""After training we can also evaluate the new model's performance on the test-set using a single function call in the Keras API."""

print("made it to checkpoint")
#result = new_model.evaluate_generator(generator_test, steps=steps_test)

#print("Test-set classification accuracy: {0:.2%}".format(result[1]))

"""We can plot some examples of mis-classified images from the test-set. Some of these images are also difficult for a human to classify.
The confusion matrix shows that the new model is especially having problems classifying the forky-class.
"""


"""Checkpoint the model, each differently weighted training data will be checkpointed individually. Each will be called
    from another file and run together in a combined fashion, but checkpoints will be created individually"""

filepath = "model_heavy_non_critical.hdf5"
new_model.save(filepath)
print('done')



