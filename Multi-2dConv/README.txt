The idea is to produce more accurate results by combining the outputs of multiple 2d convolutional networks as compared
to using just one.

Taking a standard ResNet architecture, I'd like to modify the number of convolutional layers in it, ranging from a
network that is too shallow (over generalizes the training data) to a network that is too large (over fits to the
training data), and take the average of all of these outputs.

The idea is that the average should be a floating point value that should provide a 'critical' index that is of much
higher resolution than otherwise possible. This way, images that would have been classified with the same critical
index can be separated and traiged more accurately.

Things like the criticalness of an mri scan can only be labelled generally (e.g not critical, semi-critical, critical);
this provides a mechanism for converting those accurate, general labels into a floating point number.


The idea combines the ideas of paper 1 and paper 3. Originally, the idea was to do this with an MD-LSTM, but i've
realized that classification tasks with lstms are very difficult, since a pixel's lstm output will only be based
on the pixels surrounding it, not every pixel in the image.


NEXT ON THE AGENDA
    Find a dataset that contains images with some "critical index" that labels some as critical and some as not.
    May have to label some 100 of these yourself that you can use to train the networks on :/