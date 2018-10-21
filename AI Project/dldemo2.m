% DL demo 1
tic
% inputting digit data set in Matlab
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...			% define the input data
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...										% define the input data
    'IncludeSubfolders',true,'LabelSource','foldernames');

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% show some images
figure;
perm = randperm(10000,20);				
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

labelCount = countEachLabel(imds)		% finds the number of images in each category

% image size
img = readimage(imds,1);
size(img)


numTrainFiles = 750;					% creates training and validation sets
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize'); 	%the labels are split into training and validation sets also

% network architecture
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding',1) 		% 3 = filter size, this is the size of each kernel, or feature, 8= num filters, 
    batchNormalizationLayer						% batch normalization help normalize the outputs of layers, helping reduce sensetivity to the 
												% initializations 
    reluLayer									% the activation layer
    
    maxPooling2dLayer(2,'Stride',2)				% pools the outputs of the input layer, helps the next layer detect composites of the previous layer's features,
												% so that the feature size keeps getting larger and larger until it detects what we are looking for
												% this logic remains constant for the rest of the architecture initialization
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% training options
options = trainingOptions('sgdm', ...		% train the network using stochastic gradient descent, 4 epochs, provide validation sets
    'MaxEpochs',4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% train network using training data
net = trainNetwork(imdsTrain,layers,options);

% classify validation data
YPred = classify(net,imdsValidation);			% our predictions are hte outputs of the network
YValidation = imdsValidation.Labels;			% the labels for those predictions help us measure error as we train

accuracy = sum(YPred == YValidation)/numel(YValidation)	% measure accuracy using prediction and validation labels
toc