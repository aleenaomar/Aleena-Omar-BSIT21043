clc; clear; close all;

% Load the image
img = imread('fruits_vegetables.jpg'); % Replace with your image file
imshow(img);
title('Original Image');

% Load a pre-trained deep learning model
net = googlenet; % You can use other models like resnet50

% Load an object detector (you can train your own or use a pre-trained one)
detector = vision.CascadeObjectDetector(); 

% Detect objects in the image
bbox = step(detector, img); 

% Perform classification on each detected object
for i = 1:size(bbox,1)
    % Crop the detected object
    croppedImg = imcrop(img, bbox(i, :));
    
    % Resize for classification
    resizedImg = imresize(croppedImg, [224 224]); 
    
    % Classify the object
    label = classify(net, resizedImg);
    
    % Draw bounding box and label
    img = insertObjectAnnotation(img, 'rectangle', bbox(i, :), char(label), 'Color', 'red', 'FontSize', 12);
end

% Show labeled image
figure;
imshow(img);
title('Labeled Fruits and Vegetables');
