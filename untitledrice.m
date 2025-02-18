% Load the input image
img = imread('ricepic.jpg'); % Replace with your image filename

% Ensure the image is grayscale
if size(img, 3) == 3
    img_gray = rgb2gray(img); % Convert to grayscale if it's RGB
else
    img_gray = img; % Already grayscale
end

% Convert the grayscale image to double for mathematical operations
img_double = double(img_gray);

% 1. Negative Transformation
negative_img = 255 - img_gray;

% 2. Log Transformation
c = 255 / log(1 + max(img_double(:))); % Scaling factor
log_img = c * log(1 + img_double);
log_img = uint8(log_img); % Convert back to uint8 for display

% 3. Gamma Correction
gamma = 2.5; % Adjust gamma value as needed
gamma_img = 255 * (img_double / 255).^gamma;
gamma_img = uint8(gamma_img); % Convert back to uint8

% 4. Contrast Stretching
min_intensity = min(img_double(:));
max_intensity = max(img_double(:));
contrast_img = uint8(255 * (img_double - min_intensity) / (max_intensity - min_intensity));

% Display results
figure;
subplot(2,3,1), imshow(img_gray), title('Original Image');
subplot(2,3,2), imshow(negative_img), title('Negative Transformation');
subplot(2,3,3), imshow(log_img), title('Log Transformation');
subplot(2,3,4), imshow(gamma_img), title('Gamma Correction');
subplot(2,3,5), imshow(contrast_img), title('Contrast Stretching');

