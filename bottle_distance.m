function detect_incomplete_bottles(image_path, fill_threshold)
    if nargin < 2
        fill_threshold = 0.9; % Default fill threshold
    end
    
    % Read the image
    image = imread('BOTTLE PICTURE.jpg');
    gray = rgb2gray(image);

    % Edge detection (Canny)
    edges = edge(gray, 'Canny');

    % Find contours (bottles)
    bottle_stats = regionprops(edges, 'BoundingBox');
    
    % Convert to HSV for liquid detection
    hsv_image = rgb2hsv(image);
    
    % Define threshold for detecting liquid (adjust based on liquid color)
    lower_liquid = 0.2; % Adjust for actual color
    upper_liquid = 0.8;
    
    % Extract saturation and value channels
    liquid_mask = (hsv_image(:,:,2) > lower_liquid) & (hsv_image(:,:,3) < upper_liquid);
    
    % Find contours for liquid levels
    liquid_stats = regionprops(liquid_mask, 'BoundingBox');

    % Display the original image
    figure;
    imshow(image);
    hold on;
    
    % Process detected bottles
    for i = 1:length(bottle_stats)
        bottle_box = bottle_stats(i).BoundingBox;
        x_b = bottle_box(1);
        y_b = bottle_box(2);
        w_b = bottle_box(3);
        h_b = bottle_box(4);

        for j = 1:length(liquid_stats)
            liquid_box = liquid_stats(j).BoundingBox;
            x_l = liquid_box(1);
            y_l = liquid_box(2);
            w_l = liquid_box(3);
            h_l = liquid_box(4);

            % Ensure liquid is inside the bottle
            if x_l >= x_b && (x_l + w_l) <= (x_b + w_b) && ...
               y_l >= y_b && (y_l + h_l) <= (y_b + h_b)
               
                fill_ratio = h_l / h_b; % Liquid height compared to bottle height
                
                % Mark incompletely filled bottles
                if fill_ratio < fill_threshold
                    rectangle('Position', bottle_box, 'EdgeColor', 'r', 'LineWidth', 2);
                    text(x_b, y_b - 10, 'Incomplete', 'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');
                end
            end
        end
    end
    
    title('Labeled Incompletely Filled Bottles');
    hold off;
end

% Example usage
detect_incomplete_bottles('image.png');
