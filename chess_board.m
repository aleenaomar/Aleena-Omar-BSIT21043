% Initialize a 100x100 matrix with zeros
A = zeros(100, 100); 

% Define the circle parameters
Cx = 50;  % X-coordinate of the circle center
Cy = 50;  % Y-coordinate of the circle center
Radius = 20; % Radius of the circle

% Iterate through each pixel in the matrix
for i = 1:100
    for j = 1:100
        % Calculate the Chessboard distance from (i, j) to the center (Cx, Cy)
        distance = max(abs(Cx - i), abs(Cy - j));
        
        % Check if the distance is less than or equal to the radius
        if distance <= Radius
            A(i, j) = 255; % Set the pixel value to 255 (white)
        end
    end
end

% Display the resulting image
imshow(A, []);
