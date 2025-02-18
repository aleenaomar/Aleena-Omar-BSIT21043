A = zeros(100, 100); % Initialize a 100x100 matrix with zeros
Cx = 50;             % X-coordinate of the circle center
Cy = 50;             % Y-coordinate of the circle center
Radius = 20;         % Radius of the circle

for i = 1:100
    for j = 1:100
        % Calculate the Euclidean distance from (i, j) to the center (Cx, Cy)
        distance = sqrt((Cx - i)^2 + (Cy - j)^2);
        
        % If the distance is less than or equal to the radius, set pixel value to 255
        if distance <= Radius
            A(i, j) = 255;
        end
    end
end

% Display the matrix as an image
imshow(A, []);
