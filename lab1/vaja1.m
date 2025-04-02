clear all
close all
%%
% Coordinates of base stations
x1 = 1; y1 = 1; 
x2 = 10; y2 = 5; 
x3 = 2; y3 = 4; 

% Initial estimate for the phone's location
xy = [0, 0];

% Parameters for the Levenberg-Marquardt algorithm
lambda = 0.001; % initial value
max_iters = 1000; 
tol = 1e-6; % Tolerance for checking convergence

% To track changes in parameters
xy_history = zeros(max_iters, 2);
xy_history(1, :) = xy;

for k = 1:max_iters
    % Get distances from the .p files
    d1 = ping_stolp_1;
    d2 = ping_stolp_2;
    d3 = ping_stolp_3;

    % Calculation of the function and the Jacobian
    F = [(xy(1) - x1)^2 + (xy(2) - y1)^2 - d1^2, ...
         (xy(1) - x2)^2 + (xy(2) - y2)^2 - d2^2, ...
         (xy(1) - x3)^2 + (xy(2) - y3)^2 - d3^2];
    J = [2*(xy(1) - x1), 2*(xy(2) - y1); ...
         2*(xy(1) - x2), 2*(xy(2) - y2); ...
         2*(xy(1) - x3), 2*(xy(2) - y3)];
    
    % Calculate the step
    step = (J'*J + lambda*eye(2)) \ (-J'*F');
    
    % Update of the estimate
    xy_new = xy + step';
    
    % Checking convergence
    if norm(xy_new - xy) < tol
        break;
    end
    
    % Update of the lambda parameter
    F_new = [(xy_new(1) - x1)^2 + (xy_new(2) - y1)^2 - d1^2, ...
             (xy_new(1) - x2)^2 + (xy_new(2) - y2)^2 - d2^2, ...
             (xy_new(1) - x3)^2 + (xy_new(2) - y3)^2 - d3^2];
    if norm(F_new) < norm(F)
        lambda = lambda / 10;
    else
        lambda = lambda * 10;
    end
    
    % Update of the estimate
    xy = xy_new;
    
    % Update parameter history
    xy_history(k+1, :) = xy;
end


variance = var(xy_history(1:k+1, :), 0, 1);
fprintf('Variance of x and y parameters: (%.2f, %.2f)\n', variance(1), variance(2));

fprintf('Number of iterations: %d\n', k);

fprintf('The coordinates of the mobile phone are (x, y) = (%.2f, %.2f)\n', xy(1), xy(2));

% Plot parameter changes
figure;
subplot(2,1,1);
plot(1:k+1, xy_history(1:k+1, 1));
xlabel('iteration');
ylabel('x');

subplot(2,1,2);
plot(1:k+1, xy_history(1:k+1, 2));
xlabel('iteration');
ylabel('y');


% Plotting
figure;
% Plot the base stations
plot(x1, y1, 'bo', x2, y2, 'bo', x3, y3, 'bo'); hold on;
% Plot the circles around the base stations
viscircles([x1, y1], d1, 'Color', 'r');
viscircles([x2, y2], d2, 'Color', 'r');
viscircles([x3, y3], d3, 'Color', 'r');
% Plot the estimated positions of the phone during the iterations
plot(xy_history(1:k, 1), xy_history(1:k, 2), 'g'); 
% Plot the final estimated position of the phone
plot(xy(1), xy(2), 'b*'); 
xlabel('x'); ylabel('y');

