disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%                                                                         %');
disp('%            IMPACT OF DATA STANDARDIZATION ON THE RESULTS                %');
disp('%                                                                         %');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

data = load('VAJA3.mat'); 

% Load data from struct
A_FLOW = data.A_FLOW;
C_ACID = data.C_ACID;
I_EFF = data.I_EFF;
T_H2O = data.T_H2O;

X = [A_FLOW, T_H2O, C_ACID];
Y = I_EFF;

% Compute mean value for each column of X
mean_val = mean(X, 1);

methods = ["range", "z-score", "none"];

% Errors:
errors_pca = [];
errors_pca_transformed = [];
errors_lse = [];
mae_pca = [];
mse_pca = [];
mae_pca_transformed = [];
mse_pca_transformed = [];
mae_lse = [];
mse_lse = [];

% Parameters:
params_pca_a = [];
params_pca_r = [];
params_lse_a = [];
params_lse_r = [];
params_pca_a_transformed = [];
params_pca_r_transformed = [];

% Variances:
variances_lse = [];

for idx = 1:length(methods)

    method = methods(idx);
    
    % Standardization
    if method == "range"
        min_val = min(X);
        max_val = max(X);
        X_standardized = (X - min_val) ./ (max_val - min_val);
    elseif method == "z-score"
        mean_val = mean(X);
        std_val = std(X);
        X_standardized = (X - mean_val) ./ std_val;
    else
        X_standardized = X;
    end
    
    % PCA
    Data = [X_standardized Y];
    meanData = mean(Data);
    centeredData = Data - meanData;
    F = centeredData' * centeredData / (size(Data, 1) - 1);
    [P, ~] = svd(F);
    P1 = P(:, 1);
    a_pca = -P1(1:end-1) / P1(end);
    r_pca = P1' * meanData';
    
    % LSE
    X_augmented = [X_standardized ones(size(X_standardized, 1), 1)];
    theta = (X_augmented' * X_augmented) \ (X_augmented' * Y);
    a_lse = theta(1:end-1);
    r_lse = theta(end);

    % residuals and its variance
    residuals = Y - X_augmented * theta;
    sigma2 = var(residuals);
    
    % variance of the parameters
    var_theta = sigma2 * inv(X_augmented' * X_augmented);
    
    % variance for each parameter
    variances_lse = [variances_lse; diag(var_theta)'];
    
    % Transform back to original space if required
    if method == "range"
        a_pca_transformed = a_pca .* (max_val - min_val)';
        r_pca_transformed = r_pca - sum(a_pca_transformed' .* mean_val) + mean(Y);
    elseif method == "z-score"
        a_pca_transformed = (a_pca' ./ std_val)';
        r_pca_transformed = r_pca - sum(a_pca_transformed' .* mean_val) + mean(Y);
    else
        a_pca_transformed = a_pca;
        r_pca_transformed = r_pca;
    end
    
    % Calculation of errors
    Y_pred_pca = X * a_pca + r_pca;
    Y_pred_pca_transformed = X * a_pca_transformed + r_pca_transformed;
    Y_pred_lse = X * a_lse + r_lse;
    errors_pca = [errors_pca mean(Y - Y_pred_pca)];
    errors_lse = [errors_lse mean(Y - Y_pred_lse)];
    error_pca_transformed = mean(Y - Y_pred_pca_transformed);
    errors_pca_transformed = [errors_pca_transformed, error_pca_transformed];

    % PCA (mae + mse)
    residuals_pca = Y - Y_pred_pca;
    mae_pca = [mae_pca, mean(abs(residuals_pca))];
    mse_pca = [mse_pca, mean(residuals_pca.^2)];
    
    % PCA Transformed (mae + mse)
    residuals_pca_transformed = Y - Y_pred_pca_transformed;
    mae_pca_transformed = [mae_pca_transformed, mean(abs(residuals_pca_transformed))];
    mse_pca_transformed = [mse_pca_transformed, mean(residuals_pca_transformed.^2)];
    
    % LSE (mae + mse)
    residuals_lse = Y - Y_pred_lse;
    mae_lse = [mae_lse, mean(abs(residuals_lse))];
    mse_lse = [mse_lse, mean(residuals_lse.^2)];


    % Store parameters
    r_pca = r_pca(:);  % Ensure r_pca is a column vector
    r_pca_transformed = r_pca_transformed(:);
    r_lse = r_lse(:);  % Ensure r_lse is a column vector

    params_pca_a = [params_pca_a; a_pca'];
    params_pca_r = [params_pca_r; r_pca];
    params_lse_a = [params_lse_a; a_lse'];
    params_lse_r = [params_lse_r; r_lse];
    params_pca_a_transformed = [params_pca_a_transformed; a_pca_transformed'];
    params_pca_r_transformed = [params_pca_r_transformed; r_pca_transformed];
    
    % % Make table
    % comparisonTable = array2table([Y, Y_pred_pca, Y_pred_pca_transformed, Y_pred_lse], ...
    %     'VariableNames', {'Actual_Data', 'PCA_Prediction', 'Transformed_PCA_Prediction', 'LSE_Prediction'});
    % 
    % % Table display
    % disp('Comparison of Actual and Predicted Values:');
    % disp(comparisonTable);


    % Vizualization
    figure;
    subplot(3, 1, 1);
    plot(Y, 'r', 'DisplayName', 'Actual Data');
    hold on;
    plot(Y_pred_pca, 'b', 'DisplayName', 'PCA Prediction');
    title(['PCA Prediction - ', method]);
    legend;

    subplot(3, 1, 2); 
    plot(Y, 'r', 'DisplayName', 'Actual Data');
    hold on;
    plot(Y_pred_pca_transformed, 'm', 'DisplayName', 'Transformed PCA Prediction');
    title(['Transformed PCA Prediction - ', method]);
    legend;

    subplot(3, 1, 3);
    plot(Y, 'r', 'DisplayName', 'Actual Data');
    hold on;
    plot(Y_pred_lse, 'g', 'DisplayName', 'LSE Prediction');
    title(['LSE Prediction - ', method]);
    legend;
end

T_pca = array2table([params_pca_a params_pca_r], 'VariableNames', {'a1', 'a2', 'a3', 'Intercept'}, 'RowNames', {'Range', 'Z-Score', 'None'});
T_pca_transformed = array2table([params_pca_a_transformed params_pca_r_transformed], 'VariableNames', {'a1', 'a2', 'a3', 'Intercept'}, 'RowNames', {'Range', 'Z-Score', 'None'});
T_lse = array2table([params_lse_a params_lse_r], 'VariableNames', {'a1', 'a2', 'a3', 'Intercept'}, 'RowNames', {'Range', 'Z-Score', 'None'});

disp('Parameters for PCA:');
disp(T_pca);

disp('Parameters for PCA (Transformed):');
disp(T_pca_transformed);

disp('Parameters for LSE:');
disp(T_lse);


% Mean Error table
data_for_table = [errors_pca', errors_pca_transformed', errors_lse'];
ErrorTable = array2table(data_for_table, 'VariableNames', {'PCA', 'PCA_Transformed', 'LSE'}, 'RowNames', {'Range', 'Z-Score', 'None'});
disp('Comparison of Mean Errors for Different Methods:');
disp(ErrorTable);

% Create a bar chart for ME
errors_lse_abs = abs(errors_lse);
data_to_plot = [errors_pca', errors_pca_transformed', errors_lse_abs'];
figure;
bar(data_to_plot, 'grouped');
legend('PCA', 'PCA Transformed', 'LSE');
title('Comparison of Mean Errors for Different Methods');
set(gca, 'XTickLabel', methods);
xlabel('Standardization Method');
ylabel('Mean Error');
grid on;

% MAE table
T_mae = array2table([mae_pca', mae_pca_transformed', mae_lse'], 'VariableNames', {'PCA', 'PCA Transformed', 'LSE'}, 'RowNames', {'Range', 'Z-Score', 'None'});
disp('Mean Absolute Errors (MAE) for different methods:');
disp(T_mae);

% Bar chart for MAE
figure;
bar([mae_pca', mae_pca_transformed', mae_lse'], 'grouped');
legend('PCA', 'PCA Transformed', 'LSE');
title('Comparison of Mean Absolute Errors (MAE) for Different Methods');
set(gca, 'XTickLabel', methods);
xlabel('Standardization Method');
ylabel('MAE');
grid on;

% MSE table
T_mse = array2table([mse_pca', mse_pca_transformed', mse_lse'], 'VariableNames', {'PCA', 'PCA Transformed', 'LSE'}, 'RowNames', {'Range', 'Z-Score', 'None'});
disp('Mean Squared Errors (MSE) for different methods:');
disp(T_mse);

% Bar chart for MSE
figure;
bar([mse_pca', mse_pca_transformed', mse_lse'], 'grouped');
legend('PCA', 'PCA Transformed', 'LSE');
title('Comparison of Mean Squared Errors (MSE) for Different Methods');
set(gca, 'XTickLabel', methods);
xlabel('Standardization Method');
ylabel('MSE');
grid on;

% variances for LSE parameters
T_variances_lse = array2table(variances_lse, 'VariableNames', {'a1', 'a2', 'a3', 'Intercept'}, 'RowNames', {'Range', 'Z-Score', 'None'});
disp('Variances for LSE Parameters:');
disp(T_variances_lse);
%% 2nd part of the exercise
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%                                                                         %');
disp('%                       COLLINEARITY PROBLEM                              %');
disp('%                                                                         %');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');


% Adding a new measurement x, which is dependent on T_H2O
x = 2*T_H2O + 6 + 0.1*randn(size(T_H2O, 1));

% Standardization of data
A_FLOW_standard = (A_FLOW - mean(A_FLOW)) / std(A_FLOW);
T_H2O_standard = (T_H2O - mean(T_H2O)) / std(T_H2O);
C_ACID_standard = (C_ACID - mean(C_ACID)) / std(C_ACID);
x_standard = (x - mean(x)) / std(x);

X = [A_FLOW_standard, T_H2O_standard, C_ACID_standard, x_standard];

% LSE method for parameters
THETA_LSE = (X'*X) \ X' * Y; 

% PCA for transformation of data
[coeff,~,~,~,explained] = pca(X);

% Remove the main component with negligible influence
numComponents = find(cumsum(explained) >= 95, 1); 
Ps = coeff(:, 1:numComponents);

% Mapping into the space of main components
T = X * Ps;

% Calculation of parameters with PCR
THETA_PCR_T = (T'*T) \ T' * Y; 
THETA_PCR = Ps * THETA_PCR_T;

disp('LSE parameters:');
disp(THETA_LSE);
disp('PCR parameters:');
disp(THETA_PCR);

% Table for the main components
numVariables = size(X, 2);
numComponents = size(Ps, 2); 
tableData = abs(Ps)./sum(abs(Ps));
columnNames = arrayfun(@(n) sprintf('P_%d', n), 1:numComponents, 'UniformOutput', false);
rowNames = arrayfun(@(n) sprintf('X%d', n), 1:numVariables, 'UniformOutput', false);
T_components = array2table(tableData*100, 'VariableNames', columnNames, 'RowNames', rowNames);
disp('Main Component Contributions to Variables (in percentages):');
disp(T_components);


% table with variances for LSE parameters
theta_LSE = (X'*X) \ X'*Y;
e = Y - X*theta_LSE; %the residuals
n = size(X, 1);     % number of observations
p = size(X, 2);     % number of parameters
sigma2 = (e'*e) / (n-p); % variance
Var_theta_LSE = sigma2 * inv(X'*X); % variance-covariance matrix
variances_LSE = diag(Var_theta_LSE);  % Extract variances from the diagonal
variableNames = arrayfun(@(x) ['a', num2str(x)], 1:p, 'UniformOutput', false);
T_variances_LSE = array2table(variances_LSE, 'VariableNames', {'LSE'}, 'RowNames', variableNames);
disp('Variance for Least Squares (LSE) Parameters:');
disp(T_variances_LSE);


% correlation matrix
R = corrcoef(X);
disp('Correlation matrix:');
disp(R);

% Check if two variables are collinear
threshold = 0.95; % Adjustable threshold for identifying collinearity
[row, col] = find(abs(R) > threshold & abs(R) ~= 1); 

for i = 1:length(row)
    fprintf('Variables X%d and X%d might be collinear with a correlation value of %.2f.\n', row(i), col(i), R(row(i), col(i)));
end


