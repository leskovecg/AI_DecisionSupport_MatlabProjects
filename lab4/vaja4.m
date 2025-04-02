clear all
close all
%% 
loaded_data = load('data.mat');
data_matrix = loaded_data.data;

% shuffle the rows and separate benign and malignant
data_matrix = data_matrix(randperm(size(data_matrix,1)), :);
benign = data_matrix(data_matrix(:,1) == 2, :);
malignant = data_matrix(data_matrix(:,1) == 4, :);

% training/testing data split (+ shuffle the rows)
train_ratio = 0.7;
train_data = [benign(1:round(train_ratio*size(benign,1)),:); malignant(1:round(train_ratio*size(malignant,1)),:)];
test_data = [benign(round(train_ratio*size(benign,1))+1:end,:); malignant(round(train_ratio*size(malignant,1))+1:end,:)];
train_data = train_data(randperm(size(train_data,1)), :);
test_data = test_data(randperm(size(test_data,1)), :);

% normalize the data
max_vals = max(train_data, [], 1);
min_vals = min(train_data, [], 1);
train_data(:, 2:end) = (train_data(:, 2:end) - min_vals(2:end)) ./ (max_vals(2:end) - min_vals(2:end));
test_data(:, 2:end) = (test_data(:, 2:end) - min_vals(2:end)) ./ (max_vals(2:end) - min_vals(2:end));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Create the neural network
x = train_data(:, 2:end)';
t = (train_data(:, 1) == 4)';  % 1 for malignant, 0 for benign
net = newff(x, t, 5, {'tansig','tansig'}, 'traingd');
% net = newff(x, t, 5, {'purelin','purelin'}, 'traingd');
net = init(net);
net = configure(net, x, t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PSO algorithm

% PSO parameters
nParticles = 30;
nIterations = 1000;
c1 = 1.5;
c2 = 1.5;
w = 0.9;  % inertia weight

% The number of neurons and their connections will dictate the number of variables
nVars = size(x,1)*5 + 5 + 5*1 + 1;  % IW + b1 + LW + b2
lb = -1;
ub = 1;

% Initialize particles
particles(nParticles).position = rand(1, nVars) * (ub-lb) + lb;
particles(nParticles).velocity = rand(1, nVars) * (ub-lb) + lb;
particles(nParticles).bestPosition = particles(nParticles).position;
particles(nParticles).bestScore = Inf;

globalBest.position = rand(1, nVars) * (ub-lb) + lb;
globalBest.score = Inf;

for i = 1:nParticles
    particles(i).position = rand(1, nVars) * (ub-lb) + lb;
    particles(i).velocity = rand(1, nVars) * (ub-lb) + lb;
    particles(i).bestPosition = particles(i).position;
    particles(i).bestScore = Inf;
end

% PSO loop
for iter = 1:nIterations
    for i = 1:nParticles
        % Extract weights and biases from particle's position
        IW = reshape(particles(i).position(1:size(x,1)*5), size(x,1), 5);
        b1 = particles(i).position(size(x,1)*5 + 1:size(x,1)*5 + 5)';
        LW = reshape(particles(i).position(size(x,1)*5 + 5 + 1:size(x,1)*5 + 5 + 5*1), 5, 1);
        b2 = particles(i).position(end);

        % Forward propagation
        a1 = tansig(IW'*x + b1);
        a2 = tansig(LW'*a1 + b2);

        % Compute MSE
        MSE = mean((a2 - t).^2);
        
        if MSE < particles(i).bestScore
            particles(i).bestScore = MSE;
            particles(i).bestPosition = particles(i).position;
        end
        
        if MSE < globalBest.score
            globalBest.score = MSE;
            globalBest.position = particles(i).position;
        end
    end
    
    for i = 1:nParticles
        inertia = w * particles(i).velocity;
        personalAttraction = c1 * rand() * (particles(i).bestPosition - particles(i).position);
        globalAttraction = c2 * rand() * (globalBest.position - particles(i).position);
        
        particles(i).velocity = inertia + personalAttraction + globalAttraction;
        particles(i).position = particles(i).position + particles(i).velocity;
    end
    
    disp(['Iteration ' num2str(iter) ' - Best MSE: ' num2str(globalBest.score)]);
end

% Extract best weights and biases from globalBest for final network
IW_best = reshape(globalBest.position(1:size(x,1)*5), size(x,1), 5);
b1_best = globalBest.position(size(x,1)*5 + 1:size(x,1)*5 + 5)';
LW_best = reshape(globalBest.position(size(x,1)*5 + 5 + 1:size(x,1)*5 + 5 + 5*1), 5, 1);
b2_best = globalBest.position(end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% After PSO optimization, the best-found weights and biases
net.IW{1,1} = IW_best';
net.b{1} = b1_best;
net.LW{2,1} = LW_best';
net.b{2} = b2_best;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Test the net first with updated weigths and biases
%%% net (PSO-trained network)
x_test = test_data(:, 2:end)';
t_test = (test_data(:, 1) == 4)';  % 1 for malignant, 0 for benign

predictions = sim(net, x_test);
predicted_labels = predictions > 0.5; 
accuracy = sum(predicted_labels == t_test) / length(t_test) * 100;

fprintf('Testing Accuracy for net (PSO-trained network): %.2f%%\n', accuracy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  net2 (backpropagation)
net2 = newff(x, t, 5, {'tansig','tansig'}, 'traingd');
net2 = init(net2);
net2 = configure(net2, x, t);

net2 = train(net2, x, t);
predictions_net2 = sim(net2, x_test); 
predicted_labels_net2 = predictions_net2 > 0.5; % convert to binary (1 for malignant, 0 for benign), 0.5 threshold
accuracy_net2 = sum(predicted_labels_net2 == t_test) / length(t_test) * 100;

fprintf('Testing Accuracy for net2 (trained using backpropagation): %.2f%%\n', accuracy_net2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Analysis
targets_pso = [1-t_test; t_test];
pso_outputs = [1-predicted_labels; predicted_labels];

figure;
plotconfusion(targets_pso, pso_outputs, 'PSO Trained Network');
ax = gca; 
ax.XTickLabel = {'Malign', 'Benign', '', ''};
ax.YTickLabel = {'Malign', 'Benign', '', ''};


targets_net2 = [1-t_test; t_test];
net2_outputs = [1-predicted_labels_net2; predicted_labels_net2];

figure;
plotconfusion(targets_net2, net2_outputs, 'MATLAB Trained Network');
ax = gca;
ax.XTickLabel = {'Malign', 'Benign', '', ''};
ax.YTickLabel = {'Malign', 'Benign', '', ''};
