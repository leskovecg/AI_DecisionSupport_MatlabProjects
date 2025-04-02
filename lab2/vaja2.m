clear all 
close all
%%
desired_result = 28; 
max_operations = 5; 

% Parameters of the genetic algorithm
pop_size = 100; % Population size
max_gen = 100; % Maximum number of generations

% Creating the initial population
pop = randi([1 4], pop_size, max_operations);

% GA 
for gen = 1:max_gen
    % Fitness calculation
    values = zeros(pop_size, 1);
    for i = 1:pop_size
        operations = {'+', '-', '*', '/'};
        equation = '1'; % Initial value of the equation
        for j = 1:max_operations
            equation = [equation, operations{pop(i, j)}, num2str(j+1)];
        end
        values(i) = eval(equation);
    end
    fitness = abs(desired_result - values);
    % Select the best
    [min_fit, best_idx] = min(fitness);
    % Check if we have found a solution
    if min_fit == 0
        break;
    end
    % Select parents for the next generation
    fitness = 1 ./ (1 + fitness); % Reverse fitness values so that a smaller fitness is better
    probabilities = fitness / sum(fitness); % Calculation of probability
    cum_prob = cumsum(probabilities);
    indices = arrayfun(@(x) find(cum_prob >= x, 1), rand(pop_size, 1));
    parents = pop(indices, :); % Select parents
    % Perform crossover
    crossover_point = randi([1 max_operations - 1]);
    pop = [parents(:, 1:crossover_point), parents(end:-1:1, crossover_point+1:end)];
    % Perform mutation
    mutation_mask = rand(size(pop)) < 0.01;
    pop(mutation_mask) = randi([1 4], sum(mutation_mask(:)), 1);
end

fprintf('Number of iterations performed: %d\n', gen);
fprintf('Solution in the form of an equation: %s\n', equation);
fprintf('Value of the solution: %.2f\n', values(best_idx));
fprintf('Value of the objective function: %.2f\n', min_fit);