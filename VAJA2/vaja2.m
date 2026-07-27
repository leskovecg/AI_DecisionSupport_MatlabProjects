clc; clear; close all;

% =========================================================================
% VAJA 2 - Genetski algoritem za iskanje aritmeticnega izraza
% -------------------------------------------------------------------------
% Skripta z genetskim algoritmom sestavi aritmeticni izraz iz stevk 1-9 in
% operacij +, -, *, /, katerega vrednost se cim bolj priblizna ciljni
% vrednosti. Rezultate izvozi v tabelo LaTeX in graf poteka kriterijske
% funkcije.
% =========================================================================

% === PARAMETRI ALGORITMA ===
% Priporocene vrednosti: populacija 20-100 osebkov, 100-1000 generacij,
% verjetnost mutacije 0.05-0.2. Vecja populacija pomeni vecjo zacetno
% diverziteto in manjse tveganje prezgodnje konvergence, a vec racunanja.
population_size = 20;      % stevilo osebkov v eni generaciji (sodo zaradi parjenja)
max_generations = 100;     % zgornja meja stevila generacij
mutation_prob   = 0.1;     % verjetnost spremembe posameznega gena

% Elitizem: najboljsi osebek generacije se nespremenjen prenese naprej.
% Posledica je, da najboljsa dosezena kriterijska funkcija nikoli ne pade,
% cena pa je manjsa diverziteta populacije.
use_elitism = true;

% === UPORABNISKI VNOS ===
% Vnos se bere kot niz in pretvori s str2double, ker privzeti nacin input()
% interno klice eval() nad vnesenim tekstom, kar v kombinaciji z lokalnimi
% funkcijami na dnu te skripte v nekaterih okoljih povzroci napacen
% re-parse datoteke in napako v evaluate_individual.
target_value = str2double(input('Vnesi željeno ciljno vrednost (npr. 28): ', 's'));
while ~(isscalar(target_value) && isfinite(target_value))
    fprintf('Napačen vnos: ciljna vrednost mora biti število.\n');
    target_value = str2double(input('Vnesi željeno ciljno vrednost (npr. 28): ', 's'));
end

% Najvecje stevilo stevk v izrazu. Pri max_terms = 5 izraz izgleda npr.
% takole: 9 + 5 * 4 - 3 + 2
max_terms = str2double(input('Vnesi največje število členov v izrazu (npr. 5): ', 's'));
while ~(isscalar(max_terms) && isfinite(max_terms) && max_terms >= 1 && mod(max_terms,1) == 0)
    fprintf('Napačen vnos: največje število členov mora biti celo število >= 1.\n');
    max_terms = str2double(input('Vnesi največje število členov v izrazu (npr. 5): ', 's'));
end

% === ZACETNA POPULACIJA ===
% Prva generacija je nakljucna, kar zagotovi razprsenost po iskalnem prostoru.
tic;
pop = initialize_population(population_size, max_terms);

% Zgodovina najboljsega osebka po generacijah za graf in tabelo.
best_fitness_history = zeros(max_generations,1);
best_expressions = strings(max_generations,1);
best_values = zeros(max_generations,1);

% Zastavica loci predcasno najdeno resitev od izteka stevila generacij.
solution_found = false;

% === GLAVNA ZANKA GA ===
% Ena iteracija zanke je ena generacija.
for gen = 1:max_generations

    % Ovrednotenje: vsakemu osebku pripada vrednost izraza in kriterijska
    % funkcija, ki meri, kako blizu cilju je.
    fitnesses = zeros(population_size,1);
    values = zeros(population_size,1);

    for i = 1:population_size
        [v, f] = evaluate_individual(pop{i}, target_value);
        values(i) = v;
        fitnesses(i) = f;
    end

    % Najboljsi osebek trenutne generacije.
    [best_fit, idx] = max(fitnesses);

    best_fitness_history(gen) = best_fit;
    best_expressions(gen) = strjoin(pop{idx},'');
    best_values(gen) = values(idx);

    % Ce se vrednost izraza ujema s ciljem, nadaljnje iskanje nima smisla.
    if abs(values(idx) - target_value) < 1e-6
        solution_found = true;
        break;
    end

    % === TVORJENJE NOVE GENERACIJE ===
    % Zaporedje operatorjev GA: selekcija, krizanje, mutacija.
    new_pop = cell(population_size,1);

    if use_elitism
        new_pop{1} = pop{idx};
        start_i = 2;
    else
        start_i = 1;
    end

    % Otroci nastajajo v parih, ker krizanje iz dveh starsev tvori dva otroka.
    for i = start_i:2:population_size

        % Selekcija z ruletnim kolesom: verjetnost izbire osebka je
        % sorazmerna z njegovo kriterijsko funkcijo, zato boljsi osebki
        % pogosteje prispevajo gene, slabsi pa vseeno ohranijo moznost
        % izbire in s tem raznolikost genskega zaklada.
        p1 = roulette_selection(pop, fitnesses);
        p2 = roulette_selection(pop, fitnesses);

        % Krizanje kombinira gradnike dveh delnih resitev.
        [c1, c2] = crossover(p1, p2);

        % Mutacija vnasa nove gene, ki jih v populaciji ni bilo, in tako
        % preprecuje zastoj v lokalnem optimumu.
        new_pop{i} = mutate(c1, mutation_prob);

        if i+1 <= population_size
            new_pop{i+1} = mutate(c2, mutation_prob);
        end
    end

    pop = new_pop;
end

elapsed_time = toc;

% === IZPIS REZULTATA ===
% Koncni rezultat se izpise v obeh primerih, tudi ce popolna resitev
% ni bila najdena.
if solution_found
    fprintf("\n Rešitev najdena v generaciji %d\n", gen);
else
    fprintf("\n Doseženo največje število generacij (%d) brez popolne rešitve.\n", max_generations);
    fprintf(" Izpisujem najboljšo doslej najdeno rešitev.\n");
end
fprintf('Število opravljenih generacij: %d\n', gen);
disp(['Izraz: ', char(best_expressions(gen))]);
disp(['Vrednost izraza: ', num2str(best_values(gen))]);
disp(['Kriterijska funkcija: ', num2str(best_fitness_history(gen))]);
fprintf('Čas izvajanja algoritma: %.4f s\n', elapsed_time);

% === POTI ===
% Poti se dolocijo relativno glede na lokacijo te skripte, zato delovanje
% ni odvisno od uporabnikovega racunalnika/mape.
script_dir = fileparts(mfilename('fullpath'));
tab_dir = fullfile(script_dir, 'tabs');
fig_dir = fullfile(script_dir, 'figs');

if ~exist(tab_dir, 'dir')
    mkdir(tab_dir);
end
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

% === IZVOZ TABELE ===
tex_file = fullfile(tab_dir, 'najboljsi_rezultati.tex');
fid = fopen(tex_file, 'w');

fprintf(fid, '\\begin{tabular}{|c|l|c|c|}\\hline\n');
fprintf(fid, 'Generacija & Izraz & Vrednost & Kriterijska funkcija \\\\ \\hline\n');

for i = 1:gen
    fprintf(fid, '%d & %s & %.4f & %.4f \\\\ \\hline\n', ...
        i, best_expressions(i), best_values(i), best_fitness_history(i));
end

fprintf(fid, '\\end{tabular}\n');
fclose(fid);

% === GRAF POTEKA KRITERIJSKE FUNKCIJE ===
fig = figure('Color', 'w');

plot(1:gen, best_fitness_history(1:gen), 'k-', 'LineWidth', 2);
xlabel('Generacija', 'FontSize', 12, 'Color', 'k');
ylabel('Najboljša vrednost kriterijske funkcije', 'FontSize', 12, 'Color', 'k');
title('Potek izboljševanja najboljše rešitve', 'FontSize', 13, 'Color', 'k');
grid on;
set(gca, 'FontSize', 12, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');

saveas(fig, fullfile(fig_dir, 'potek_kriterijske_funkcije.png'));

% === FUNKCIJE ===

function pop = initialize_population(N, max_terms)
    % Razpolozljive aritmeticne operacije.
    ops = {'+', '-', '*', '/'};

    % Populacija je celica, vsak element je en kandidat za resitev.
    pop = cell(N,1);

    for i = 1:N
        % Izraz ima obliko stevka operator stevka ... stevka,
        % torej je skupna dolzina 2*max_terms - 1.
        expr = cell(1, max_terms*2-1);

        for j = 1:max_terms
            % Liha mesta so stevke 1-9.
            expr{2*j-1} = num2str(randi([1,9]));

            % Soda mesta so operatorji.
            if j < max_terms
                expr{2*j} = ops{randi(4)};
            end
        end

        pop{i} = expr;
    end
end

function [value, fitness] = evaluate_individual(indiv, target)
    try
        % Genotip (celica simbolov) se pretvori v fenotip (niz),
        % npr. {'9','+','5','*','4'} -> '9+5*4'.
        expr = strjoin(indiv,'');

        value = eval(expr);

        % Kriterijska funkcija 1/(1+|cilj-vrednost|) je padajoca funkcija
        % absolutne napake, omejena na (0,1]. Popoln zadetek da vrednost 1,
        % z oddaljenostjo od cilja pa gladko pada proti 0, zato je primerna
        % za ruletno kolo, ki zahteva nenegativne utezi.
        fitness = 1 / (1 + abs(target - value));

    catch
        % Neveljaven izraz (npr. deljenje z nic) dobi najslabso oceno,
        % s cimer ga selekcija prakticno izloci.
        value = NaN;
        fitness = 0;
    end
end

function selected = roulette_selection(pop, fitnesses)
    % Ruletno kolo: velikost izseka je sorazmerna kriterijski funkciji
    % osebka, izbira pa je nakljucna, zato je selekcijski pritisk mehak -
    % boljsi osebki so v prednosti, slabsi pa niso povsem izkljuceni.

    total_fitness = sum(fitnesses);

    % Ce so vsi fitnessi 0 (ali je vsota NaN, npr. zaradi neveljavnih
    % izrazov), ruletna izbira ni definirana, zato uporabimo enakomerno
    % nakljucno izbiro med osebki.
    if total_fitness == 0 || isnan(total_fitness)
        idx = randi(numel(pop));
    else
        probs = fitnesses / total_fitness;     % normirane verjetnosti
        cum_probs = cumsum(probs);             % kumulativne verjetnosti
        r = rand;
        idx = find(cum_probs >= r, 1, 'first');
    end

    selected = pop{idx};
end

function [child1, child2] = crossover(p1, p2)
    % Enotockovno krizanje: izbere se tocka reza, repa starsev se zamenjata.
    % Tako se v enem otroku zdruzita gradnika dveh razlicnih delnih resitev.

    % Pri izrazu z eno samo stevko (length(p1) == 1) krizanje nima smisla,
    % hkrati bi randi(0) javil napako, zato otroka podedujeta starsa.
    if length(p1) <= 1
        child1 = p1;
        child2 = p2;
        return;
    end

    % Tocka reza je na lihem mestu, da ostane zgradba izraza smiselna
    % (stevka/operator/stevka/...).
    point = 2*randi(floor(length(p1)/2)) - 1;

    child1 = [p1(1:point), p2(point+1:end)];
    child2 = [p2(1:point), p1(point+1:end)];
end

function mutated = mutate(indiv, pm)
    % Mutacija nakljucno spremeni posamezen gen. Je edini operator, ki v
    % populacijo vnese vrednost, ki je ni imel noben stars, zato vzdrzuje
    % diverziteto in omogoca izhod iz lokalnega optimuma.

    ops = {'+', '-', '*', '/'};

    for i = 1:length(indiv)
        if rand < pm
            if mod(i,2)==0
                % Soda mesta so operatorji.
                indiv{i} = ops{randi(4)};
            else
                % Liha mesta so stevke.
                indiv{i} = num2str(randi(9));
            end
        end
    end

    mutated = indiv;
end
