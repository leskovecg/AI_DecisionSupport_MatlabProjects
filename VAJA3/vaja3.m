%% VAJA 3 - LSE, PCA in PCR pri identifikaciji modela
% Del 1: vpliv standardizacije podatkov na modela LSE in PCA.
% Del 2: problem kolinearnosti vhodov in primerjava LSE proti PCR.

clear; clc; close all;
rng(1);   % ponovljivost eksperimenta s kolinearnostjo

%% ------------------------------------------------------------------------
% 0. Nastavitve
% -------------------------------------------------------------------------

save_figures = true;      % shranjevanje grafov
save_tables  = true;      % shranjevanje tabel v .mat
num_runs     = 100;        % stevilo ponovitev za eksperimentalno varianco
corr_threshold = 0.95;     % prag za javljanje mozne kolinearnosti
cond_warn_threshold = 1e10; % prag pogojenostnega stevila za opozorilo

% Poti zasidramo na lokacijo skripte, zato se izhodi pisejo poleg nje,
% neodvisno od trenutne delovne mape.
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)      % zascita, ce se skripta poganja po odsekih
    script_dir = pwd;
end
fig_path = fullfile(script_dir, 'figs');
tab_path = fullfile(script_dir, 'tabs');

if save_figures && ~exist(fig_path, 'dir')
    mkdir(fig_path);
end
if save_tables && ~exist(tab_path, 'dir')
    mkdir(tab_path);
end

%% ------------------------------------------------------------------------
% 1. Nalaganje podatkov
% -------------------------------------------------------------------------

data = load(fullfile(script_dir, 'VAJA3.mat'));

A_FLOW = data.A_FLOW(:);
T_H2O  = data.T_H2O(:);
C_ACID = data.C_ACID(:);
I_EFF  = data.I_EFF(:);

X = [A_FLOW, T_H2O, C_ACID];
Y = I_EFF;

var_names_3 = {'A_FLOW','T_H2O','C_ACID'};
N = size(X, 1);

fprintf('============================================================\n');
fprintf('VAJA 3 - identifikacija modela z LSE, PCA in PCR\n');
fprintf('============================================================\n');
fprintf('Stevilo vzorcev: %d\n\n', N);

% Terminologija je pojasnjena enkrat, preden se uporabi v izpisih spodaj.
fprintf('BIAS = povprecna vrednost napake e = y - y_hat. Blizu 0 = nepristranski model.\n\n');

%% ------------------------------------------------------------------------
% 2. DEL 1 - vpliv standardizacije na LSE in PCA
% -------------------------------------------------------------------------
% Po navodilih primerjamo:
% a) skaliranje min-max na [0,1]
% b) standardizacijo z-score
% c) brez standardizacije
%
% Pri LSE uporabimo model s prostim clenom:
%   y = a1*x1 + a2*x2 + a3*x3 + r
%
% Pri PCA zgradimo implicitni model iz [X Y]:
%   p' * ([x y] - v) = 0
% in ga pretvorimo v eksplicitno obliko:
%   y = a1*x1 + a2*x2 + a3*x3 + r
%
% POMEMBNO:
% Pri standardiziranih primerih transformiramo OBA modela (LSE in PCA) nazaj
% v originalni prostor, ker mora primerjava potekati v originalnem prostoru.

% PCA (Del 1) in PCR (Del 2) uporabljata isto lastno dekompozicijo, a
% nasprotno logiko izbire lastnih vektorjev. Locitev je izpisana tukaj, da
% se ne zamenja s pojasnilom PCR v Delu 2 spodaj.
fprintf(['[PCA - Del 1] Implicitni model iz VSEH spremenljivk [X,Y]; normala = lastni ' ...
    'vektor NAJMANJSE lastne vrednosti (total least squares).\n\n']);

methods = {'minmax', 'zscore', 'none'};
method_labels = {'Min-Max [0,1]', 'Z-Score', 'None'};

% Hranilniki rezultatov
results_part1 = struct();

for k = 1:numel(methods)
    method = methods{k};

    % Standardizacija X in Y po izbrani metodi
    [X_std, x_shift, x_scale] = standardize_inputs(X, method);
    [Y_std, y_shift, y_scale] = standardize_output(Y, method);

    % -------------------------------------------------
    % LSE v standardiziranem prostoru
    % -------------------------------------------------
    Phi_std = [X_std, ones(N,1)];

    % Normalne enacbe resimo z operatorjem \, izpisemo pogojenostno stevilo
    % matrike Phi_std'*Phi_std in morebitno opozorilo MATLABa o skoraj
    % singularni matriki pretvorimo v razumljivo sporocilo.
    A_phi = Phi_std' * Phi_std;
    b_phi = Phi_std' * Y_std;
    theta_lse_std = solve_normal_equations_checked(A_phi, b_phi, ...
        sprintf('Phi_std^T*Phi_std [%s]', method_labels{k}), cond_warn_threshold);

    % Transformacija parametrov LSE nazaj v originalni prostor
    theta_lse_orig = transform_explicit_model_to_original( ...
        theta_lse_std(1:end-1), theta_lse_std(end), ...
        x_shift, x_scale, y_shift, y_scale);

    % Teoreticna kovarianca parametrov LSE v standardiziranem prostoru
    e_lse_std = Y_std - Phi_std * theta_lse_std;
    sigma2_std = (e_lse_std' * e_lse_std) / (N - size(Phi_std,2));
    % A\eye(size(A)) namesto inv(A): s tem se izognemo eksplicitnemu
    % racunanju inverza, ki je numericno manj stabilno od resevanja
    % ekvivalentnega linearnega sistema s faktorizacijo.
    Cov_lse_std = sigma2_std * (A_phi \ eye(size(A_phi)));

    % Transformacija kovariance v originalni prostor parametrov
    Cov_lse_orig = transform_lse_covariance_to_original( ...
        Cov_lse_std, x_shift, x_scale, y_shift, y_scale);
    var_lse_orig = diag(Cov_lse_orig);

    % -------------------------------------------------
    % PCA v standardiziranem prostoru
    % -------------------------------------------------
    % Prilagajamo hiperravnino v prostoru [X_std, Y_std].
    % Pravilni normalni vektor je lastni vektor, ki pripada NAJMANJSI
    % lastni vrednosti (smer najmanjse variance).
    [a_pca_std, r_pca_std, p_normal, eigvals] = pca_implicit_to_explicit(X_std, Y_std);

    % Transformacija modela PCA nazaj v originalni prostor
    theta_pca_orig = transform_explicit_model_to_original( ...
        a_pca_std, r_pca_std, x_shift, x_scale, y_shift, y_scale);

    % -------------------------------------------------
    % Ovrednotenje obeh modelov v ORIGINALNEM prostoru
    % -------------------------------------------------
    metrics_lse = evaluate_explicit_model(X, Y, theta_lse_orig);
    metrics_pca = evaluate_explicit_model(X, Y, theta_pca_orig);

    % Ortogonalni (pravokotni) RMSE v standardiziranem prostoru [X,Y] je
    % kriterij, ki ga PCA/TLS dejansko minimizira, za razliko od NRMSE, ki
    % meri le navpicno napako v y, kakrsno minimizira LSE.
    orth_rmse_lse = orthogonal_rmse(X_std, Y_std, theta_lse_std(1:end-1), theta_lse_std(end));
    orth_rmse_pca = orthogonal_rmse(X_std, Y_std, a_pca_std, r_pca_std);

    % Shranjevanje rezultatov
    results_part1(k).method = method_labels{k};
    results_part1(k).theta_lse_orig = theta_lse_orig(:)';
    results_part1(k).theta_pca_orig = theta_pca_orig(:)';
    results_part1(k).var_lse_orig = var_lse_orig(:)';
    results_part1(k).metrics_lse = metrics_lse;
    results_part1(k).metrics_pca = metrics_pca;
    results_part1(k).pca_normal_vector = p_normal(:)';
    results_part1(k).pca_eigenvalues = eigvals(:)';
    results_part1(k).orth_rmse_lse = orth_rmse_lse;
    results_part1(k).orth_rmse_pca = orth_rmse_pca;

    % Izpis enacbe modela v berljivi obliki, originalni prostor
    print_model_equation(theta_lse_orig, var_names_3, sprintf('LSE, %s', method_labels{k}));
    print_model_equation(theta_pca_orig, var_names_3, sprintf('PCA, %s', method_labels{k}));
end
fprintf('\n');

%% ------------------------------------------------------------------------
% 3. Prikaz rezultatov DELA 1
% -------------------------------------------------------------------------

fprintf('==================== DEL 1: STANDARDIZACIJA ====================\n\n');

% Tabele parametrov v originalnem prostoru
LSE_param_table = zeros(numel(methods), 4);
PCA_param_table = zeros(numel(methods), 4);
LSE_var_table   = zeros(numel(methods), 4);

LSE_bias = zeros(numel(methods),1);
LSE_stdE = zeros(numel(methods),1);
LSE_nrmse = zeros(numel(methods),1);
LSE_orthRMSE = zeros(numel(methods),1);

PCA_bias = zeros(numel(methods),1);
PCA_stdE = zeros(numel(methods),1);
PCA_nrmse = zeros(numel(methods),1);
PCA_orthRMSE = zeros(numel(methods),1);

for k = 1:numel(methods)
    LSE_param_table(k,:) = results_part1(k).theta_lse_orig;
    PCA_param_table(k,:) = results_part1(k).theta_pca_orig;
    LSE_var_table(k,:)   = results_part1(k).var_lse_orig;

    LSE_bias(k)  = results_part1(k).metrics_lse.mean_error;   % povprecna napaka je bias
    LSE_stdE(k)  = results_part1(k).metrics_lse.std_error;
    LSE_nrmse(k) = results_part1(k).metrics_lse.nrmse;
    LSE_orthRMSE(k) = results_part1(k).orth_rmse_lse;

    PCA_bias(k)  = results_part1(k).metrics_pca.mean_error;   % povprecna napaka je bias
    PCA_stdE(k)  = results_part1(k).metrics_pca.std_error;
    PCA_nrmse(k) = results_part1(k).metrics_pca.nrmse;
    PCA_orthRMSE(k) = results_part1(k).orth_rmse_pca;
end

T_LSE_params = array2table(LSE_param_table, ...
    'VariableNames', {'a1','a2','a3','r'}, ...
    'RowNames', method_labels);

T_PCA_params = array2table(PCA_param_table, ...
    'VariableNames', {'a1','a2','a3','r'}, ...
    'RowNames', method_labels);

T_LSE_var = array2table(LSE_var_table, ...
    'VariableNames', {'var_a1','var_a2','var_a3','var_r'}, ...
    'RowNames', method_labels);

% Stolpci so poimenovani *_bias, ker gre za povprecno napako, in dopolnjeni
% s stolpcema ortogonalnega RMSE za oba modela.
T_metrics = table(LSE_bias, LSE_stdE, LSE_nrmse, LSE_orthRMSE, ...
                   PCA_bias, PCA_stdE, PCA_nrmse, PCA_orthRMSE, ...
    'RowNames', method_labels, ...
    'VariableNames', {'LSE_bias','LSE_stdErr','LSE_NRMSE','LSE_orthRMSE', ...
                      'PCA_bias','PCA_stdErr','PCA_NRMSE','PCA_orthRMSE'});

disp('Parametri LSE v ORIGINALNEM prostoru:');
disp(T_LSE_params);
disp('Parametri PCA v ORIGINALNEM prostoru:');
disp(T_PCA_params);
disp('Teoreticne variance parametrov LSE v ORIGINALNEM prostoru:');
disp(T_LSE_var);
disp('Metrike napake v ORIGINALNEM prostoru (BIAS = povprecna napaka, glej razlago zgoraj):');
disp(T_metrics);

% Pojasnilo, zakaj PCA po NRMSE izpade slabse, ceprav je po svojem lastnem
% kriteriju optimalen.
fprintf(['[POJASNILO] PCA/TLS minimizira PRAVOKOTNO (ortogonalno) napako v skupnem\n' ...
    'prostoru [X,Y], LSE pa minimizira NAVPICNO (vertikalno) napako v y. NRMSE meri\n' ...
    'ravno navpicno napako, zato je po TEM kriteriju LSE vedno enako dober ali boljsi\n' ...
    'od PCA - to ni napaka metode PCA, temvec posledica drugacnega kriterija optimizacije.\n' ...
    'Stolpca *_orthRMSE zgoraj kazeta obratno sliko: po ortogonalnem kriteriju je PCA\n' ...
    'enako dober ali boljsi od LSE, kar potrjuje, da gre za razlicna, ne nasprotujoca si\n' ...
    'kriterija.\n\n']);

% Samodejno priporocilo modela na podlagi |bias| in std(error)
for k = 1:numel(methods)
    recommend_model(LSE_bias(k), LSE_stdE(k), 'LSE', PCA_bias(k), PCA_stdE(k), 'PCA', ...
        sprintf('DEL 1 - %s', method_labels{k}));
end

% Kolicinska ocena robustnosti LSE na standardizacijo: cez tri metode se
% theta_lse_orig skoraj ne spremeni.
lse_range = max(LSE_param_table,[],1) - min(LSE_param_table,[],1);
fprintf(['\n[ROBUSTNOST LSE] max(abs(theta_a - theta_b)) med pari standardizacij, ' ...
    'po parametrih (a1,a2,a3,r): %.3e  %.3e  %.3e  %.3e\n'], lse_range);
fprintf('  => najvecja razlika kjerkoli: %.3e (pricakovano ~0 => LSE je robusten na standardizacijo)\n\n', ...
    max(lse_range));

% Enak izracun razpona za PCA, za neposredno primerjavo z robustnostjo LSE.
pca_range = max(PCA_param_table,[],1) - min(PCA_param_table,[],1);
fprintf(['[OBCUTLJIVOST PCA] max(abs(theta_a - theta_b)) med pari standardizacij, ' ...
    'po parametrih (a1,a2,a3,r): %.3e  %.3e  %.3e  %.3e\n'], pca_range);
fprintf('  => najvecja razlika kjerkoli: %.3e (PCA JE obcutljiv na skalo/standardizacijo podatkov)\n\n', ...
    max(pca_range));

%% ------------------------------------------------------------------------
% 4. Grafi DELA 1
% -------------------------------------------------------------------------

% Primerjava NRMSE
figure('Name','PART 1 - NRMSE comparison','Color','w');
bar([LSE_nrmse, PCA_nrmse], 'grouped');
grid on;
xlabel('Metoda standardizacije');
ylabel('NRMSE');
title('Primerjava NRMSE: LSE proti PCA');
set(gca, 'XTickLabel', method_labels);
legend({'LSE','PCA'}, 'Location', 'northwest');

% Primerjava biasa
figure('Name','PART 1 - Mean error comparison','Color','w');
bar([LSE_bias, PCA_bias], 'grouped');
grid on;
xlabel('Metoda standardizacije');
ylabel('Bias (povprečna napaka)');
title('Primerjava biasa: LSE proti PCA');
set(gca, 'XTickLabel', method_labels);
legend({'LSE','PCA'}, 'Location', 'northwest');

% Primerjava standardnega odklona napake
figure('Name','PART 1 - Error standard deviation comparison','Color','w');
bar([LSE_stdE, PCA_stdE], 'grouped');
grid on;
xlabel('Metoda standardizacije');
ylabel('Standardni odklon napake');
title('Primerjava konsistentnosti: LSE proti PCA');
set(gca, 'XTickLabel', method_labels);
legend({'LSE','PCA'}, 'Location', 'northwest');

% Primerjava parametrov
figure('Name','PART 1 - Parameter comparison','Color','w');
param_names = {'a1','a2','a3','r'};
for i = 1:4
    subplot(2,2,i);
    bar([LSE_param_table(:,i), PCA_param_table(:,i)], 'grouped');
    grid on;
    set(gca, 'XTickLabel', method_labels);
    xlabel('Metoda standardizacije');
    ylabel('Vrednost');
    title(sprintf('Parameter %s', param_names{i}));
    legend({'LSE','PCA'}, 'Location', 'best');
end

% Porazdelitev napake pri standardizaciji z-score. Povprecje in standardni
% odklon povesta le prva dva momenta, histogram pa pokaze obliko
% porazdelitve (simetrijo, centriranost, repe).
k_zscore = find(strcmp(methods, 'zscore'), 1);
e_lse_hist = results_part1(k_zscore).metrics_lse.error;
e_pca_hist = results_part1(k_zscore).metrics_pca.error;

figure('Name','PART 1 - Error distribution','Color','w');
subplot(1,2,1);
histogram(e_lse_hist, 20, 'FaceColor', [0.2 0.4 0.8]);
grid on; hold on;
xline(0, 'k--', 'LineWidth', 1.2);
xline(mean(e_lse_hist), 'r-', 'LineWidth', 1.2);
xlabel('Napaka e = y - y_{hat}');
ylabel('Število vzorcev');
title(sprintf('LSE, Z-Score (bias %.2e)', mean(e_lse_hist)));
legend({'histogram','e = 0','povprečje'}, 'Location', 'best');

subplot(1,2,2);
histogram(e_pca_hist, 20, 'FaceColor', [0.8 0.4 0.2]);
grid on; hold on;
xline(0, 'k--', 'LineWidth', 1.2);
xline(mean(e_pca_hist), 'r-', 'LineWidth', 1.2);
xlabel('Napaka e = y - y_{hat}');
ylabel('Število vzorcev');
title(sprintf('PCA, Z-Score (bias %.2e)', mean(e_pca_hist)));
legend({'histogram','e = 0','povprečje'}, 'Location', 'best');

fprintf('Porazdelitev napake (Z-Score): LSE bias = %.4e, std = %.4f; PCA bias = %.4e, std = %.4f\n\n', ...
    mean(e_lse_hist), std(e_lse_hist), mean(e_pca_hist), std(e_pca_hist));

%% ------------------------------------------------------------------------
% 5. DEL 2 - problem kolinearnosti: LSE proti PCR
% -------------------------------------------------------------------------
% Dodamo novo spremenljivko:
%   x_new = 2*T_H2O + 6 + 0.1*randn(...)
%
% Nato standardiziramo vhode in primerjamo:
% - LSE brez prostega clena
% - PCR
%
% Pri PCR najprej izvedemo PCA na standardizirani matriki vhodov X,
% odstranimo komponento z zanemarljivim vplivom (tisto z najmanjso lastno
% vrednostjo), izvedemo regresijo v prostoru glavnih komponent in parametre
% preslikamo nazaj v originalni standardizirani prostor vhodov.

fprintf('\n==================== DEL 2: KOLINEARNOST ====================\n\n');

% Pojasnilo PCR, eksplicitno postavljeno nasproti pojasnilu PCA iz Dela 1.
fprintf(['[PCR - Del 2] PCA samo na VHODIH (X); odstranimo NAJMANJSO komponento; LSE ' ...
    'regresija na preostalih (najvecjih) komponentah.\n\n']);

% Izhodiscni model na treh originalnih standardiziranih vhodih (brez prostega clena).
% Premik in skalo shranimo, da lahko model spodaj transformiramo nazaj v
% originalni prostor.
[X3_std, x3_shift, x3_scale] = standardize_inputs(X, 'zscore');
[Y_std, y_shift, y_scale]    = standardize_output(Y, 'zscore');

theta_baseline_3vars = (X3_std' * X3_std) \ (X3_std' * Y_std);

% Transformacija izhodiscnega modela brez prostega clena nazaj v originalni prostor.
theta_baseline_3vars_orig = transform_noIntercept_model_to_original( ...
    theta_baseline_3vars, x3_shift, x3_scale, y_shift, y_scale);

fprintf('Izhodiscni parameter LSE za T_H2O (3 standardizirani vhodi, brez prostega clena): %.6f\n\n', ...
    theta_baseline_3vars(2));

print_model_equation(theta_baseline_3vars, var_names_3, 'LSE izhodiscni, standardiziran, brez prostega clena');
print_model_equation(theta_baseline_3vars_orig, var_names_3, 'LSE izhodiscni, ORIGINALNI prostor');

% En reprezentativen zagon
x_new = 2*T_H2O + 6 + 0.1*randn(N,1);
X_col = [A_FLOW, T_H2O, C_ACID, x_new];
var_names_4 = {'A_FLOW','T_H2O','C_ACID','X_new'};

% Standardizacija vhodov in izhoda; premik in skalo shranimo za transformacijo nazaj.
[X_col_std, xcol_shift, xcol_scale] = standardize_inputs(X_col, 'zscore');
[Y_col_std, ycol_shift, ycol_scale] = standardize_output(Y, 'zscore');

% -----------------------------
% LSE brez prostega clena
% -----------------------------
% Sistem resimo z operatorjem \, izpisemo pogojenostno stevilo matrike
% X_col_std'*X_col_std (pricakovano veliko, ker so vhodi skoraj kolinearni)
% in morebitno opozorilo MATLABa pretvorimo v razumljivo sporocilo.
A_col = X_col_std' * X_col_std;
b_col = X_col_std' * Y_col_std;
theta_lse_col = solve_normal_equations_checked(A_col, b_col, ...
    'X_col_std^T*X_col_std (kolinearno, reprezentativni zagon)', cond_warn_threshold);

e_lse_col = Y_col_std - X_col_std * theta_lse_col;
sigma2_lse_col = (e_lse_col' * e_lse_col) / (N - size(X_col_std,2));
% Tudi tu A\eye(size(A)) namesto inv(A) za kovariancno matriko.
Cov_lse_col = sigma2_lse_col * (A_col \ eye(size(A_col)));
var_lse_col_theoretical = diag(Cov_lse_col);

% -----------------------------
% PCR
% -----------------------------
[theta_pcr_col, Ps, eigvals_desc, T_scores] = pcr_regression_remove_smallest_pc(X_col_std, Y_col_std);

% Transformacija kolinearnih modelov brez prostega clena (LSE, PCR) nazaj
% v originalni prostor spremenljivk.
theta_lse_col_orig = transform_noIntercept_model_to_original( ...
    theta_lse_col, xcol_shift, xcol_scale, ycol_shift, ycol_scale);
theta_pcr_col_orig = transform_noIntercept_model_to_original( ...
    theta_pcr_col, xcol_shift, xcol_scale, ycol_shift, ycol_scale);

% Metrike v standardiziranem prostoru izhoda (samo za primerjavo)
metrics_lse_col = evaluate_model_no_intercept(X_col_std, Y_col_std, theta_lse_col);
metrics_pcr_col = evaluate_model_no_intercept(X_col_std, Y_col_std, theta_pcr_col);

% Korelacijska matrika kolinearnih vhodov
R_col = corrcoef(X_col_std);

% Tabela prispevkov za prvi dve ohranjeni glavni komponenti
num_pc_to_show = min(2, size(Ps,2));
PC_contrib = abs(Ps(:,1:num_pc_to_show)) ./ sum(abs(Ps(:,1:num_pc_to_show)), 1);
T_PC_contrib = array2table(PC_contrib * 100, ...
    'VariableNames', arrayfun(@(i) sprintf('P_%d', i), 1:num_pc_to_show, 'UniformOutput', false), ...
    'RowNames', var_names_4);

disp('Parametri LSE ob kolinearnosti (standardizirani vhodi, brez prostega clena):');
disp(array2table(theta_lse_col(:)', 'VariableNames', {'a1','a2','a3','a4'}));

disp('Parametri PCR ob kolinearnosti (preslikani nazaj v standardizirani prostor vhodov):');
disp(array2table(theta_pcr_col(:)', 'VariableNames', {'a1','a2','a3','a4'}));

disp('Teoreticne variance parametrov LSE (kolinearni primer):');
disp(array2table(var_lse_col_theoretical(:)', ...
    'VariableNames', {'var_a1','var_a2','var_a3','var_a4'}));

disp('Korelacijska matrika standardiziranih kolinearnih vhodov:');
disp(array2table(R_col, 'VariableNames', var_names_4, 'RowNames', var_names_4));

disp('Odstotni prispevki spremenljivk k izbranim glavnim komponentam:');
disp(T_PC_contrib);

% Izpis enacb modelov za kolinearni primer
print_model_equation(theta_lse_col, var_names_4, 'LSE, standardiziran, brez prostega clena (kolinearno)');
print_model_equation(theta_pcr_col, var_names_4, 'PCR, standardiziran, brez prostega clena (kolinearno)');

% Isti modeli, transformirani v ORIGINALNI prostor
disp('Parametra LSE in PCR (kolinearni primer), transformirana v ORIGINALNI prostor:');
T_col_params_orig = array2table([theta_lse_col_orig'; theta_pcr_col_orig'], ...
    'VariableNames', {'a1','a2','a3','a4','r'}, ...
    'RowNames', {'LSE','PCR'});
disp(T_col_params_orig);
print_model_equation(theta_lse_col_orig, var_names_4, 'LSE, ORIGINALNI prostor (kolinearno)');
print_model_equation(theta_pcr_col_orig, var_names_4, 'PCR, ORIGINALNI prostor (kolinearno)');

fprintf('Standardizirane metrike LSE: BIAS = %.6f, std napake = %.6f, NRMSE = %.6f\n', ...
    metrics_lse_col.mean_error, metrics_lse_col.std_error, metrics_lse_col.nrmse);
fprintf('Standardizirane metrike PCR: BIAS = %.6f, std napake = %.6f, NRMSE = %.6f\n\n', ...
    metrics_pcr_col.mean_error, metrics_pcr_col.std_error, metrics_pcr_col.nrmse);

% Urejena tabela bias/std/NRMSE za Del 2
T_metrics_col = table([metrics_lse_col.mean_error; metrics_pcr_col.mean_error], ...
                      [metrics_lse_col.std_error; metrics_pcr_col.std_error], ...
                      [metrics_lse_col.nrmse; metrics_pcr_col.nrmse], ...
    'RowNames', {'LSE','PCR'}, ...
    'VariableNames', {'BIAS','stdErr','NRMSE'});
disp('Metrike (BIAS/std/NRMSE) - kolinearni primer, standardiziran prostor:');
disp(T_metrics_col);

% Samodejno priporocilo modela tudi za kolinearni primer
recommend_model(metrics_lse_col.mean_error, metrics_lse_col.std_error, 'LSE', ...
                 metrics_pcr_col.mean_error, metrics_pcr_col.std_error, 'PCR', ...
                 'DEL 2 - kolinearnost');

fprintf('Parameter za T_H2O v kolinearnem modelu LSE: %.6f\n', theta_lse_col(2));
fprintf('Parameter za X_new  v kolinearnem modelu LSE: %.6f\n', theta_lse_col(4));
fprintf('Njuna vsota (LSE): %.6f\n\n', theta_lse_col(2) + theta_lse_col(4));

fprintf('Parameter za T_H2O v kolinearnem modelu PCR: %.6f\n', theta_pcr_col(2));
fprintf('Parameter za X_new  v kolinearnem modelu PCR: %.6f\n', theta_pcr_col(4));
fprintf('Njuna vsota (PCR): %.6f\n\n', theta_pcr_col(2) + theta_pcr_col(4));

%% ------------------------------------------------------------------------
% 6. Eksperimentalna varianca parametrov iz ponovljenih zagonov
% -------------------------------------------------------------------------
% Celoten eksperiment s kolinearnostjo veckrat ponovimo, ker x_new vsebuje
% nakljucni sum. Iz ponovitev ocenimo varianco parametrov eksperimentalno.

theta_lse_all = zeros(4, num_runs);
theta_pcr_all = zeros(4, num_runs);

for run = 1:num_runs
    x_new_run = 2*T_H2O + 6 + 0.1*randn(N,1);
    X_col_run = [A_FLOW, T_H2O, C_ACID, x_new_run];

    [X_col_run_std, ~, ~] = standardize_inputs(X_col_run, 'zscore');
    [Y_run_std, ~, ~] = standardize_output(Y, 'zscore');

    % LSE brez prostega clena
    theta_lse_run = (X_col_run_std' * X_col_run_std) \ (X_col_run_std' * Y_run_std);
    theta_lse_all(:, run) = theta_lse_run;

    % PCR
    [theta_pcr_run, ~, ~, ~] = pcr_regression_remove_smallest_pc(X_col_run_std, Y_run_std);
    theta_pcr_all(:, run) = theta_pcr_run;
end

var_lse_exp = var(theta_lse_all, 0, 2);
var_pcr_exp = var(theta_pcr_all, 0, 2);

T_variance_exp = table(var_lse_exp, var_pcr_exp, ...
    'RowNames', {'a1','a2','a3','a4'}, ...
    'VariableNames', {'LSE_expVar','PCR_expVar'});

disp('Eksperimentalne variance parametrov iz ponovljenih zagonov:');
disp(T_variance_exp);

% Napaka modela sama po sebi ne razkrije kolinearnosti: NRMSE(LSE) je
% skoraj enak NRMSE(PCR), varianca parametrov pa se razlikuje za rede
% velikosti.
fprintf(['\n[POJASNILO] NRMSE(LSE) = %.6f je priblizno enak NRMSE(PCR) = %.6f, ' ...
    'medtem ko je povprecna eksperimentalna varianca parametrov LSE (%.3e) za rede\n' ...
    'velikosti vecja od PCR (%.3e). To pokaze, da napaka modela sama po sebi ne\n' ...
    'razkrije problema kolinearnosti - potrebno je pogledati varianco/stabilnost parametrov.\n\n'], ...
    metrics_lse_col.nrmse, metrics_pcr_col.nrmse, mean(var_lse_exp), mean(var_pcr_exp));

%% ------------------------------------------------------------------------
% 7. Grafi DELA 2
% -------------------------------------------------------------------------

% Primerjava parametrov za en reprezentativen zagon
figure('Name','PART 2 - Parameter comparison','Color','w');
bar([theta_lse_col(:), theta_pcr_col(:)], 'grouped');
grid on;
xlabel('Parameter');
ylabel('Vrednost');
title('Parametri LSE proti PCR v kolinearnem primeru');
set(gca, 'XTickLabel', {'a1','a2','a3','a4'});
legend({'LSE','PCR'}, 'Location', 'best');

% Primerjava varianc
figure('Name','PART 2 - Experimental variance comparison','Color','w');
bar([var_lse_exp(:), var_pcr_exp(:)], 'grouped');
grid on;
xlabel('Parameter');
ylabel('Eksperimentalna varianca');
title('Eksperimentalna varianca parametrov: LSE proti PCR');
set(gca, 'XTickLabel', {'a1','a2','a3','a4'});
legend({'LSE','PCR'}, 'Location', 'best');

% Prispevki spremenljivk k prvima dvema ohranjenima glavnima komponentama
figure('Name','PART 2 - PC contributions','Color','w');
bar(PC_contrib * 100, 'grouped');
grid on;
xlabel('Spremenljivka');
ylabel('Prispevek [%]');
title('Prispevki spremenljivk k izbranim glavnim komponentam');
set(gca, 'XTickLabel', var_names_4);
legend(arrayfun(@(i) sprintf('P_%d', i), 1:num_pc_to_show, 'UniformOutput', false), ...
    'Location', 'best');

% Korelacijska matrika kot slika
figure('Name','PART 2 - Correlation matrix','Color','w');
imagesc(R_col);
colorbar;
axis equal tight;
title('Korelacijska matrika standardiziranih kolinearnih vhodov');
set(gca, 'XTick', 1:4, 'XTickLabel', var_names_4, ...
         'YTick', 1:4, 'YTickLabel', var_names_4);

% Dvojni stolpicni graf: NRMSE(LSE) ~ NRMSE(PCR) levo, varianca parametrov
% LSE >> PCR desno (logaritemska skala).
figure('Name','PART 2 - Error vs Variance','Color','w');
subplot(1,2,1);
bar([metrics_lse_col.nrmse, metrics_pcr_col.nrmse]);
grid on;
set(gca, 'XTickLabel', {'LSE','PCR'});
ylabel('NRMSE');
title('NRMSE: LSE \approx PCR');

subplot(1,2,2);
bar([mean(var_lse_exp), mean(var_pcr_exp)]);
grid on;
set(gca, 'XTickLabel', {'LSE','PCR'}, 'YScale', 'log');
ylabel('Povprečna eksperimentalna varianca parametrov (log)');
title('Varianca: LSE >> PCR');

%% ------------------------------------------------------------------------
% 8. Javljanje mozne kolinearnosti iz korelacijske matrike
% -------------------------------------------------------------------------

fprintf('Potencialno kolinearni pari spremenljivk (|corr| > %.2f):\n', corr_threshold);
[row_idx, col_idx] = find(abs(R_col) > corr_threshold & abs(R_col) < 1);

already_reported = false(size(row_idx));
for i = 1:length(row_idx)
    if row_idx(i) < col_idx(i)
        fprintf('  %s in %s : corr = %.4f\n', ...
            var_names_4{row_idx(i)}, var_names_4{col_idx(i)}, R_col(row_idx(i), col_idx(i)));
    end
end

fprintf('\nKoncano.\n');

%% ------------------------------------------------------------------------
% 9. Shranjevanje
% -------------------------------------------------------------------------

if save_tables
    save(fullfile(tab_path, 'part1_results.mat'), 'results_part1', ...
         'T_LSE_params', 'T_PCA_params', 'T_LSE_var', 'T_metrics', ...
         'lse_range', 'pca_range');

    save(fullfile(tab_path, 'part2_results.mat'), 'theta_baseline_3vars', ...
         'theta_lse_col', 'theta_pcr_col', 'var_lse_col_theoretical', ...
         'theta_lse_all', 'theta_pcr_all', 'var_lse_exp', 'var_pcr_exp', ...
         'T_variance_exp', 'R_col', 'T_PC_contrib', ...
         'theta_baseline_3vars_orig', 'theta_lse_col_orig', 'theta_pcr_col_orig', ...
         'T_col_params_orig', 'T_metrics_col');
end

if save_figures
    figs = findall(0, 'Type', 'figure');
    for i = 1:length(figs)
        if isgraphics(figs(i), 'figure')
            fig_name = get(figs(i), 'Name');
            if isempty(fig_name)
                fig_name = sprintf('figure_%02d', i);
            end
            saveas(figs(i), fullfile(fig_path, [sanitize_filename(fig_name), '.png']));
        end
    end
end

%% ========================================================================
% Lokalne pomozne funkcije
% ========================================================================

function [X_std, shift, scale] = standardize_inputs(X, method)
% Standardizira matriko vhodov X po izbrani metodi.
% Vrne:
%   X_std  - standardizirana matrika vhodov
%   shift  - vektor premika
%   scale  - vektor skale

    switch lower(method)
        case 'minmax'
            shift = min(X, [], 1);
            scale = max(X, [], 1) - min(X, [], 1);
            X_std = (X - shift) ./ scale;

        case 'zscore'
            shift = mean(X, 1);
            scale = std(X, 0, 1);
            X_std = (X - shift) ./ scale;

        case 'none'
            shift = zeros(1, size(X,2));
            scale = ones(1, size(X,2));
            X_std = X;

        otherwise
            error('Neznana metoda standardizacije.');
    end
end

function [Y_std, shift, scale] = standardize_output(Y, method)
% Standardizira izhodni vektor Y po izbrani metodi.

    switch lower(method)
        case 'minmax'
            shift = min(Y);
            scale = max(Y) - min(Y);
            Y_std = (Y - shift) / scale;

        case 'zscore'
            shift = mean(Y);
            scale = std(Y);
            Y_std = (Y - shift) / scale;

        case 'none'
            shift = 0;
            scale = 1;
            Y_std = Y;

        otherwise
            error('Neznana metoda standardizacije.');
    end
end

function theta_orig = transform_explicit_model_to_original(a_std, r_std, x_shift, x_scale, y_shift, y_scale)
% Transformira model:
%   y_std = a_std' * x_std + r_std
% nazaj v originalni prostor:
%   y = a_orig' * x + r_orig
%
% kjer velja:
%   x_std = (x - x_shift) ./ x_scale
%   y_std = (y - y_shift) / y_scale

    a_std = a_std(:);
    x_shift = x_shift(:);
    x_scale = x_scale(:);

    a_orig = y_scale * (a_std ./ x_scale);
    r_orig = y_shift + y_scale * r_std - sum(a_orig .* x_shift);

    theta_orig = [a_orig; r_orig];
end

function theta_orig = transform_noIntercept_model_to_original(a_std, x_shift, x_scale, y_shift, y_scale)
% Transformira model, prilagojen BREZ prostega clena v standardiziranem
% prostoru,
%   y_std = a_std' * x_std      (brez prostega clena v standardiziranem prostoru)
% nazaj v ORIGINALNI prostor spremenljivk.
%
% Z vstavitvijo x_std = (x - x_shift)./x_scale in y_std = (y - y_shift)/y_scale:
%   (y - y_shift)/y_scale = sum_i a_std_i * (x_i - x_shift_i)/x_scale_i
%   y = sum_i [ y_scale*a_std_i/x_scale_i ] * x_i  +  [ y_shift - sum_i(a_orig_i*x_shift_i) ]
%     = a_orig' * x + r_orig
%
% OPOMBA: ceprav model v standardiziranem prostoru NIMA prostega clena, ima
% transformirani model v ORIGINALNEM prostoru praviloma NENICELN prosti
% clen r_orig. Vzrok je, da sta z-score in min-max afini transformaciji z
% nenicelnim premikom (povprecje oziroma minimum): "brez prostega clena" je
% lastnost izbranega (premaknjenega in skaliranega) koordinatnega sistema,
% ne pa lastnost same zveze med x in y.

    a_std = a_std(:);
    x_shift = x_shift(:);
    x_scale = x_scale(:);

    a_orig = y_scale * (a_std ./ x_scale);
    r_orig = y_shift - sum(a_orig .* x_shift);

    theta_orig = [a_orig; r_orig];

    if abs(r_orig) > 1e-8
        fprintf(['[OPAZANJE] Model brez prostega clena v standardiziranem prostoru dobi po ' ...
            'transformaciji nazaj v ORIGINALNI prostor NENICELN prosti clen (r_orig = %.6f).\n' ...
            'To je posledica afine standardizacije z necelnim povprecjem/minimumom - "brez ' ...
            'prostega clena" velja le v standardiziranem koordinatnem sistemu.\n\n'], r_orig);
    end
end

function Cov_orig = transform_lse_covariance_to_original(Cov_std, x_shift, x_scale, y_shift, y_scale)
% Transformira kovarianco [a_std; r_std] v kovarianco [a_orig; r_orig].
%
% Afina transformacija je:
%   [a_orig; r_orig] = M * [a_std; r_std] + c
% zato velja:
%   Cov_orig = M * Cov_std * M'

    n = length(x_scale);
    D = diag(y_scale ./ x_scale(:));
    M = [D, zeros(n,1);
        -(x_shift(:)' * D), y_scale];

    Cov_orig = M * Cov_std * M';
end

function [a, r, p, eigvals_sorted] = pca_implicit_to_explicit(X_std, Y_std)
% Prilagodi implicitni model PCA v prostoru [X Y] in ga pretvori v
% eksplicitno obliko.
%
% Implicitni model je:
%   p' * ([x y] - v) = 0
%
% Pravilni normalni vektor p je lastni vektor, ki pripada NAJMANJSI lastni
% vrednosti kovariancne matrike.
%
% To je uporaba PCA iz Dela 1: uporabi VSE spremenljivke [X,Y] skupaj in
% OHRANI lastni vektor najmanjse lastne vrednosti (smer najmanjse variance,
% total least squares). Nasprotno pocne funkcija
% pcr_regression_remove_smallest_pc spodaj (Del 2), ki izvede PCA samo na
% VHODIH in komponento z najmanjso lastno vrednostjo ZAVRZE, nato pa na
% preostalih (najvecjih) komponentah izvede navadno regresijo LSE.

    Z = [X_std, Y_std];
    v = mean(Z, 1);
    Zc = Z - v;
    F = (Zc' * Zc) / (size(Z,1) - 1);

    [V, D] = eig(F);
    eigvals = diag(D);
    [eigvals_sorted, idx] = sort(eigvals, 'ascend');  % najmanjsa najprej
    p = V(:, idx(1));                                 % normalni vektor

    % Zaradi konsistentnosti izberemo predznak s pozitivnim koeficientom pri y
    if p(end) < 0
        p = -p;
    end

    % Pretvorba p' * (z - v) = 0 v eksplicitno obliko y = a'*x + r
    a = -p(1:end-1) / p(end);
    r = (p' * v') / p(end);
end

function rmse_orth = orthogonal_rmse(X_std, Y_std, a, r)
% Ortogonalni (pravokotni) RMSE do hiperravnine
%   a'*x - y + r = 0
% v standardiziranem prostoru [X,Y]. To je kriterij, ki ga PCA/TLS
% dejansko minimizira, za razliko od NRMSE, ki meri le navpicno napako v y,
% kakrsno minimizira LSE. Velja za poljuben eksplicitni model (a,r), ne
% glede na to, ali izvira iz LSE ali iz PCA.

    n = [a(:); -1];
    Z = [X_std, Y_std];
    d = (Z * n + r) / norm(n);
    rmse_orth = sqrt(mean(d.^2));
end

function metrics = evaluate_explicit_model(X, Y, theta)
% Ovrednoti eksplicitni model:
%   y_hat = X*a + r

    a = theta(1:end-1);
    r = theta(end);
    Y_hat = X * a + r;
    e = Y - Y_hat;

    metrics.yhat = Y_hat;
    metrics.error = e;
    metrics.mean_error = mean(e);
    metrics.bias = metrics.mean_error;  % bias je povprecna napaka
    metrics.std_error = std(e);
    metrics.nrmse = sqrt(mean(e.^2)) / std(Y);
end

function metrics = evaluate_model_no_intercept(X, Y, theta)
% Ovrednoti model brez prostega clena:
%   y_hat = X*theta

    Y_hat = X * theta;
    e = Y - Y_hat;

    metrics.yhat = Y_hat;
    metrics.error = e;
    metrics.mean_error = mean(e);
    metrics.bias = metrics.mean_error;  % bias je povprecna napaka
    metrics.std_error = std(e);
    metrics.nrmse = sqrt(mean(e.^2)) / std(Y);
end

function [theta_pcr, Ps, eigvals_desc, T] = pcr_regression_remove_smallest_pc(X_std, Y_std)
% PCR:
% 1) PCA na standardizirani matriki vhodov X_std
% 2) odstranitev glavne komponente z najmanjso lastno vrednostjo
% 3) regresija v prostoru glavnih komponent
% 4) preslikava parametrov nazaj v originalni standardizirani prostor vhodov
%
% To je uporaba PCA iz Dela 2 (PCR): PCA tece samo na VHODIH (brez Y),
% komponenta z najmanjso lastno vrednostjo se ZAVRZE (nasprotna logika od
% funkcije pca_implicit_to_explicit zgoraj, ki lastni vektor najmanjse
% lastne vrednosti OHRANI kot normalo modela), nato pa se na preostalih
% (najvecjih) komponentah izvede navadna regresija LSE.

    C = cov(X_std);
    [V, D] = eig(C);
    eigvals = diag(D);
    [eigvals_desc, idx] = sort(eigvals, 'descend');
    V = V(:, idx);

    % Odstranimo najmanj pomembno komponento (najmanjsa lastna vrednost)
    Ps = V(:, 1:end-1);

    % Rezultati projekcije v ohranjenem prostoru glavnih komponent
    T = X_std * Ps;

    % Regresija v prostoru glavnih komponent
    theta_pc = (T' * T) \ (T' * Y_std);

    % Preslikava nazaj v originalni standardizirani prostor vhodov
    theta_pcr = Ps * theta_pc;
end

function theta = solve_normal_equations_checked(A, b, label, cond_warn_threshold)
% Resi normalne enacbe A*theta = b z operatorjem \. Ta je boljsi od
% theta = inv(A)*b, ker uporabi faktorizacijo, prilagojeno strukturi
% matrike A (npr. Choleskyjevo za simetricno pozitivno definitno A), in
% nikoli ne tvori eksplicitnega, potencialno slabo pogojenega inverza -
% kar je hitreje in numericno stabilneje, se posebej ko je A blizu
% singularne.
%
% Dodatno izpisemo pogojenostno stevilo matrike A in morebitno opozorilo
% MATLABa o skoraj singularni matriki oziroma rangovni pomanjkljivosti
% pretvorimo v eksplicitno, berljivo sporocilo.

    cond_val = cond(A);
    fprintf('Pogojenostno stevilo (%s): %.2e\n', label, cond_val);
    if cond_val > cond_warn_threshold
        fprintf('[OPOZORILO] Matrika "%s" je zelo slabo pogojena (cond = %.2e) - ocenjeni parametri so lahko numericno nezanesljivi.\n', ...
            label, cond_val);
    end

    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:rankDeficientMatrix');
    lastwarn('');

    theta = A \ b;

    [warnMsg, ~] = lastwarn();
    warning('on', 'MATLAB:nearlySingularMatrix');
    warning('on', 'MATLAB:singularMatrix');
    warning('on', 'MATLAB:rankDeficientMatrix');

    if ~isempty(warnMsg)
        fprintf('[OPOZORILO] MATLAB je pri resevanju sistema "%s" javil: %s\n', label, warnMsg);
    end
end

function print_model_equation(theta, var_names, model_name)
% Sestavi in izpise enacbo modela v berljivi obliki iz vektorja parametrov,
% na primer:
%   y = 0.7156*A_FLOW + 1.2953*T_H2O - 0.1521*C_ACID - 39.9200   [LSE, Min-Max]
%
% theta ima lahko numel(var_names) elementov (brez prostega clena) ali
% numel(var_names)+1 elementov (zadnji element je prosti clen r).

    theta = theta(:);
    n = numel(var_names);

    if numel(theta) == n + 1
        a = theta(1:end-1);
        r = theta(end);
        has_intercept = true;
    elseif numel(theta) == n
        a = theta;
        r = 0;
        has_intercept = false;
    else
        error('print_model_equation: dolzina theta se ne ujema z var_names.');
    end

    eq_str = 'y =';
    for i = 1:numel(a)
        if i == 1 && a(i) >= 0
            eq_str = sprintf('%s %.4f*%s', eq_str, a(i), var_names{i});
        elseif a(i) >= 0
            eq_str = sprintf('%s + %.4f*%s', eq_str, a(i), var_names{i});
        else
            eq_str = sprintf('%s - %.4f*%s', eq_str, abs(a(i)), var_names{i});
        end
    end

    if has_intercept
        if r >= 0
            eq_str = sprintf('%s + %.4f', eq_str, r);
        else
            eq_str = sprintf('%s - %.4f', eq_str, abs(r));
        end
    end

    fprintf('%s   [%s]\n', eq_str, model_name);
end

function recommend_model(bias_a, std_a, name_a, bias_b, std_b, name_b, context_label)
% Primerja dva modela po |bias| in std(error) ter izpise priporocilo. Ce je
% en model strogo boljsi po OBEH metrikah, zmaga neposredno; sicer je
% primarni kriterij |bias| (std(error) je razsodnik ob izenacenju), ker je
% za namene identifikacije praviloma zazelen nepristranski model.

    if abs(bias_a) <= abs(bias_b) && std_a <= std_b
        winner = name_a;
    elseif abs(bias_b) <= abs(bias_a) && std_b <= std_a
        winner = name_b;
    elseif abs(bias_a) ~= abs(bias_b)
        if abs(bias_a) < abs(bias_b)
            winner = name_a;
        else
            winner = name_b;
        end
    else
        if std_a < std_b
            winner = name_a;
        else
            winner = name_b;
        end
    end

    fprintf(['[%s] Priporocen model: %s (manjsi |bias| in/ali std(error)). ' ...
        '|bias|: %s=%.6f, %s=%.6f ; std(error): %s=%.6f, %s=%.6f\n'], ...
        context_label, winner, name_a, abs(bias_a), name_b, abs(bias_b), ...
        name_a, std_a, name_b, std_b);
end

function name_out = sanitize_filename(name_in)
% Zamenja znake, ki so lahko problematicni v imenih datotek.

    name_out = regexprep(name_in, '[^\w\d-_]', '_');
end
