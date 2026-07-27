clear all
close all
clc

% Ponovljivost
rng(42);

%% Ustvarjanje map, ce se ne obstajata
% Poti zasidramo na lokacijo skripte, zato se izhodi pisejo poleg nje,
% neodvisno od trenutne delovne mape.
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)      % zascita, ce se skripta poganja po odsekih
    script_dir = pwd;
end
fig_path = fullfile(script_dir, 'figs');
tab_path = fullfile(script_dir, 'tabs');
if ~exist(fig_path, 'dir'), mkdir(fig_path); end
if ~exist(tab_path, 'dir'), mkdir(tab_path); end

%% Koordinate baznih postaj
x1 = 1; y1 = 1;
x2 = 10; y2 = 5;
x3 = 2; y3 = 4;
base_stations = [x1 y1; x2 y2; x3 y3];

%% Skupni parametri optimizacije
lambda0    = 0.001;
max_iters  = 1000;
tol        = 1e-6;
lambda_max = 1e10;   % varovalka pred neskoncno zanko zavrnitev
xy0        = [0, 0]; % zacetna ocena

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DEL 1: GLAVNA LOKALIZACIJA (N = 100, neutezeni proti utezenemu LM)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('=====================================================\n');
fprintf('DEL 1: GLAVNA LOKALIZACIJA\n');
fprintf('=====================================================\n');

% Glavno stevilo meritev
N_main = 100;

% Surove meritve hranimo v vektorjih (ne le povprecja), da lahko ocenimo
% tudi varianco meritev
meritve1 = arrayfun(@(~) ping_stolp_1(), 1:N_main);
meritve2 = arrayfun(@(~) ping_stolp_2(), 1:N_main);
meritve3 = arrayfun(@(~) ping_stolp_3(), 1:N_main);

% Povprecje in varianca izmerjenih razdalj po postajah
d1 = mean(meritve1);  var1 = var(meritve1);
d2 = mean(meritve2);  var2 = var(meritve2);
d3 = mean(meritve3);  var3 = var(meritve3);

d_main    = [d1; d2; d3];
var_main  = [var1; var2; var3];
std_main  = sqrt(var_main);

fprintf('Statistika meritev po baznih postajah (N = %d):\n', N_main);
fprintf('  Postaja |  Povprec. |  Varianca |      Std\n');
for i = 1:3
    fprintf('  %7d | %9.4f | %9.4f | %8.4f\n', i, d_main(i), var_main(i), std_main(i));
end
fprintf('\n');

% Izvoz statistike postaj v tabelo LaTeX
tex_file_var = fullfile(tab_path, 'station_variance.tex');
fid = fopen(tex_file_var, 'w');
fprintf(fid, '\\begin{tabular}{|c|c|c|c|}\\hline\n');
fprintf(fid, 'Bazna postaja & Povpre\\v{c}je $\\bar{d}_i$ & Varianca $\\sigma_i^2$ & Std $\\sigma_i$ \\\\\\hline\n');
for i = 1:3
    fprintf(fid, '%d & %.4f & %.4f & %.4f \\\\\n', i, d_main(i), var_main(i), std_main(i));
end
fprintf(fid, '\\hline\\end{tabular}\n');
fclose(fid);
fprintf('Tabela LaTeX shranjena v: %s\n\n', tex_file_var);

% (a) Neutezeni LM: W = identiteta (vsem postajam zaupamo enako)
W_unw = eye(3);
[xy_unw, iters_unw, hist_unw] = run_lm(base_stations, d_main, W_unw, ...
    xy0, lambda0, max_iters, tol, lambda_max);

% (b) Utezeni LM (WLS): W = diag(1/var_i) -> manj zanesljiva postaja
% (vecja varianca meritev) dobi manjso utez
W_wls = diag(1 ./ var_main);
[xy_wls, iters_wls, hist_wls] = run_lm(base_stations, d_main, W_wls, ...
    xy0, lambda0, max_iters, tol, lambda_max);

fprintf('Glavna lokalizacija z N = %d meritvami:\n', N_main);
fprintf('  Neutezeni LM: (x, y) = (%.4f, %.4f), iteracij = %d\n', ...
    xy_unw(1), xy_unw(2), iters_unw);
fprintf('  Utezeni LM:   (x, y) = (%.4f, %.4f), iteracij = %d\n\n', ...
    xy_wls(1), xy_wls(2), iters_wls);

final_estimate = xy_wls;   % utezena ocena je glavni rezultat
used_history   = hist_wls;

%% Graf konvergence (glavna lokalizacija, utezeni LM)
figure;
subplot(2,1,1);
plot(1:size(used_history,1), used_history(:,1), 'b', 'LineWidth', 1.5);
xlabel('Sprejeti korak', 'Color', 'k');
ylabel('x', 'Color', 'k');
title(sprintf('Konvergenca x (N = %d, utežen LM)', N_main), 'Color', 'k');
grid on;

subplot(2,1,2);
plot(1:size(used_history,1), used_history(:,2), 'r', 'LineWidth', 1.5);
xlabel('Sprejeti korak', 'Color', 'k');
ylabel('y', 'Color', 'k');
title(sprintf('Konvergenca y (N = %d, utežen LM)', N_main), 'Color', 'k');
grid on;

ax = findall(gcf, 'Type', 'axes');
for i = 1:length(ax)
    ax(i).XColor = 'k';
    ax(i).YColor = 'k';
    ax(i).GridColor = [0.7 0.7 0.7];
    ax(i).Color = 'w';
    ax(i).FontSize = 12;
end
set(gcf, 'Color', 'w');
saveas(gcf, fullfile(fig_path, 'main_convergence_plot.png'));

%% Graf geometrije (glavna lokalizacija)
figure; hold on; axis equal;
set(gcf, 'Color', 'w');
set(gca, 'Color', 'w');

plot(base_stations(:,1), base_stations(:,2), 'bo', ...
    'MarkerSize', 10, 'DisplayName', 'Bazne postaje');

% Kroznice narisemo rocno (brez dodatnih orodij)
theta = linspace(0, 2*pi, 200);
plot(x1 + d1*cos(theta), y1 + d1*sin(theta), 'k', 'HandleVisibility', 'off');
plot(x2 + d2*cos(theta), y2 + d2*sin(theta), 'k', 'HandleVisibility', 'off');
plot(x3 + d3*cos(theta), y3 + d3*sin(theta), 'k', 'HandleVisibility', 'off');

plot(used_history(:,1), used_history(:,2), 'g--', ...
    'LineWidth', 1.5, 'DisplayName', 'Pot optimizacije (utežena)');

plot(xy_unw(1), xy_unw(2), 'ms', ...
    'MarkerSize', 12, 'LineWidth', 1.5, 'DisplayName', 'Ocena (neutežena)');

plot(xy_wls(1), xy_wls(2), 'r*', ...
    'MarkerSize', 12, 'DisplayName', 'Ocena (utežena)');

xlabel('x', 'Color', 'k');
ylabel('y', 'Color', 'k');
title(sprintf('Geometrijski prikaz lokalizacije (N = %d)', N_main), 'Color', 'k');
grid on;

ax = gca;
ax.GridColor = [0.6 0.6 0.6];
ax.XColor = 'k';
ax.YColor = 'k';
ax.FontSize = 12;

lgd = legend('Location', 'northeast');
lgd.TextColor = 'k';
lgd.Box = 'on';
lgd.Color = 'w';

saveas(gcf, fullfile(fig_path, 'main_geometry_plot.png'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DEL 2: ANALIZA MERITEV ZA RAZLICNE N (varianca Monte Carlo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('=====================================================\n');
fprintf('DEL 2: ANALIZA MERITEV (Monte Carlo)\n');
fprintf('=====================================================\n');

N_values  = [1, 5, 10, 50, 100];
num_tests = length(N_values);
M         = 100;   % stevilo neodvisnih ponovitev na posamezen N

% OPOMBA: tu uporabimo neutezeno razlicico zaradi konsistentnosti cez vse N:
% pri N = 1 variance meritev ni mogoce oceniti (varianca enega vzorca je 0),
% zato utezi 1/var_i ne bi bile definirane.

results_mean_xy  = zeros(num_tests, 2);   % povprecje koncnih ocen cez M zagonov
results_iters    = zeros(num_tests, 1);   % povprecno stevilo iteracij
results_mc_var   = zeros(num_tests, 2);   % varianca Monte Carlo KONCNIH ocen
all_estimates    = zeros(M, 2, num_tests);% vse koncne ocene (za razsevni diagram)

for idx = 1:num_tests
    N = N_values(idx);

    est_N   = zeros(M, 2);
    iters_N = zeros(M, 1);

    for m = 1:M
        % Sveze neodvisne meritve za to ponovitev
        dN = [mean(arrayfun(@(~) ping_stolp_1(), 1:N));
              mean(arrayfun(@(~) ping_stolp_2(), 1:N));
              mean(arrayfun(@(~) ping_stolp_3(), 1:N))];

        [xy_m, k_m, ~] = run_lm(base_stations, dN, eye(3), ...
            xy0, lambda0, max_iters, tol, lambda_max);

        est_N(m, :) = xy_m;
        iters_N(m)  = k_m;
    end

    all_estimates(:, :, idx) = est_N;
    results_mean_xy(idx, :)  = mean(est_N, 1);
    results_iters(idx)       = mean(iters_N);
    % Varianca Monte Carlo koncnih ocen cez M neodvisnih zagonov
    % -> to je dejanska varianca ocene polozaja
    results_mc_var(idx, :)   = var(est_N, 0, 1);

    fprintf(['N = %3d --> povprecje (x, y) = (%.4f, %.4f), povpr. iteracij = %.1f, ' ...
             'MC varianca = (%.6f, %.6f)\n'], ...
        N, results_mean_xy(idx,1), results_mean_xy(idx,2), results_iters(idx), ...
        results_mc_var(idx,1), results_mc_var(idx,2));
end
fprintf('\n');

% Izvoz rezultatov v tabelo LaTeX
tex_file = fullfile(tab_path, 'measurement_results.tex');
fid = fopen(tex_file, 'w');
fprintf(fid, '\\begin{tabular}{|c|c|c|c|c|c|}\\hline\n');
fprintf(fid, ['\\v{S}tevilo meritev (N) & Povpr. $x$ & Povpr. $y$ & ' ...
              'Povpr. \\v{s}t. iteracij & MC Var($x$) & MC Var($y$) \\\\\\hline\n']);
for i = 1:num_tests
    fprintf(fid, '%d & %.4f & %.4f & %.1f & %.6f & %.6f \\\\\n', ...
        N_values(i), results_mean_xy(i,1), results_mean_xy(i,2), ...
        results_iters(i), results_mc_var(i,1), results_mc_var(i,2));
end
fprintf(fid, '\\hline\\end{tabular}\n');
fclose(fid);
fprintf('Tabela LaTeX shranjena v: %s\n\n', tex_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DEL 3: PRIMERJAVA UTEZENEGA IN NEUTEZENEGA (N = 100, Monte Carlo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('=====================================================\n');
fprintf('DEL 3: PRIMERJAVA UTEZENEGA IN NEUTEZENEGA\n');
fprintf('=====================================================\n');

M_cmp = 100;
N_cmp = 100;

est_unw_all = zeros(M_cmp, 2);
est_wls_all = zeros(M_cmp, 2);

for m = 1:M_cmp
    % Sveze meritve; ISTE podatke uporabimo za obe razlicici
    m1 = arrayfun(@(~) ping_stolp_1(), 1:N_cmp);
    m2 = arrayfun(@(~) ping_stolp_2(), 1:N_cmp);
    m3 = arrayfun(@(~) ping_stolp_3(), 1:N_cmp);

    d_cmp = [mean(m1); mean(m2); mean(m3)];
    v_cmp = [var(m1); var(m2); var(m3)];

    [xy_a, ~, ~] = run_lm(base_stations, d_cmp, eye(3), ...
        xy0, lambda0, max_iters, tol, lambda_max);
    [xy_b, ~, ~] = run_lm(base_stations, d_cmp, diag(1 ./ v_cmp), ...
        xy0, lambda0, max_iters, tol, lambda_max);

    est_unw_all(m, :) = xy_a;
    est_wls_all(m, :) = xy_b;
end

mean_unw = mean(est_unw_all, 1);   var_unw = var(est_unw_all, 0, 1);
mean_wls = mean(est_wls_all, 1);   var_wls = var(est_wls_all, 0, 1);
std_unw  = sqrt(var_unw);          std_wls = sqrt(var_wls);

fprintf('Primerjava cez M = %d ponovitev (vsakic N = %d):\n', M_cmp, N_cmp);
fprintf('  Neutezeno: povprecje = (%.4f, %.4f), Var = (%.6f, %.6f), Std = (%.4f, %.4f)\n', ...
    mean_unw(1), mean_unw(2), var_unw(1), var_unw(2), std_unw(1), std_unw(2));
fprintf('  Utezeno:   povprecje = (%.4f, %.4f), Var = (%.6f, %.6f), Std = (%.4f, %.4f)\n\n', ...
    mean_wls(1), mean_wls(2), var_wls(1), var_wls(2), std_wls(1), std_wls(2));

% Izvoz primerjave v tabelo LaTeX
tex_file_cmp = fullfile(tab_path, 'weighted_comparison.tex');
fid = fopen(tex_file_cmp, 'w');
fprintf(fid, '\\begin{tabular}{|l|c|c|c|c|c|c|}\\hline\n');
fprintf(fid, ['Varianta & Povpr. $x$ & Povpr. $y$ & Var($x$) & Var($y$) & ' ...
              'Std($x$) & Std($y$) \\\\\\hline\n']);
fprintf(fid, 'Neute\\v{z}eno & %.4f & %.4f & %.6f & %.6f & %.4f & %.4f \\\\\n', ...
    mean_unw(1), mean_unw(2), var_unw(1), var_unw(2), std_unw(1), std_unw(2));
fprintf(fid, 'Ute\\v{z}eno (WLS) & %.4f & %.4f & %.6f & %.6f & %.4f & %.4f \\\\\n', ...
    mean_wls(1), mean_wls(2), var_wls(1), var_wls(2), std_wls(1), std_wls(2));
fprintf(fid, '\\hline\\end{tabular}\n');
fclose(fid);
fprintf('Tabela LaTeX shranjena v: %s\n\n', tex_file_cmp);

%% Razsevni diagram: oblaka ocen za obe razlicici
figure; hold on; axis equal;
plot(est_unw_all(:,1), est_unw_all(:,2), 'bo', ...
    'MarkerSize', 5, 'DisplayName', 'Neutežene ocene');
plot(est_wls_all(:,1), est_wls_all(:,2), 'rx', ...
    'MarkerSize', 6, 'DisplayName', 'Utežene ocene');
plot(mean_unw(1), mean_unw(2), 'bs', 'MarkerSize', 12, ...
    'LineWidth', 2, 'DisplayName', 'Povprečje (neuteženo)');
plot(mean_wls(1), mean_wls(2), 'rs', 'MarkerSize', 12, ...
    'LineWidth', 2, 'DisplayName', 'Povprečje (uteženo)');

xlabel('x', 'Color', 'k');
ylabel('y', 'Color', 'k');
title(sprintf('Utežene proti neuteženim ocenam (M = %d, N = %d)', M_cmp, N_cmp), ...
    'FontWeight', 'bold', 'Color', 'k');
grid on;

set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', ...
    'GridColor', [0.6 0.6 0.6], 'GridAlpha', 0.3, 'FontSize', 12);
l = legend('Location', 'best');
set(l, 'TextColor', 'k', 'Color', 'w', 'EdgeColor', 'k');
set(gcf, 'Color', 'w');
saveas(gcf, fullfile(fig_path, 'weighted_vs_unweighted.png'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DEL 4: GRAFI ZA ANALIZO MERITEV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Graf 1: povprecno stevilo iteracij v odvisnosti od N
figure;
bar(N_values, results_iters, 'FaceColor', [0.2 0.6 1]);
xlabel('Število meritev N', 'Color', 'k');
ylabel('Povprečno število iteracij do konvergence', 'Color', 'k');
title('Hitrost konvergence v odvisnosti od števila meritev', 'FontWeight', 'bold', 'Color', 'k');
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
grid on;
set(gca, 'GridColor', [0.6 0.6 0.6], 'GridAlpha', 0.3);
set(gcf, 'Color', 'w');
saveas(gcf, fullfile(fig_path, 'iterations_vs_measurements.png'));

%% Graf 2: varianca Monte Carlo koncnih ocen v odvisnosti od N
figure;

subplot(2,1,1);
plot(N_values, results_mc_var(:,1), 'bs-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Število meritev N', 'Color', 'k');
ylabel('MC Var(x)', 'Color', 'k');
title('Varianca Monte Carlo končne ocene x v odvisnosti od N', 'Color', 'k');
grid on;

subplot(2,1,2);
plot(N_values, results_mc_var(:,2), 'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Število meritev N', 'Color', 'k');
ylabel('MC Var(y)', 'Color', 'k');
title('Varianca Monte Carlo končne ocene y v odvisnosti od N', 'Color', 'k');
grid on;

ax = findall(gcf, 'Type', 'axes');
for i = 1:length(ax)
    ax(i).XColor = 'k';
    ax(i).YColor = 'k';
    ax(i).GridColor = [0.7 0.7 0.7];
    ax(i).Color = 'w';
    ax(i).FontSize = 12;
end
set(gcf, 'Color', 'w');
saveas(gcf, fullfile(fig_path, 'variance_vs_measurements.png'));

%% Graf 3: razsevni diagram koncnih ocen Monte Carlo za vsak N
figure; hold on; axis equal;
colors = lines(num_tests);
for idx = 1:num_tests
    plot(all_estimates(:,1,idx), all_estimates(:,2,idx), '.', ...
        'MarkerSize', 10, 'Color', colors(idx,:), ...
        'DisplayName', sprintf('N = %d', N_values(idx)));
end

xlabel('x', 'Color', 'k');
ylabel('y', 'Color', 'k');
title(sprintf('Končne ocene Monte Carlo (M = %d zagonov na N)', M), ...
    'FontWeight', 'bold', 'Color', 'k');
grid on;

set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', ...
    'GridColor', [0.6 0.6 0.6], 'GridAlpha', 0.3, 'FontSize', 12);
l = legend('Location', 'best');
set(l, 'TextColor', 'k', 'Color', 'w', 'EdgeColor', 'k');
set(gcf, 'Color', 'w');
saveas(gcf, fullfile(fig_path, 'mc_scatter.png'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DEL 5: POVZETEK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('=====================================================\n');
fprintf('POVZETEK\n');
fprintf('=====================================================\n');
fprintf('Glavna lokalizacija (N = %d):\n', N_main);
fprintf('  Neutezena ocena = (%.4f, %.4f), iteracij = %d\n', ...
    xy_unw(1), xy_unw(2), iters_unw);
fprintf('  Utezena ocena   = (%.4f, %.4f), iteracij = %d\n\n', ...
    xy_wls(1), xy_wls(2), iters_wls);

fprintf('Statistika meritev po postajah (N = %d):\n', N_main);
for i = 1:3
    fprintf('  Postaja %d: povprecje = %.4f, varianca = %.4f, std = %.4f\n', ...
        i, d_main(i), var_main(i), std_main(i));
end
fprintf('\n');

fprintf('Analiza Monte Carlo (M = %d ponovitev na N):\n', M);
for i = 1:num_tests
    fprintf(['  N = %3d: povprecje (x, y) = (%.4f, %.4f), povpr. iteracij = %.1f, ' ...
             'MC Var = (%.6f, %.6f)\n'], ...
        N_values(i), results_mean_xy(i,1), results_mean_xy(i,2), ...
        results_iters(i), results_mc_var(i,1), results_mc_var(i,2));
end
fprintf('\n');

fprintf('Utezeno proti neutezenemu (M = %d, N = %d):\n', M_cmp, N_cmp);
fprintf('  Neutezeno: povprecje = (%.4f, %.4f), Std = (%.4f, %.4f)\n', ...
    mean_unw(1), mean_unw(2), std_unw(1), std_unw(2));
fprintf('  Utezeno:   povprecje = (%.4f, %.4f), Std = (%.4f, %.4f)\n\n', ...
    mean_wls(1), mean_wls(2), std_wls(1), std_wls(2));

fprintf('Shranjene slike:\n');
fprintf('  - %s\n', fullfile(fig_path, 'main_convergence_plot.png'));
fprintf('  - %s\n', fullfile(fig_path, 'main_geometry_plot.png'));
fprintf('  - %s\n', fullfile(fig_path, 'weighted_vs_unweighted.png'));
fprintf('  - %s\n', fullfile(fig_path, 'iterations_vs_measurements.png'));
fprintf('  - %s\n', fullfile(fig_path, 'variance_vs_measurements.png'));
fprintf('  - %s\n', fullfile(fig_path, 'mc_scatter.png'));
fprintf('Shranjene tabele:\n');
fprintf('  - %s\n', tex_file_var);
fprintf('  - %s\n', tex_file);
fprintf('  - %s\n', tex_file_cmp);
fprintf('=====================================================\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOKALNE FUNKCIJE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xy, iters, hist_xy] = run_lm(bs, d, W, xy0, lambda0, max_iters, tol, lambda_max)
% Levenberg-Marquardt z logiko sprejmi/zavrni in utezenim kriterijem.
%   bs         ... 3x2 koordinate baznih postaj
%   d          ... 3x1 (povprecene) izmerjene razdalje
%   W          ... 3x3 matrika utezi (eye(3) neutezeno, diag(1/var_i) za WLS)
% Korak je SPREJET le, ce se utezeni ostanek F'*W*F izboljsa; sicer se
% polozaj NE posodobi in lambda se poveca.
% hist_xy vsebuje zacetno tocko in vse SPREJETE korake.

    xy     = xy0;
    lambda = lambda0;

    hist_xy = zeros(max_iters + 1, 2);
    hist_xy(1, :) = xy;
    n_hist = 1;

    k = 0;
    for k = 1:max_iters
        % Vektor ostankov in Jacobijeva matrika v trenutni tocki
        F = (xy(1) - bs(:,1)).^2 + (xy(2) - bs(:,2)).^2 - d.^2;
        J = [2*(xy(1) - bs(:,1)), 2*(xy(2) - bs(:,2))];

        crit = F' * W * F;   % utezena kvadratna norma ostanka

        % Kandidatni korak Levenberg-Marquardt
        step = (J' * W * J + lambda * eye(2)) \ (-J' * W * F);
        xy_new = xy + step';

        % Utezeni ostanek v kandidatni tocki
        F_new = (xy_new(1) - bs(:,1)).^2 + (xy_new(2) - bs(:,2)).^2 - d.^2;
        crit_new = F_new' * W * F_new;

        if crit_new < crit
            % Izboljsanje -> korak SPREJMEMO
            xy = xy_new;
            n_hist = n_hist + 1;
            hist_xy(n_hist, :) = xy;
            lambda = lambda / 10;

            if norm(step) < tol
                break;
            end
        else
            % Ni izboljsanja -> korak ZAVRNEMO (ostanemo v xy), povecamo lambda
            lambda = lambda * 10;

            if lambda > lambda_max
                fprintf('OPOZORILO: lambda je presegla %.0e, LM se predcasno ustavi.\n', lambda_max);
                break;
            end
        end
    end

    iters   = k;
    hist_xy = hist_xy(1:n_hist, :);
end
