%% fuzz_encounter.m
% Fuzzing-based failure search for ACAS Xu collision avoidance policy.
%
% Finds initial encounter geometries where ACAS Xu is active but fails to
% prevent an NMAC (min separation < 500 ft).
%
% Two-phase approach:
%   Phase 1 — Random: uniformly sample the 5-D ACAS Xu input space
%              (rho, theta, psi, v_own, v_int) across the full operational
%              envelope and simulate each encounter with ACAS Xu active.
%   Phase 2 — Mutation: take the K nearest-to-failure seeds from Phase 1,
%              apply Gaussian perturbations, and re-evaluate to find NMACs.
%
% A "failure" is defined as: ACAS Xu active, min lateral separation < 500 ft.

clear; close all; clc;

%% ============================================================
% Section 1: Setup paths (same as simulate_encounter.m)
% ============================================================
root       = fileparts(mfilename('fullpath'));
acas_path  = fullfile(root, 'AcasXu');
bayes_path = fullfile(root, 'em-model-manned-bayes');

if isempty(getenv('AEM_DIR_BAYES'))
    setenv('AEM_DIR_BAYES', bayes_path);
end
run(fullfile(bayes_path, 'startup_bayes.m'));

addpath(genpath(fullfile(acas_path, 'main',  'functions')));
addpath(genpath(fullfile(acas_path, 'other', 'functions')));
addpath(genpath(fullfile(acas_path, 'networks', 'nnv_format')));
addpath(genpath('/Users/nevingeorge/Desktop/CS238V/Project/nnv/code'));

%% ============================================================
% Section 2: Simulation parameters
% ============================================================
tc           = 2;      % Control period (s)
tr           = 0.05;   % ODE integration step (s)
sim_duration = 80;     % Simulation duration (s)
NMAC_DIST_FT = 500;    % NMAC threshold (ft)

scale_mean  = [19791.091, 0, 0, 650, 600];
scale_range = [60261, 2*pi, 2*pi, 1100, 1200];

%% ============================================================
% Section 3: Load networks
% ============================================================
fprintf('Loading ACAS Xu networks...\n');
orig_dir = pwd;
cd(fullfile(acas_path, 'main'));
nets = {
    LoadAcasXu('../networks/nnv_format/ACASXU_run2a_1_1_batch_2000.mat'), ...
    LoadAcasXu('../networks/nnv_format/ACASXU_run2a_2_1_batch_2000.mat'), ...
    LoadAcasXu('../networks/nnv_format/ACASXU_run2a_3_1_batch_2000.mat'), ...
    LoadAcasXu('../networks/nnv_format/ACASXU_run2a_4_1_batch_2000.mat'), ...
    LoadAcasXu('../networks/nnv_format/ACASXU_run2a_5_1_batch_2000.mat')  ...
};
cd(orig_dir);
fprintf('  Networks loaded.\n\n');

%% ============================================================
% Section 4: Fuzzing configuration
% ============================================================

% Phase 1: how many random scenarios to evaluate
N_random = 500;

% Phase 2: how many near-failure seeds to take, and mutants per seed
K_seeds  = 20;
N_mutants = 20;

% ACAS Xu operational bounds (from normalization constants)
rho_min   = 1000;   rho_max   = 60261;  % ft     (NN trained range: 0–60261)
theta_min = -pi;    theta_max = pi;     % rad    (bearing to intruder, ownship frame)
psi_min   = -pi;    psi_max   = pi;     % rad    (intruder heading - ownship heading)
v_own_min = 650;    v_own_max = 1750;   % ft/s
v_int_min = 600;    v_int_max = 1800;   % ft/s

% Gaussian mutation step sizes for Phase 2
% [rho(ft), theta(rad), psi(rad), v_own(ft/s), v_int(ft/s)]
sigma = [3000, pi/6, pi/6, 100, 100];

%% ============================================================
% Section 5: Phase 1 — Random fuzzing
% ============================================================
fprintf('=== Phase 1: Random Fuzzing (%d scenarios) ===\n', N_random);
rng(0);

params_rand = [
    rho_min   + (rho_max   - rho_min)   * rand(N_random, 1), ...
    theta_min + (theta_max - theta_min) * rand(N_random, 1), ...
    psi_min   + (psi_max   - psi_min)   * rand(N_random, 1), ...
    v_own_min + (v_own_max - v_own_min) * rand(N_random, 1), ...
    v_int_min + (v_int_max - v_int_min) * rand(N_random, 1)  ...
];

min_seps_rand = zeros(N_random, 1);
for i = 1:N_random
    p  = params_rand(i, :);
    x0 = make_initial_state(p(1), p(2), p(3));
    [~, ms, ~] = run_sim(x0, p(4), p(5), nets, sim_duration, tc, tr, ...
                         scale_mean, scale_range, true);
    min_seps_rand(i) = ms;

    if mod(i, 100) == 0
        fprintf('  %d/%d  (NMACs so far: %d)\n', i, N_random, ...
                sum(min_seps_rand(1:i) < NMAC_DIST_FT));
    end
end

nmac_rand = min_seps_rand < NMAC_DIST_FT;
fprintf('Phase 1 done: %d/%d NMACs found\n\n', sum(nmac_rand), N_random);

%% ============================================================
% Section 6: Phase 2 — Mutation-based fuzzing
% ============================================================
fprintf('=== Phase 2: Mutation Fuzzing (%d seeds × %d mutants) ===\n', ...
        K_seeds, N_mutants);

% Seed from the K scenarios with smallest min separation
[~, sort_idx] = sort(min_seps_rand, 'ascend');
seeds = params_rand(sort_idx(1:min(K_seeds, N_random)), :);

n_mut = K_seeds * N_mutants;
params_mut   = zeros(n_mut, 5);
min_seps_mut = zeros(n_mut, 1);

row = 0;
for s = 1:size(seeds, 1)
    for m = 1:N_mutants
        row = row + 1;

        % Gaussian perturbation, clipped to valid bounds
        p    = seeds(s, :) + sigma .* randn(1, 5);
        p(1) = fuzz_clip(p(1), rho_min,   rho_max);
        p(2) = fuzz_clip(p(2), theta_min, theta_max);
        p(3) = fuzz_clip(p(3), psi_min,   psi_max);
        p(4) = fuzz_clip(p(4), v_own_min, v_own_max);
        p(5) = fuzz_clip(p(5), v_int_min, v_int_max);

        params_mut(row, :) = p;

        x0 = make_initial_state(p(1), p(2), p(3));
        [~, ms, ~] = run_sim(x0, p(4), p(5), nets, sim_duration, tc, tr, ...
                             scale_mean, scale_range, true);
        min_seps_mut(row) = ms;
    end

    if mod(s, 5) == 0
        fprintf('  Seed %d/%d  (mut NMACs so far: %d)\n', s, size(seeds, 1), ...
                sum(min_seps_mut(1:row) < NMAC_DIST_FT));
    end
end

nmac_mut = min_seps_mut < NMAC_DIST_FT;
fprintf('Phase 2 done: %d/%d NMACs found\n\n', sum(nmac_mut), n_mut);

%% ============================================================
% Section 7: Aggregate and report
% ============================================================
all_params   = [params_rand;   params_mut];
all_min_seps = [min_seps_rand; min_seps_mut];
all_nmac     = all_min_seps < NMAC_DIST_FT;
N_total      = N_random + n_mut;

fprintf('========== Fuzzing Results ==========\n');
fprintf('  Total scenarios evaluated : %d\n', N_total);
fprintf('  Total NMAC failures found : %d (%.1f%%)\n', ...
        sum(all_nmac), 100 * mean(all_nmac));
fprintf('  Min separation (w/ ACAS Xu): %.0f ft\n', min(all_min_seps));
fprintf('=====================================\n\n');

if sum(all_nmac) > 0
    fprintf('Failure cases (rho ft | theta deg | psi deg | v_own ft/s | v_int ft/s | min_sep ft):\n');
    fp = all_params(all_nmac, :);
    fs = all_min_seps(all_nmac);
    [fs_sorted, si] = sort(fs, 'ascend');
    fp_sorted = fp(si, :);
    for k = 1:size(fp_sorted, 1)
        fprintf('  rho=%6.0f  theta=%6.1f°  psi=%6.1f°  v_own=%5.0f  v_int=%5.0f  -> %4.0f ft\n', ...
                fp_sorted(k,1), rad2deg(fp_sorted(k,2)), rad2deg(fp_sorted(k,3)), ...
                fp_sorted(k,4), fp_sorted(k,5), fs_sorted(k));
    end
    fprintf('\n');
end

%% ============================================================
% Section 8: Visualization
% ============================================================

% --- Figure 1: Parameter-space coverage and min separation heatmaps ---
figure(1); clf;

subplot(2, 2, 1);
scatter(rad2deg(all_params(:,2)), rad2deg(all_params(:,3)), 20, ...
        min(all_min_seps, 5000), 'filled');
colorbar; colormap(flipud(hot));
clim([0, 5000]);
hold on;
if sum(all_nmac) > 0
    scatter(rad2deg(all_params(all_nmac,2)), rad2deg(all_params(all_nmac,3)), ...
            60, 'g', 'filled', 'Marker', 'p');
end
xlabel('\theta (deg)'); ylabel('\psi (deg)');
title('Min separation by bearing and heading diff');
legend('scenarios', 'NMACs', 'Location', 'best');

subplot(2, 2, 2);
scatter(all_params(:,1) / 1000, all_min_seps, 15, 'b', 'filled');
hold on;
yline(NMAC_DIST_FT, 'r--', 'LineWidth', 2, 'DisplayName', 'NMAC threshold');
if sum(all_nmac) > 0
    scatter(all_params(all_nmac,1) / 1000, all_min_seps(all_nmac), ...
            60, 'g', 'filled', 'Marker', 'p', 'DisplayName', 'NMAC');
end
xlabel('Initial range \rho (kft)'); ylabel('Min separation (ft)');
title('Min separation vs. initial range');
legend('Location', 'best'); grid on;

subplot(2, 2, 3);
scatter(all_params(:,4), all_params(:,5), 20, ...
        min(all_min_seps, 5000), 'filled');
colorbar; colormap(flipud(hot));
clim([0, 5000]);
hold on;
if sum(all_nmac) > 0
    scatter(all_params(all_nmac,4), all_params(all_nmac,5), ...
            60, 'g', 'filled', 'Marker', 'p');
end
xlabel('v\_own (ft/s)'); ylabel('v\_int (ft/s)');
title('Min separation by speeds');

subplot(2, 2, 4);
histogram(all_min_seps, 40, 'FaceColor', 'b', 'FaceAlpha', 0.6);
hold on;
xline(NMAC_DIST_FT, 'r--', 'LineWidth', 2);
xlabel('Min separation (ft)'); ylabel('Count');
title(sprintf('Separation distribution  (%d NMACs / %d total)', ...
              sum(all_nmac), N_total));
grid on;

sgtitle('ACAS Xu Fuzzing Results');

% --- Figure 2: Worst failure trajectory (if any found) ---
if sum(all_nmac) > 0
    [~, worst_idx] = min(all_min_seps);
    p_worst  = all_params(worst_idx, :);
    x0_worst = make_initial_state(p_worst(1), p_worst(2), p_worst(3));
    [traj_w, ~, advs_w] = run_sim(x0_worst, p_worst(4), p_worst(5), nets, ...
                                  sim_duration, tc, tr, scale_mean, scale_range, true);

    t_plot = linspace(0, sim_duration, size(traj_w, 1));
    dists_w = sqrt((traj_w(:,4) - traj_w(:,1)).^2 + ...
                   (traj_w(:,5) - traj_w(:,2)).^2);

    figure(2); clf;

    subplot(1, 3, 1);
    plot(traj_w(:,1), traj_w(:,2), 'b-',  'LineWidth', 2, 'DisplayName', 'Ownship');
    hold on;
    plot(traj_w(:,4), traj_w(:,5), 'r-',  'LineWidth', 2, 'DisplayName', 'Intruder');
    scatter(traj_w(1,1), traj_w(1,2), 80, 'd', 'b', 'filled', 'HandleVisibility', 'off');
    scatter(traj_w(1,4), traj_w(1,5), 80, 'd', 'r', 'filled', 'HandleVisibility', 'off');
    theta_c = linspace(0, 2*pi, 100);
    [~, cpa] = min(dists_w);
    plot(traj_w(cpa,1) + NMAC_DIST_FT*cos(theta_c), ...
         traj_w(cpa,2) + NMAC_DIST_FT*sin(theta_c), ...
         'm--', 'LineWidth', 1.5, 'DisplayName', '500 ft radius');
    xlabel('X (ft)'); ylabel('Y (ft)');
    title(sprintf('Trajectory  (min sep = %.0f ft)', min(dists_w)));
    legend('Location', 'best'); grid on; axis equal;

    subplot(1, 3, 2);
    plot(t_plot, dists_w, 'b-', 'LineWidth', 2);
    hold on;
    yline(NMAC_DIST_FT, 'r--', 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Separation (ft)');
    title('Separation vs. Time');
    grid on;

    subplot(1, 3, 3);
    stairs(t_plot(1:length(advs_w)), advs_w, 'b-', 'LineWidth', 2);
    yticks([deg2rad(-3.0), deg2rad(-1.5), 0, deg2rad(1.5), deg2rad(3.0)]);
    yticklabels({'SR (-3°/s)', 'WR (-1.5°/s)', 'COC', 'WL (+1.5°/s)', 'SL (+3°/s)'});
    xlabel('Time (s)'); ylabel('Advisory');
    title('Advisory Timeline');
    grid on;

    sgtitle(sprintf(['Worst Failure: \\rho=%.0f ft, \\theta=%.1f°, ' ...
                     '\\psi=%.1f°,  v_{own}=%.0f,  v_{int}=%.0f ft/s'], ...
            p_worst(1), rad2deg(p_worst(2)), rad2deg(p_worst(3)), ...
            p_worst(4), p_worst(5)));
end

%% ============================================================
% Local functions
% ============================================================

function x0 = make_initial_state(rho, theta, psi)
% Construct a 9-D initial state from ACAS Xu NN input parameters.
%
% Ownship is fixed at the origin heading north (pi/2 rad in math convention).
% Intruder is placed at bearing theta (relative to ownship nose) at range rho,
% and heading (psi + ownship_heading).
%
% The constructed geometry satisfies:
%   environment(xo, xi) == [rho, theta, psi]  (up to angle wrapping)

    own_hdg = pi / 2;                         % north

    % Intruder position: rotate "ahead" direction by theta
    int_x = rho * cos(theta + own_hdg);
    int_y = rho * sin(theta + own_hdg);
    int_hdg = psi + own_hdg;

    xo = [0; 0; own_hdg];
    xi = [int_x; int_y; int_hdg];

    [rho0, theta0, psi0] = environment(xo, xi);
    x0 = [xo; xi; rho0; theta0; psi0];
end


function v = fuzz_clip(x, lo, hi)
    v = max(lo, min(hi, x));
end


function [traj, min_dist, advs] = run_sim(x0, v_own, v_int, nets, tf, tc, tr, ...
                                          scale_mean, scale_range, use_acas)
    outCp = [0 0 0 0 0 0 1 0 0;
             0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 0 0 1];

    dyns  = @(x, u) dyns_custom(x, u, v_own, v_int);
    plant = NonLinearODE(9, 1, dyns, tr, tc, outCp);

    advisory_map = [0, deg2rad(1.5), deg2rad(-1.5), deg2rad(3.0), deg2rad(-3.0)];

    timeV  = 0:tc:tf;
    nSteps = length(timeV);
    adv_own = 0;
    x       = x0;

    traj = zeros(nSteps, 9);
    advs = zeros(nSteps, 1);
    traj(1, :) = x';
    advs(1)    = adv_own;

    for k = 1:nSteps-1
        [~, yp] = plant.evaluate(x, adv_own);
        x = yp(end, :)';

        if use_acas
            ycp = outCp * x;
            u1  = ycp(1);
            u2  = set_angleRange(ycp(2));
            u3  = set_angleRange(ycp(3));

            uNN = ([u1, u2, u3, v_own, v_int] - scale_mean) ./ scale_range;

            net_idx = find(advisory_map == adv_own, 1);
            if isempty(net_idx), net_idx = 1; end

            yNN     = nets{net_idx}.evaluate(uNN');
            adv_own = argmin_advise(yNN);
        end

        traj(k+1, :) = x';
        advs(k+1)    = adv_own;
    end

    dists    = sqrt((traj(:,4) - traj(:,1)).^2 + (traj(:,5) - traj(:,2)).^2);
    min_dist = min(dists);
end


function dx = dyns_custom(x, u, v_own, v_int)
    dx(1,1) = v_own * cos(x(3));
    dx(2,1) = v_own * sin(x(3));
    dx(3,1) = u;

    dx(4,1) = v_int * cos(x(6));
    dx(5,1) = v_int * sin(x(6));
    dx(6,1) = 0;

    dx(7,1) = ((x(5)-x(2))*(dx(5)-dx(2)) + (x(4)-x(1))*(dx(4)-dx(1))) ...
              / sqrt((x(4)-x(1))^2 + (x(5)-x(2))^2);

    dx(8,1) = (2*(dx(5)-dx(2))*(x(4)-x(1)+x(7)) ...
              - 2*(x(5)-x(2))*(dx(4)-dx(1)+dx(7))) ...
              / (eps + (x(5)-x(2))^2 + (x(4)-x(1)+x(7))^2) - dx(3);

    dx(9,1) = dx(6) - dx(3);
end
