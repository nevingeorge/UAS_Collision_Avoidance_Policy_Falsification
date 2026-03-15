%% plot_encounter_cases.m
% Generate trajectory images for a nominal case (ACAS Xu avoids NMAC) and
% a failure case (ACAS Xu does NOT avoid NMAC).
%
% Run this script from the project root after simulate_encounter.m has been
% run (so paths are already set), OR run it standalone (it sets up paths
% internally). Saves two PNG files:
%   trajectory_nominal.png  - trial where ACAS Xu successfully avoids NMAC
%   trajectory_failure.png  - trial where ACAS Xu fails to avoid NMAC

clear; close all; clc;

%% ============================================================
% Path setup (mirrors simulate_encounter.m)
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
addpath(genpath(fullfile(root, 'nnv', 'code')));

%% ============================================================
% Simulation parameters
% ============================================================
rng_seed_base = 42;
tc            = 2;
tr            = 0.05;
sim_duration  = 80;
scale_mean    = [19791.091, 0, 0, 650, 600];
scale_range   = [60261, 2*pi, 2*pi, 1100, 1200];
NMAC_DIST_FT  = 500;
init_sep_ft   = 43736;

%% ============================================================
% Load encounter model
% ============================================================
mdl = UncorEncounterModel('parameters_filename', ...
    fullfile(bayes_path, 'model', 'uncor_1200only_fwse_v1p2.txt'));
start_dist    = cell(mdl.n_initial, 1);
start_dist{1} = 1;  start_dist{2} = 4;  start_dist{3} = 2;
mdl.start     = start_dist;

idxV    = find(strcmp(mdl.labels_initial, '"v"'));
idxDPsi = find(strcmp(mdl.labels_initial, '"\dot \psi"'));

N_search = 200;  % search this many trials for both case types
fprintf('Sampling encounter model (%d trials)...\n', N_search);
[inits_own, ~, ~, ~] = mdl.sample(N_search, 60, 'seed', rng_seed_base);
[inits_int, ~, ~, ~] = mdl.sample(N_search, 60, 'seed', rng_seed_base + 1000);

v_own_raw = inits_own(:, idxV);
v_int_raw = inits_int(:, idxV);
dpsi_int  = inits_int(:, idxDPsi);

v_own_ft_s = arrayfun(@(v) scale_speed(v, 650,  1750), v_own_raw);
v_int_ft_s = arrayfun(@(v) scale_speed(v, 600,  1800), v_int_raw);

%% ============================================================
% Load networks
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

%% ============================================================
% Search for nominal case (encounter-model sampling)
% ============================================================
nominal_traj = [];  nominal_advs = [];  nominal_trial = -1;
nominal_base = [];

fprintf('Searching for nominal case (encounter model)...\n');
for i = 1:N_search
    v_own = v_own_ft_s(i);
    v_int = v_int_ft_s(i);
    delta_heading = deg2rad(dpsi_int(i)) * tc;

    xo = [0; 0; pi/2];
    xi = [0; init_sep_ft; pi/2 + pi + delta_heading];
    [rho0, theta0, psi0] = environment(xo, xi);
    x0 = [xo; xi; rho0; theta0; psi0];

    [traj_a, min_a, advs_a] = run_sim(x0, v_own, v_int, nets, sim_duration, ...
                                       tc, tr, scale_mean, scale_range, true);
    [traj_b, min_b, ~]      = run_sim(x0, v_own, v_int, nets, sim_duration, ...
                                       tc, tr, scale_mean, scale_range, false);

    % Nominal: ACAS Xu avoids NMAC AND baseline would have caused one
    if ~(min_a < NMAC_DIST_FT) && (min_b < NMAC_DIST_FT)
        nominal_traj  = traj_a;
        nominal_base  = traj_b;
        nominal_advs  = advs_a;
        nominal_trial = i;
        fprintf('  Found nominal case: trial %d (ACAS min sep = %.0f ft, baseline = %.0f ft)\n', ...
            i, min_a, min_b);
        break;
    end
end

if isempty(nominal_traj)
    error('Could not find a nominal case in %d trials. Try increasing N_search.', N_search);
end

%% ============================================================
% Search for failure case (fuzzing: random + mutation)
% ============================================================
fprintf('\nSearching for failure case via fuzzing...\n');

% ACAS Xu full operational bounds
rho_min   = 1000;   rho_max   = 60261;
theta_min = -pi;    theta_max = pi;
psi_min   = -pi;    psi_max   = pi;
v_own_min = 650;    v_own_max = 1750;
v_int_min = 600;    v_int_max = 1800;

% Gaussian mutation step sizes
sigma_fuzz = [3000, pi/6, pi/6, 100, 100];

N_fuzz_rand = 500;
K_seeds     = 20;
N_mutants   = 20;

rng(0);
params_rand = [
    rho_min   + (rho_max   - rho_min)   * rand(N_fuzz_rand, 1), ...
    theta_min + (theta_max - theta_min) * rand(N_fuzz_rand, 1), ...
    psi_min   + (psi_max   - psi_min)   * rand(N_fuzz_rand, 1), ...
    v_own_min + (v_own_max - v_own_min) * rand(N_fuzz_rand, 1), ...
    v_int_min + (v_int_max - v_int_min) * rand(N_fuzz_rand, 1)  ...
];

min_seps_rand = zeros(N_fuzz_rand, 1);
for i = 1:N_fuzz_rand
    p  = params_rand(i, :);
    x0 = make_initial_state(p(1), p(2), p(3));
    [~, ms, ~] = run_sim(x0, p(4), p(5), nets, sim_duration, tc, tr, ...
                         scale_mean, scale_range, true);
    min_seps_rand(i) = ms;
    if mod(i, 100) == 0
        fprintf('  Phase 1: %d/%d  (NMACs: %d)\n', i, N_fuzz_rand, ...
            sum(min_seps_rand(1:i) < NMAC_DIST_FT));
    end
end
fprintf('Phase 1 done: %d/%d NMACs\n', sum(min_seps_rand < NMAC_DIST_FT), N_fuzz_rand);

% Check if we already have a failure from Phase 1
all_fuzz_params   = params_rand;
all_fuzz_min_seps = min_seps_rand;

% Phase 2: mutate the K nearest-to-failure seeds
[~, sort_idx] = sort(min_seps_rand, 'ascend');
seeds = params_rand(sort_idx(1:min(K_seeds, N_fuzz_rand)), :);

n_mut = K_seeds * N_mutants;
params_mut   = zeros(n_mut, 5);
min_seps_mut = zeros(n_mut, 1);
row = 0;
for s = 1:size(seeds, 1)
    for m = 1:N_mutants
        row = row + 1;
        p    = seeds(s,:) + sigma_fuzz .* randn(1, 5);
        p(1) = max(rho_min,   min(rho_max,   p(1)));
        p(2) = max(theta_min, min(theta_max, p(2)));
        p(3) = max(psi_min,   min(psi_max,   p(3)));
        p(4) = max(v_own_min, min(v_own_max, p(4)));
        p(5) = max(v_int_min, min(v_int_max, p(5)));
        params_mut(row,:) = p;
        x0 = make_initial_state(p(1), p(2), p(3));
        [~, ms, ~] = run_sim(x0, p(4), p(5), nets, sim_duration, tc, tr, ...
                             scale_mean, scale_range, true);
        min_seps_mut(row) = ms;
    end
    if mod(s, 5) == 0
        fprintf('  Phase 2: seed %d/%d  (mut NMACs: %d)\n', s, size(seeds,1), ...
            sum(min_seps_mut(1:row) < NMAC_DIST_FT));
    end
end
fprintf('Phase 2 done: %d/%d NMACs\n', sum(min_seps_mut < NMAC_DIST_FT), n_mut);

all_fuzz_params   = [params_rand;   params_mut];
all_fuzz_min_seps = [min_seps_rand; min_seps_mut];
all_fuzz_nmac     = all_fuzz_min_seps < NMAC_DIST_FT;

if sum(all_fuzz_nmac) == 0
    error('Fuzzing found no NMAC failures. Try increasing N_fuzz_rand or N_mutants.');
end

% Use the worst (smallest min-sep) failure
[~, worst_idx] = min(all_fuzz_min_seps);
p_fail = all_fuzz_params(worst_idx, :);
x0_fail = make_initial_state(p_fail(1), p_fail(2), p_fail(3));
[failure_traj, min_fail, failure_advs] = run_sim(x0_fail, p_fail(4), p_fail(5), nets, ...
    sim_duration, tc, tr, scale_mean, scale_range, true);
[failure_base, ~, ~] = run_sim(x0_fail, p_fail(4), p_fail(5), nets, ...
    sim_duration, tc, tr, scale_mean, scale_range, false);
failure_trial = worst_idx;
fprintf('  Failure case: min sep = %.0f ft  (rho=%.0f ft, theta=%.1f°, psi=%.1f°)\n', ...
    min_fail, p_fail(1), rad2deg(p_fail(2)), rad2deg(p_fail(3)));

%% ============================================================
% Helper: draw aircraft arrow
% ============================================================
function plot_aircraft(ax, x, y, hdg_rad, color, scale)
% Draw a simple triangle pointing in heading direction.
    L = scale;          % body length
    W = scale * 0.35;   % half-width
    % Local body coords (nose at front)
    local_pts = [L 0; -L/2 W; -L/2 -W; L 0]';
    R = [cos(hdg_rad) -sin(hdg_rad); sin(hdg_rad) cos(hdg_rad)];
    world_pts = R * local_pts;
    fill(ax, world_pts(1,:) + x, world_pts(2,:) + y, color, ...
        'EdgeColor', 'k', 'LineWidth', 0.8, 'FaceAlpha', 0.85);
end

%% ============================================================
% Helper: annotate heading arrows along a trajectory
% ============================================================
function plot_heading_arrows(ax, traj, color, every_n, arrow_scale)
    n = size(traj, 1);
    idx = 1:every_n:n;
    for k = idx
        x   = traj(k,1);  y   = traj(k,2);
        hdg = traj(k,3);
        dx  = arrow_scale * cos(hdg);
        dy  = arrow_scale * sin(hdg);
        quiver(ax, x, y, dx, dy, 0, 'Color', color, ...
            'MaxHeadSize', 2, 'LineWidth', 1.2);
    end
end

%% ============================================================
% Plot helper
% ============================================================
function fig = plot_trajectory(traj_acas, traj_base, advs, t_plot, ...
                                NMAC_DIST_FT, trial_num, case_label, color_title)
    fig = figure('Units', 'pixels', 'Position', [100 100 1100 480], ...
                 'Color', 'w');

    % --- Left panel: Spatial trajectory ---
    ax1 = subplot(1,2,1);
    hold(ax1, 'on');

    % Trajectories
    plot(ax1, traj_base(:,1),  traj_base(:,2),  '--', 'Color', [0.8 0.2 0.2], ...
        'LineWidth', 1.8, 'DisplayName', 'Ownship (no ACAS Xu)');
    plot(ax1, traj_acas(:,1),  traj_acas(:,2),  '-',  'Color', [0.1 0.4 0.9], ...
        'LineWidth', 2.2, 'DisplayName', 'Ownship (ACAS Xu)');
    plot(ax1, traj_acas(:,4),  traj_acas(:,5),  '-',  'Color', [0.15 0.6 0.15], ...
        'LineWidth', 2.2, 'DisplayName', 'Intruder');

    % Heading arrows every ~10 steps
    arrow_scale = 1200;
    arrow_step  = max(1, floor(size(traj_acas,1)/8));
    % Ownship (ACAS Xu)
    idx_arr = 1:arrow_step:size(traj_acas,1);
    for k = idx_arr
        quiver(ax1, traj_acas(k,1), traj_acas(k,2), ...
            arrow_scale*cos(traj_acas(k,3)), arrow_scale*sin(traj_acas(k,3)), ...
            0, 'Color', [0.1 0.4 0.9], 'MaxHeadSize', 2.5, 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
    end
    % Intruder
    for k = idx_arr
        quiver(ax1, traj_acas(k,4), traj_acas(k,5), ...
            arrow_scale*cos(traj_acas(k,6)), arrow_scale*sin(traj_acas(k,6)), ...
            0, 'Color', [0.15 0.6 0.15], 'MaxHeadSize', 2.5, 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
    end

    % Start markers
    scatter(ax1, traj_acas(1,1), traj_acas(1,2), 120, 's', ...
        'MarkerFaceColor', [0.1 0.4 0.9], 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.2, 'DisplayName', 'Ownship start');
    scatter(ax1, traj_acas(1,4), traj_acas(1,5), 120, 's', ...
        'MarkerFaceColor', [0.15 0.6 0.15], 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.2, 'DisplayName', 'Intruder start');

    % CPA marker + NMAC circle on ACAS Xu run
    dists_a = sqrt((traj_acas(:,4)-traj_acas(:,1)).^2 + ...
                   (traj_acas(:,5)-traj_acas(:,2)).^2);
    [min_d, cpa_idx] = min(dists_a);
    xc = traj_acas(cpa_idx,1);  yc = traj_acas(cpa_idx,2);
    theta_c = linspace(0, 2*pi, 120);
    plot(ax1, xc + NMAC_DIST_FT*cos(theta_c), yc + NMAC_DIST_FT*sin(theta_c), ...
        'm--', 'LineWidth', 1.8, 'HandleVisibility', 'off');
    scatter(ax1, xc, yc, 80, 'p', 'MarkerFaceColor', 'm', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1, 'DisplayName', ...
        sprintf('CPA (%.0f ft)', min_d));

    xlabel(ax1, 'X position (ft)', 'FontSize', 11);
    ylabel(ax1, 'Y position (ft)', 'FontSize', 11);
    title(ax1, sprintf('Encounter Trajectory — %s\n(Trial %d)', case_label, trial_num), ...
        'FontSize', 12, 'Color', color_title, 'FontWeight', 'bold');
    legend(ax1, 'Location', 'best', 'FontSize', 9);
    grid(ax1, 'on');  axis(ax1, 'equal');  box(ax1, 'on');

    % --- Right panel: Separation + advisories ---
    ax2 = subplot(1,2,2);
    yyaxis(ax2, 'left');

    dists_b = sqrt((traj_base(:,4)-traj_base(:,1)).^2 + ...
                   (traj_base(:,5)-traj_base(:,2)).^2);
    plot(ax2, t_plot, dists_a, '-',  'Color', [0.1 0.4 0.9], 'LineWidth', 2.2, ...
        'DisplayName', 'Sep. (ACAS Xu)');
    hold(ax2, 'on');
    plot(ax2, t_plot, dists_b, '--', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.8, ...
        'DisplayName', 'Sep. (no ACAS Xu)');
    yline(ax2, NMAC_DIST_FT, 'k--', 'LineWidth', 1.5, 'DisplayName', ...
        'NMAC threshold');
    ylabel(ax2, 'Separation (ft)', 'FontSize', 11);
    ax2.YColor = 'k';

    yyaxis(ax2, 'right');
    adv_deg = rad2deg(advs);
    stairs(ax2, t_plot(1:length(advs)), adv_deg, '-', ...
        'Color', [0.6 0.1 0.6], 'LineWidth', 1.5, 'DisplayName', 'Advisory (°/s)');
    ylabel(ax2, 'Advisory (°/s)', 'FontSize', 11);
    ax2.YColor = [0.5 0.0 0.5];
    yticks(ax2, [-3.0, -1.5, 0, 1.5, 3.0]);
    yticklabels(ax2, {'-3° (SR)', '-1.5° (WR)', '0 (COC)', '+1.5° (WL)', '+3° (SL)'});

    xlabel(ax2, 'Time (s)', 'FontSize', 11);
    title(ax2, 'Separation vs. Time & Advisory', 'FontSize', 12);
    legend(ax2, 'Location', 'northeast', 'FontSize', 9);
    grid(ax2, 'on');  box(ax2, 'on');

    set(fig, 'PaperPositionMode', 'auto');
end

%% ============================================================
% Build time vector
% ============================================================
t_plot_nom = linspace(0, sim_duration, size(nominal_traj, 1));
t_plot_fail = linspace(0, sim_duration, size(failure_traj, 1));

%% ============================================================
% Figure A: Nominal case
% ============================================================
fig_nom = plot_trajectory(nominal_traj, nominal_base, nominal_advs, ...
    t_plot_nom, NMAC_DIST_FT, nominal_trial, ...
    'Nominal (ACAS Xu avoids NMAC)', [0.05 0.45 0.05]);

exportgraphics(fig_nom, fullfile(root, 'trajectory_nominal.png'), ...
    'Resolution', 200, 'BackgroundColor', 'white');
fprintf('\nSaved: trajectory_nominal.png\n');

%% ============================================================
% Figure B: Failure case
% ============================================================
fig_fail = plot_trajectory(failure_traj, failure_base, failure_advs, ...
    t_plot_fail, NMAC_DIST_FT, failure_trial, ...
    'Failure (ACAS Xu does not avoid NMAC)', [0.75 0.1 0.1]);

exportgraphics(fig_fail, fullfile(root, 'trajectory_failure.png'), ...
    'Resolution', 200, 'BackgroundColor', 'white');
fprintf('Saved: trajectory_failure.png\n');

%% ============================================================
% Local functions
% ============================================================

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
    x = x0;

    traj = zeros(nSteps, 9);
    advs = zeros(nSteps, 1);
    traj(1,:) = x';
    advs(1)   = adv_own;

    for k = 1:nSteps-1
        [~, yp] = plant.evaluate(x, adv_own);
        x = yp(end,:)';

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

        traj(k+1,:) = x';
        advs(k+1)   = adv_own;
    end

    dists    = sqrt((traj(:,4)-traj(:,1)).^2 + (traj(:,5)-traj(:,2)).^2);
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


function v_ft_s = scale_speed(v_ktas, v_min_ft_s, v_max_ft_s)
    KTAS_TO_FT_S = 1.68780972;
    v_raw = v_ktas * KTAS_TO_FT_S;
    if v_raw >= v_min_ft_s && v_raw <= v_max_ft_s
        v_ft_s = v_raw;
    elseif v_raw < v_min_ft_s
        v_ft_s = v_min_ft_s + (v_raw / v_min_ft_s) * 50;
    else
        v_ft_s = v_max_ft_s;
    end
end


function x0 = make_initial_state(rho, theta, psi)
% Construct a 9-D initial state from ACAS Xu NN input geometry.
% Ownship at origin heading north; intruder at bearing theta, range rho,
% heading (psi + ownship_heading).
    own_hdg = pi / 2;
    int_x   = rho * cos(theta + own_hdg);
    int_y   = rho * sin(theta + own_hdg);
    int_hdg = psi + own_hdg;
    xo = [0; 0; own_hdg];
    xi = [int_x; int_y; int_hdg];
    [rho0, theta0, psi0] = environment(xo, xi);
    x0 = [xo; xi; rho0; theta0; psi0];
end
