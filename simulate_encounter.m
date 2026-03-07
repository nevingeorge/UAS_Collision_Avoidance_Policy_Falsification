%% simulate_encounter.m
% ACAS Xu + Bayesian Encounter Model Simulation
%
% Samples encounter parameters (speeds, headings) from the MIT Lincoln Lab
% uncorrelated Bayesian encounter model (em-model-manned-bayes), constructs
% near-collision geometries, and runs ACAS Xu in closed loop for N Monte
% Carlo trials. Reports NMAC avoidance rate vs. a no-ACAS-Xu baseline.
%
% Speed scaling note:
%   The encounter model represents general aviation (~60-250 KTAS = ~100-420 ft/s).
%   ACAS Xu was designed for commercial jets: v_own in [650,1750] ft/s,
%   v_int in [600,1800] ft/s. We linearly scale each sampled speed into the
%   ACAS Xu normalization bounds so the NN operates in its trained regime.
%   The encounter model still contributes the realistic *shape* of the speed
%   distribution and the heading geometry.

clear; close all; clc;

%% ============================================================
% Section 1: Setup paths
% ============================================================
root       = fileparts(mfilename('fullpath'));
acas_path  = fullfile(root, 'AcasXu');
bayes_path = fullfile(root, 'em-model-manned-bayes');

% Set AEM_DIR_BAYES if not already set
if isempty(getenv('AEM_DIR_BAYES'))
    setenv('AEM_DIR_BAYES', bayes_path);
end

% Initialize em-model-manned-bayes + em-core paths
run(fullfile(bayes_path, 'startup_bayes.m'));

% Add ACAS Xu paths (NNV must already be on path)
addpath(genpath(fullfile(acas_path, 'main',  'functions')));
addpath(genpath(fullfile(acas_path, 'other', 'functions')));
addpath(genpath(fullfile(acas_path, 'networks', 'nnv_format')));
addpath(genpath('/Users/nevingeorge/Desktop/CS238V/Project/nnv/code'));


%% ============================================================
% Section 2: Simulation parameters
% ============================================================
N             = 100;    % Monte Carlo trials
rng_seed_base = 42;    % Base random seed
tc            = 2;     % Control period (s)
tr            = 0.05;  % ODE integration step (s)
sim_duration  = 80;    % Simulation duration per trial (s)

% ACAS Xu normalization constants (from sim_TestPoints.m)
scale_mean  = [19791.091, 0, 0, 650, 600];
scale_range = [60261, 2*pi, 2*pi, 1100, 1200];

% NMAC threshold
NMAC_DIST_FT = 500;

% Base encounter geometry: ownship at origin, intruder ~43736 ft ahead
% This matches ACAS Xu test-point 9 style (head-on approach geometry)
init_sep_ft = 43736;

%% ============================================================
% Section 3: Load encounter model and sample parameters
% ============================================================
mdl = UncorEncounterModel('parameters_filename', ...
    fullfile(bayes_path, 'model', 'uncor_1200only_fwse_v1p2.txt'));

% Set start distribution: CONUS (G=1), Other airspace (A=4), 500-1200 ft AGL (L=2)
start_dist       = cell(mdl.n_initial, 1);
start_dist{1}    = 1;   % G: CONUS
start_dist{2}    = 4;   % A: Other airspace
start_dist{3}    = 2;   % L: 500-1200 ft AGL
mdl.start        = start_dist;

% Find indices of model variables
idxV    = find(strcmp(mdl.labels_initial, '"v"'));
idxDPsi = find(strcmp(mdl.labels_initial, '"\dot \psi"'));

% Sample N encounters independently for ownship and intruder
fprintf('Sampling encounter model...\n');
[inits_own, ~, ~, ~] = mdl.sample(N, 60, 'seed', rng_seed_base);
[inits_int, ~, ~, ~] = mdl.sample(N, 60, 'seed', rng_seed_base + 1000);

% Extract per-trial parameters
%   v in KTAS -> scale to ACAS Xu range (ft/s)
%   dpsi in deg/s -> used to offset intruder heading for varied geometry
v_own_raw  = inits_own(:, idxV);    % KTAS
v_int_raw  = inits_int(:, idxV);    % KTAS
dpsi_int   = inits_int(:, idxDPsi); % deg/s

v_own_ft_s = arrayfun(@(v) scale_speed(v, 650,  1750), v_own_raw);
v_int_ft_s = arrayfun(@(v) scale_speed(v, 600,  1800), v_int_raw);

fprintf('  Ownship speed range: [%.0f, %.0f] ft/s\n', min(v_own_ft_s), max(v_own_ft_s));
fprintf('  Intruder speed range: [%.0f, %.0f] ft/s\n', min(v_int_ft_s), max(v_int_ft_s));

%% ============================================================
% Section 4: Load neural networks (once, before Monte Carlo loop)
% ============================================================
fprintf('Loading ACAS Xu networks...\n');
% LoadAcasXu uses relative paths internally via load().
% MATLAB resolves relative paths from CWD, so we cd to AcasXu/main/
% (same convention as sim_TestPoints.m) before loading, then restore CWD.
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
fprintf('  Networks loaded.\n');

%% ============================================================
% Section 5: Monte Carlo loop
% ============================================================
min_sep_acas = zeros(N, 1);
min_sep_base = zeros(N, 1);
nmac_acas    = false(N, 1);
nmac_base    = false(N, 1);

% Store full trajectories and advisories for trial 1 (plotting)
traj1_acas = [];  traj1_base = [];  advs1 = [];

timeV = 0:tc:sim_duration;

fprintf('\nRunning %d Monte Carlo trials...\n', N);
for i = 1:N
    v_own = v_own_ft_s(i);
    v_int = v_int_ft_s(i);

    % Small heading offset derived from intruder heading rate sample.
    % Multiplied by tc (control period) to give a realistic angular deviation.
    delta_heading = deg2rad(dpsi_int(i)) * tc;

    % Initial states:
    %   Ownship  at origin heading north (pi/2 in standard math convention)
    %   Intruder placed init_sep_ft ahead, heading south toward ownship,
    %   plus small angular perturbation from encounter model.
    xo = [0; 0; pi/2];
    xi = [0; init_sep_ft; pi/2 + pi + delta_heading];

    % Derived NN inputs (rho, theta, psi) from initial geometry
    [rho0, theta0, psi0] = environment(xo, xi);

    % Full 9-D initial state: [xo; xi; rho; theta; psi]
    x0 = [xo; xi; rho0; theta0; psi0];

    % Run with ACAS Xu
    [traj_a, min_a, advs_a] = run_sim(x0, v_own, v_int, nets, sim_duration, ...
                                       tc, tr, scale_mean, scale_range, true);
    % Run baseline (no ACAS Xu)
    [traj_b, min_b, ~]      = run_sim(x0, v_own, v_int, nets, sim_duration, ...
                                       tc, tr, scale_mean, scale_range, false);

    min_sep_acas(i) = min_a;
    min_sep_base(i) = min_b;
    nmac_acas(i)    = min_a < NMAC_DIST_FT;
    nmac_base(i)    = min_b < NMAC_DIST_FT;

    % Save trial 1 for detailed plots
    if i == 1
        traj1_acas = traj_a;
        traj1_base = traj_b;
        advs1      = advs_a;
    end

    if mod(i, 10) == 0
        fprintf('  Trial %d/%d complete\n', i, N);
    end
end

%% ============================================================
% Section 6: Summary statistics
% ============================================================
rate_acas = mean(nmac_acas) * 100;
rate_base = mean(nmac_base) * 100;
improvement = rate_base - rate_acas;

fprintf('\n========== Monte Carlo Results (%d trials) ==========\n', N);
fprintf('  NMAC rate  WITH ACAS Xu : %.1f%%\n', rate_acas);
fprintf('  NMAC rate WITHOUT ACAS Xu: %.1f%%\n', rate_base);
fprintf('  NMAC avoidance improvement: %.1f percentage points\n', improvement);
fprintf('  Mean min separation WITH    ACAS Xu: %.0f ft\n', mean(min_sep_acas));
fprintf('  Mean min separation WITHOUT ACAS Xu: %.0f ft\n', mean(min_sep_base));
fprintf('=====================================================\n');

%% ============================================================
% Section 7: Visualization
% ============================================================

% Time vector for plots
t_plot = 0:tc:sim_duration;  % length = length(timeV)
if length(t_plot) ~= size(traj1_acas, 1)
    t_plot = linspace(0, sim_duration, size(traj1_acas, 1));
end

% --- Figure 1: Spatial Trajectory (trial 1) ---
figure(1); clf;
% Ownship trajectories
plot(traj1_acas(:,1), traj1_acas(:,2), 'b-',  'LineWidth', 2, 'DisplayName', 'Ownship (ACAS Xu)');
hold on;
plot(traj1_base(:,1), traj1_base(:,2), 'r--', 'LineWidth', 2, 'DisplayName', 'Ownship (Baseline)');
% Intruder trajectory (same initial, same constant heading, same speed → one trajectory)
plot(traj1_acas(:,4), traj1_acas(:,5), 'k-',  'LineWidth', 2, 'DisplayName', 'Intruder');
% Start markers
scatter(traj1_acas(1,1), traj1_acas(1,2), 80, 'd', 'b', 'filled', 'DisplayName', 'Own start');
scatter(traj1_acas(1,4), traj1_acas(1,5), 80, 'd', 'k', 'filled', 'DisplayName', 'Int start');
% NMAC circle at closest approach point (ACAS Xu run)
dists_a = sqrt((traj1_acas(:,4)-traj1_acas(:,1)).^2 + (traj1_acas(:,5)-traj1_acas(:,2)).^2);
[~, cpa_idx] = min(dists_a);
theta_circ = linspace(0, 2*pi, 100);
x_cpa = traj1_acas(cpa_idx, 1); y_cpa = traj1_acas(cpa_idx, 2);
plot(x_cpa + NMAC_DIST_FT*cos(theta_circ), y_cpa + NMAC_DIST_FT*sin(theta_circ), ...
    'm--', 'LineWidth', 1.5, 'DisplayName', '500 ft NMAC radius');
xlabel('X Position (ft)'); ylabel('Y Position (ft)');
title(sprintf('Trial 1 Encounter Trajectory'));
legend('Location', 'best'); grid on; axis equal;

% --- Figure 2: Separation Distance vs. Time (trial 1) ---
figure(2); clf;
dists_b = sqrt((traj1_base(:,4)-traj1_base(:,1)).^2 + (traj1_base(:,5)-traj1_base(:,2)).^2);
plot(t_plot, dists_a, 'b-',  'LineWidth', 2, 'DisplayName', 'ACAS Xu');
hold on;
plot(t_plot, dists_b, 'r--', 'LineWidth', 2, 'DisplayName', 'Baseline');
yline(NMAC_DIST_FT, 'k--', 'LineWidth', 1.5, 'DisplayName', 'NMAC threshold (500 ft)');
% Label minimum distances
text(t_plot(end)*0.7, min(dists_a)+200, sprintf('Min: %.0f ft', min(dists_a)), 'Color', 'b');
text(t_plot(end)*0.7, min(dists_b)-300, sprintf('Min: %.0f ft', min(dists_b)), 'Color', 'r');
xlabel('Time (s)'); ylabel('Separation Distance (ft)');
title('Trial 1: Separation Distance vs. Time');
legend('Location', 'best'); grid on;

% --- Figure 3: Advisory Timeline (trial 1) ---
figure(3); clf;
stairs(t_plot(1:length(advs1)), advs1, 'b-', 'LineWidth', 2);
yticks([deg2rad(-3.0), deg2rad(-1.5), 0, deg2rad(1.5), deg2rad(3.0)]);
yticklabels({'SR (-3.0°/s)', 'WR (-1.5°/s)', 'COC', 'WL (+1.5°/s)', 'SL (+3.0°/s)'});
xlabel('Time (s)'); ylabel('Advisory');
title('Trial 1: ACAS Xu Advisory Timeline');
grid on;

% --- Figure 4: Monte Carlo Summary ---
figure(4); clf;

subplot(1,2,1);
edges = linspace(0, max([min_sep_acas; min_sep_base])*1.05, 30);
histogram(min_sep_acas, edges, 'FaceColor', 'b', 'FaceAlpha', 0.6, 'DisplayName', 'ACAS Xu');
hold on;
histogram(min_sep_base, edges, 'FaceColor', 'r', 'FaceAlpha', 0.6, 'DisplayName', 'Baseline');
xline(NMAC_DIST_FT, 'k--', 'LineWidth', 2, 'DisplayName', 'NMAC threshold');
xlabel('Min Separation (ft)'); ylabel('Count');
title('Min Separation Distribution'); legend('Location', 'best'); grid on;

subplot(1,2,2);
bar([rate_base, rate_acas], 'FaceColor', 'flat', 'CData', [1 0 0; 0 0 1]);
set(gca, 'XTickLabel', {'Baseline', 'ACAS Xu'});
ylabel('NMAC Rate (%)');
title(sprintf('NMAC Rate (N=%d)\nImprovement: %.1f pp', N, improvement));
grid on; ylim([0, 100]);
text(1, rate_base + 2, sprintf('%.1f%%', rate_base), 'HorizontalAlignment', 'center');
text(2, rate_acas + 2, sprintf('%.1f%%', rate_acas), 'HorizontalAlignment', 'center');

sgtitle(sprintf('Monte Carlo Summary: N=%d trials', N));

%% ============================================================
% Local Functions
% ============================================================

function [traj, min_dist, advs] = run_sim(x0, v_own, v_int, nets, tf, tc, tr, ...
                                           scale_mean, scale_range, use_acas)
% RUN_SIM  Run one closed-loop encounter simulation.
%
% Inputs:
%   x0          - 9x1 initial state [xo(3); xi(3); rho; theta; psi]
%   v_own       - Ownship speed (ft/s)
%   v_int       - Intruder speed (ft/s)
%   nets        - Cell array of 5 FFNNS objects (ACAS Xu networks)
%   tf          - Simulation end time (s)
%   tc          - Control period (s)
%   tr          - ODE integration step (s)
%   scale_mean  - NN input normalization means (1x5)
%   scale_range - NN input normalization ranges (1x5)
%   use_acas    - logical: true = ACAS Xu active, false = no-op (COC always)
%
% Outputs:
%   traj     - (nSteps x 9) state trajectory matrix
%   min_dist - Minimum lateral separation distance (ft)
%   advs     - (nSteps x 1) advisory history (rad/s)

    % Output matrix selects [rho; theta; psi] from 9-D state
    outCp = [0 0 0 0 0 0 1 0 0;
             0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 0 0 1];

    % Parameterized dynamics
    dyns  = @(x, u) dyns_custom(x, u, v_own, v_int);
    plant = NonLinearODE(9, 1, dyns, tr, tc, outCp);

    % Advisory lookup: advisory value → network index
    advisory_map = [0, deg2rad(1.5), deg2rad(-1.5), deg2rad(3.0), deg2rad(-3.0)];

    timeV   = 0:tc:tf;
    nSteps  = length(timeV);
    adv_own = 0;       % Initial advisory: COC
    x       = x0;

    traj = zeros(nSteps, 9);
    advs = zeros(nSteps, 1);
    traj(1, :) = x';
    advs(1)    = adv_own;

    for k = 1:nSteps-1
        % Propagate plant one control period
        [~, yp] = plant.evaluate(x, adv_own);
        x = yp(end, :)';

        % Evaluate ACAS Xu network if enabled
        if use_acas
            ycp = outCp * x;
            u1  = ycp(1);
            u2  = set_angleRange(ycp(2));
            u3  = set_angleRange(ycp(3));

            % Normalize inputs per ACAS Xu specification
            uNN = ([u1, u2, u3, v_own, v_int] - scale_mean) ./ scale_range;

            % Select network based on previous advisory
            net_idx = find(advisory_map == adv_own, 1);
            if isempty(net_idx)
                net_idx = 1;
            end

            yNN     = nets{net_idx}.evaluate(uNN');
            adv_own = argmin_advise(yNN);
        end
        % If use_acas = false, adv_own stays 0 (COC) throughout

        traj(k+1, :) = x';
        advs(k+1)    = adv_own;
    end

    % Minimum lateral (horizontal) separation
    dists    = sqrt((traj(:,4) - traj(:,1)).^2 + (traj(:,5) - traj(:,2)).^2);
    min_dist = min(dists);
end


function dx = dyns_custom(x, u, v_own, v_int)
% DYNS_CUSTOM  Combined Dubins aircraft dynamics (2 aircraft) with
%              parameterized speeds. Same structure as dyns_tp1.m.
%
%   State x (9x1):
%     x(1), x(2), x(3)  - ownship  (x_ft, y_ft, heading_rad)
%     x(4), x(5), x(6)  - intruder (x_ft, y_ft, heading_rad)
%     x(7)              - rho   (range, ft)
%     x(8)              - theta (bearing angle, rad)
%     x(9)              - psi   (heading difference, rad)
%   Control u: ownship turn rate (rad/s)

    % Ownship kinematics
    dx(1,1) = v_own * cos(x(3));  % xdot (ft/s)
    dx(2,1) = v_own * sin(x(3));  % ydot (ft/s)
    dx(3,1) = u;                  % heading rate (rad/s)

    % Intruder kinematics (constant heading)
    dx(4,1) = v_int * cos(x(6));  % xdot (ft/s)
    dx(5,1) = v_int * sin(x(6));  % ydot (ft/s)
    dx(6,1) = 0;                  % constant heading

    % NN environment inputs (range and bearing derivatives)
    dx(7,1) = ((x(5)-x(2))*(dx(5)-dx(2)) + (x(4)-x(1))*(dx(4)-dx(1))) ...
              / sqrt((x(4)-x(1))^2 + (x(5)-x(2))^2);

    dx(8,1) = (2*(dx(5)-dx(2))*(x(4)-x(1)+x(7)) ...
              - 2*(x(5)-x(2))*(dx(4)-dx(1)+dx(7))) ...
              / (eps + (x(5)-x(2))^2 + (x(4)-x(1)+x(7))^2) - dx(3);

    dx(9,1) = dx(6) - dx(3);     % heading difference rate
end


function v_ft_s = scale_speed(v_ktas, v_min_ft_s, v_max_ft_s)
% SCALE_SPEED  Convert KTAS to ft/s and clamp/scale to ACAS Xu range.
%
% General aviation speeds (~60-250 KTAS = ~100-420 ft/s) are below the
% ACAS Xu normalization range (650-1750 ft/s for ownship). We clamp out-
% of-range values to the minimum with a small proportional variation so
% the N trials still have some speed diversity.

    KTAS_TO_FT_S = 1.68780972;
    v_raw = v_ktas * KTAS_TO_FT_S;

    if v_raw >= v_min_ft_s && v_raw <= v_max_ft_s
        % Within ACAS Xu range: use directly
        v_ft_s = v_raw;
    elseif v_raw < v_min_ft_s
        % Below range (general aviation): scale to add slight variation
        % near the minimum, preserving relative ordering across trials
        v_ft_s = v_min_ft_s + (v_raw / v_min_ft_s) * 50;
    else
        % Above range: clamp to maximum
        v_ft_s = v_max_ft_s;
    end
end
