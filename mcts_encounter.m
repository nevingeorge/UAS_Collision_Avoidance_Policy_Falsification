%% mcts_encounter.m
% Monte Carlo Tree Search for ACAS Xu failure discovery.
%
% The 5-D initial-condition space (rho, theta, psi, v_own, v_int) is
% treated as a search tree. Each node represents a hyperrectangular region.
% MCTS adaptively focuses the simulation budget on regions that have
% historically produced near-failures or failures (NMAC < 500 ft).
%
% Algorithm per iteration:
%   1. Selection   — traverse from root via UCB1 to a leaf node
%   2. Rollout     — sample N_rollout points uniformly within that leaf,
%                    simulate each with ACAS Xu active
%   3. Expansion   — if the leaf has >= expand_thresh visits, bisect it
%                    along the longest normalized dimension
%   4. Backprop    — propagate reward and failure counts to all ancestors
%
% Reward: exp(-min_sep / reward_decay_ft)
%   -> 1.0 at NMAC (min_sep = 0), ~0.61 at 1000 ft, ~0.007 at 5000 ft
%
% Tree data structure (struct-of-arrays, all indexed by node id):
%   tree.bounds{k}       — 5x2 [lo, hi] for each parameter dim
%   tree.n_visits(k)     — number of simulations through node k
%   tree.sum_reward(k)   — cumulative reward in subtree of k
%   tree.n_failures(k)   — cumulative NMACs found in subtree of k
%   tree.best_min_sep(k) — best (smallest) min separation in subtree of k
%   tree.parent(k)       — parent node index (0 = root)
%   tree.children{k}     — [left_idx, right_idx] or [] if leaf

clear; close all; clc;

%% ============================================================
% Section 1: Setup paths
% ============================================================
root_dir   = fileparts(mfilename('fullpath'));
acas_path  = fullfile(root_dir, 'AcasXu');
bayes_path = fullfile(root_dir, 'em-model-manned-bayes');

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
tc           = 2;
tr           = 0.05;
sim_duration = 80;
NMAC_DIST_FT = 500;
scale_mean   = [19791.091, 0, 0, 650, 600];
scale_range  = [60261, 2*pi, 2*pi, 1100, 1200];

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
% Section 4: MCTS configuration
% ============================================================

% Search budget
N_iter        = 200;   % MCTS iterations  (total sims = N_iter * N_rollout)
N_rollout     = 5;     % rollout simulations per leaf visit
expand_thresh = 10;    % min visits before a leaf is split into 2 children

% UCB1 exploration constant (sqrt(2) is standard; increase to explore more)
C_ucb = sqrt(2);

% Reward decay: exp(-min_sep / reward_decay_ft)
reward_decay_ft = 1000;

rng_seed = 42;

% 5-D parameter bounds: [rho(ft), theta(rad), psi(rad), v_own(ft/s), v_int(ft/s)]
% Corresponds to the ACAS Xu NN input normalization bounds.
param_bounds = [
    1000,  60261;   % rho     (range)
    -pi,   pi;      % theta   (bearing to intruder, ownship frame)
    -pi,   pi;      % psi     (intruder heading - ownship heading)
    650,   1750;    % v_own
    600,   1800;    % v_int
];

% Full-space ranges, used for normalized dimension selection during expansion
full_ranges = param_bounds(:,2) - param_bounds(:,1);

%% ============================================================
% Section 5: Tree initialization
% ============================================================
% Node 1 = root, covers the entire parameter space
tree.bounds       = {param_bounds};
tree.n_visits     = 0;
tree.sum_reward   = 0;
tree.n_failures   = 0;
tree.best_min_sep = inf;
tree.parent       = 0;        % 0 signals "no parent"
tree.children     = {[]};     % [] signals leaf node

% Logging
max_sims     = N_iter * N_rollout;
all_params   = zeros(max_sims, 5);
all_min_seps = zeros(max_sims, 1);
fail_log     = zeros(0, 6);   % columns: [rho, theta, psi, v_own, v_int, min_sep]
cum_failures = zeros(N_iter, 1);
sim_count    = 0;

rng(rng_seed);

%% ============================================================
% Section 6: MCTS main loop
% ============================================================
fprintf('=== MCTS Failure Search: %d iters x %d rollouts = %d simulations ===\n', ...
        N_iter, N_rollout, N_iter * N_rollout);

for iter = 1:N_iter

    % ---- 1. Selection: walk tree via UCB1 to a leaf ----
    node_idx = mcts_select(tree, C_ucb);

    % ---- 2. Rollout: sample N_rollout points from leaf's region ----
    bounds = tree.bounds{node_idx};
    iter_min_seps = zeros(N_rollout, 1);
    iter_rewards  = zeros(N_rollout, 1);

    for r = 1:N_rollout
        % Uniform sample within the leaf's hyperrectangle
        p = bounds(:,1)' + (bounds(:,2) - bounds(:,1))' .* rand(1, 5);

        x0 = make_initial_state(p(1), p(2), p(3));
        [~, ms, ~] = run_sim(x0, p(4), p(5), nets, sim_duration, tc, tr, ...
                             scale_mean, scale_range, true);

        sim_count = sim_count + 1;
        all_params(sim_count, :) = p;
        all_min_seps(sim_count)  = ms;
        iter_min_seps(r) = ms;
        iter_rewards(r)  = exp(-ms / reward_decay_ft);

        if ms < NMAC_DIST_FT
            fail_log(end+1, :) = [p, ms]; %#ok<AGROW>
        end
    end

    % ---- 3. Expansion: bisect leaf if it has been visited enough ----
    if tree.n_visits(node_idx) >= expand_thresh && isempty(tree.children{node_idx})
        tree = mcts_expand(tree, node_idx, full_ranges);
    end

    % ---- 4. Backpropagation: update node and all ancestors ----
    tree = mcts_backprop(tree, node_idx, ...
                         sum(iter_rewards), N_rollout, ...
                         sum(iter_min_seps < NMAC_DIST_FT), ...
                         min(iter_min_seps));

    cum_failures(iter) = size(fail_log, 1);

    if mod(iter, 25) == 0 || iter == 1
        n_nodes = length(tree.n_visits);
        fprintf('  Iter %3d/%d | sims: %4d | nodes: %3d | failures: %d | best sep: %.0f ft\n', ...
                iter, N_iter, sim_count, n_nodes, ...
                size(fail_log, 1), min([all_min_seps(1:sim_count); inf]));
    end
end

%% ============================================================
% Section 7: Results
% ============================================================
all_params   = all_params(1:sim_count, :);
all_min_seps = all_min_seps(1:sim_count);

fprintf('\n========== MCTS Results ==========\n');
fprintf('  Total simulations    : %d\n', sim_count);
fprintf('  Tree nodes created   : %d\n', length(tree.n_visits));
fprintf('  NMAC failures found  : %d (%.1f%%)\n', size(fail_log, 1), ...
        100 * size(fail_log, 1) / sim_count);
fprintf('  Min separation found : %.0f ft\n', min(all_min_seps));
fprintf('===================================\n\n');

if ~isempty(fail_log)
    [~, si] = sort(fail_log(:,6), 'ascend');
    fs = fail_log(si, :);
    fprintf('Failure cases sorted by severity:\n');
    fprintf('  %-10s %-12s %-12s %-10s %-10s %-12s\n', ...
            'rho (ft)', 'theta (deg)', 'psi (deg)', 'v_own', 'v_int', 'min_sep (ft)');
    for k = 1:size(fs, 1)
        fprintf('  %-10.0f %-12.1f %-12.1f %-10.0f %-10.0f %-12.0f\n', ...
                fs(k,1), rad2deg(fs(k,2)), rad2deg(fs(k,3)), ...
                fs(k,4), fs(k,5), fs(k,6));
    end
    fprintf('\n');
end

%% ============================================================
% Section 8: Visualization
% ============================================================

% --- Figure 1: MCTS convergence ---
figure(1); clf;

subplot(1, 2, 1);
plot(1:N_iter, cum_failures, 'b-', 'LineWidth', 2);
xlabel('MCTS Iteration'); ylabel('Cumulative Failures Found');
title('Failure Discovery over Iterations');
grid on;

subplot(1, 2, 2);
[sorted_seps, sort_order] = sort(all_min_seps, 'ascend');
plot(1:sim_count, sorted_seps, 'b-', 'LineWidth', 1.5);
hold on;
yline(NMAC_DIST_FT, 'r--', 'LineWidth', 2, 'DisplayName', 'NMAC (500 ft)');
xlabel('Simulation index (sorted by min sep)'); ylabel('Min separation (ft)');
title('Sorted min separations across all simulations');
legend('Location', 'best'); grid on;

sgtitle('MCTS Search Progress');

% --- Figure 2: Parameter-space coverage ---
figure(2); clf;

subplot(2, 2, 1);
scatter(rad2deg(all_params(:,2)), rad2deg(all_params(:,3)), 15, ...
        min(all_min_seps, 5000), 'filled');
colorbar; colormap(flipud(hot)); clim([0, 5000]);
if ~isempty(fail_log)
    hold on;
    scatter(rad2deg(fail_log(:,2)), rad2deg(fail_log(:,3)), 80, ...
            'g', 'filled', 'Marker', 'p', 'DisplayName', 'NMAC');
    legend('Location', 'best');
end
xlabel('\theta (deg)'); ylabel('\psi (deg)');
title('Coverage: bearing \theta vs heading diff \psi');

subplot(2, 2, 2);
scatter(all_params(:,1) / 1000, all_min_seps, 15, 'b', 'filled');
hold on;
yline(NMAC_DIST_FT, 'r--', 'LineWidth', 2, 'DisplayName', 'NMAC threshold');
if ~isempty(fail_log)
    scatter(fail_log(:,1) / 1000, fail_log(:,6), 80, 'g', 'filled', ...
            'Marker', 'p', 'DisplayName', 'NMAC');
end
xlabel('Initial \rho (kft)'); ylabel('Min separation (ft)');
title('Min separation vs. initial range');
legend('Location', 'best'); grid on;

subplot(2, 2, 3);
scatter(all_params(:,4), all_params(:,5), 15, min(all_min_seps, 5000), 'filled');
colorbar; colormap(flipud(hot)); clim([0, 5000]);
if ~isempty(fail_log)
    hold on;
    scatter(fail_log(:,4), fail_log(:,5), 80, 'g', 'filled', 'Marker', 'p');
end
xlabel('v\_own (ft/s)'); ylabel('v\_int (ft/s)');
title('Coverage: ownship vs intruder speed');

subplot(2, 2, 4);
histogram(all_min_seps, 40, 'FaceColor', 'b', 'FaceAlpha', 0.6);
hold on;
xline(NMAC_DIST_FT, 'r--', 'LineWidth', 2, 'DisplayName', 'NMAC threshold');
xlabel('Min separation (ft)'); ylabel('Count');
title(sprintf('Separation histogram  (%d NMACs / %d sims)', ...
              size(fail_log, 1), sim_count));
grid on;

sgtitle('MCTS Parameter-Space Coverage');

% --- Figure 3: Tree statistics ---
figure(3); clf;

subplot(1, 2, 1);
histogram(tree.n_visits, max(tree.n_visits), 'FaceColor', 'b', 'FaceAlpha', 0.6);
xlabel('Visits per node'); ylabel('Number of nodes');
title(sprintf('Node visit distribution  (%d nodes total)', length(tree.n_visits)));
grid on;

subplot(1, 2, 2);
mean_rewards = tree.sum_reward ./ max(tree.n_visits, 1);
histogram(mean_rewards, 30, 'FaceColor', 'r', 'FaceAlpha', 0.6);
xlabel('Mean reward per node'); ylabel('Number of nodes');
title('Node reward distribution (higher = more dangerous region)');
grid on;

sgtitle('MCTS Tree Statistics');

% --- Figure 4: Worst failure trajectory ---
if ~isempty(fail_log)
    [~, widx] = min(fail_log(:, 6));
    pw   = fail_log(widx, 1:5);
    x0w  = make_initial_state(pw(1), pw(2), pw(3));
    [traj_w, ~, advs_w] = run_sim(x0w, pw(4), pw(5), nets, sim_duration, tc, tr, ...
                                  scale_mean, scale_range, true);
    t_plot = linspace(0, sim_duration, size(traj_w, 1));
    dists_w = sqrt((traj_w(:,4) - traj_w(:,1)).^2 + (traj_w(:,5) - traj_w(:,2)).^2);

    figure(4); clf;

    subplot(1, 3, 1);
    plot(traj_w(:,1), traj_w(:,2), 'b-', 'LineWidth', 2, 'DisplayName', 'Ownship');
    hold on;
    plot(traj_w(:,4), traj_w(:,5), 'r-', 'LineWidth', 2, 'DisplayName', 'Intruder');
    scatter(traj_w(1,1), traj_w(1,2), 80, 'd', 'b', 'filled', 'HandleVisibility', 'off');
    scatter(traj_w(1,4), traj_w(1,5), 80, 'd', 'r', 'filled', 'HandleVisibility', 'off');
    [~, cpa_idx] = min(dists_w);
    tc2 = linspace(0, 2*pi, 100);
    plot(traj_w(cpa_idx,1) + NMAC_DIST_FT*cos(tc2), ...
         traj_w(cpa_idx,2) + NMAC_DIST_FT*sin(tc2), ...
         'm--', 'LineWidth', 1.5, 'DisplayName', '500 ft radius');
    xlabel('X (ft)'); ylabel('Y (ft)');
    title(sprintf('Trajectory  (min sep = %.0f ft)', min(dists_w)));
    legend('Location', 'best'); grid on; axis equal;

    subplot(1, 3, 2);
    plot(t_plot, dists_w, 'b-', 'LineWidth', 2);
    hold on;
    yline(NMAC_DIST_FT, 'r--', 'LineWidth', 2, 'DisplayName', 'NMAC threshold');
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

    sgtitle(sprintf( ...
        'Worst MCTS Failure:  \\rho=%.0f ft,  \\theta=%.1f°,  \\psi=%.1f°,  v_{own}=%.0f,  v_{int}=%.0f ft/s', ...
        pw(1), rad2deg(pw(2)), rad2deg(pw(3)), pw(4), pw(5)));
end

%% ============================================================
% MCTS helper functions
% ============================================================

function leaf_idx = mcts_select(tree, C)
% Traverse from root to a leaf, choosing children by UCB1.
% Unvisited children always score +Inf and are chosen first.
    leaf_idx = 1;
    while ~isempty(tree.children{leaf_idx})
        children      = tree.children{leaf_idx};
        parent_visits = max(tree.n_visits(leaf_idx), 1);
        ucb_scores    = zeros(1, length(children));

        for ci = 1:length(children)
            c = children(ci);
            if tree.n_visits(c) == 0
                ucb_scores(ci) = inf;
            else
                ucb_scores(ci) = tree.sum_reward(c) / tree.n_visits(c) + ...
                                 C * sqrt(log(parent_visits) / tree.n_visits(c));
            end
        end

        [~, best_ci] = max(ucb_scores);
        leaf_idx = children(best_ci);
    end
end


function tree = mcts_expand(tree, node_idx, full_ranges)
% Bisect node_idx along the dimension with the largest normalized extent.
% Two child nodes are added to the tree and node_idx is no longer a leaf.
    bounds = tree.bounds{node_idx};

    % Normalize current region widths by full parameter-space range
    norm_widths = (bounds(:,2) - bounds(:,1)) ./ full_ranges;
    [~, split_dim] = max(norm_widths);
    split_val = mean(bounds(split_dim, :));

    bounds_L = bounds;  bounds_L(split_dim, 2) = split_val;
    bounds_R = bounds;  bounds_R(split_dim, 1) = split_val;

    n = length(tree.n_visits);
    L = n + 1;
    R = n + 2;

    tree.bounds{L}       = bounds_L;
    tree.bounds{R}       = bounds_R;
    tree.n_visits(L)     = 0;
    tree.n_visits(R)     = 0;
    tree.sum_reward(L)   = 0;
    tree.sum_reward(R)   = 0;
    tree.n_failures(L)   = 0;
    tree.n_failures(R)   = 0;
    tree.best_min_sep(L) = inf;
    tree.best_min_sep(R) = inf;
    tree.parent(L)       = node_idx;
    tree.parent(R)       = node_idx;
    tree.children{L}     = [];
    tree.children{R}     = [];

    tree.children{node_idx} = [L, R];
end


function tree = mcts_backprop(tree, node_idx, total_reward, n_sims, n_failures, best_ms)
% Walk from node_idx to root (parent == 0), updating cumulative statistics.
    idx = node_idx;
    while idx ~= 0
        tree.n_visits(idx)     = tree.n_visits(idx)     + n_sims;
        tree.sum_reward(idx)   = tree.sum_reward(idx)   + total_reward;
        tree.n_failures(idx)   = tree.n_failures(idx)   + n_failures;
        tree.best_min_sep(idx) = min(tree.best_min_sep(idx), best_ms);
        idx = tree.parent(idx);
    end
end


function x0 = make_initial_state(rho, theta, psi)
% Construct a 9-D initial state from ACAS Xu input parameters.
%
% Ownship at origin heading north (pi/2). Intruder placed at bearing theta
% (relative to ownship nose) at range rho, with heading (psi + pi/2).
% By construction: environment(xo, xi) recovers [rho, theta, psi].
    own_hdg = pi / 2;
    int_x   = rho * cos(theta + own_hdg);
    int_y   = rho * sin(theta + own_hdg);
    int_hdg = psi + own_hdg;

    xo = [0; 0; own_hdg];
    xi = [int_x; int_y; int_hdg];
    [rho0, theta0, psi0] = environment(xo, xi);
    x0 = [xo; xi; rho0; theta0; psi0];
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
