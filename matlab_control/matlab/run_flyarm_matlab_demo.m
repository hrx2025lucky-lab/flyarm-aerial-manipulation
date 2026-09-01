function run_flyarm_matlab_demo(opts)
%RUN_FLYARM_MATLAB_DEMO Generate first-version flyarm control plots.
%
% Usage:
%   run_flyarm_matlab_demo
%   run_flyarm_matlab_demo(show=true)
%
% The default mode is paper-output mode: figures are saved and closed.
% Set show=true when tuning parameters in MATLAB and you want visible figures.

arguments
    opts.show (1, 1) logical = false
end

root = fileparts(fileparts(mfilename("fullpath")));
urdfPath = fullfile(fileparts(root), "urdf", "flyarm_parallel_gripper.urdf");
outDir = fullfile(root, "output");
if ~exist(outDir, "dir")
    mkdir(outDir);
end

P = flyarm_params_from_urdf(urdfPath);
P = flyarm_articulation_config(P);
scenarios = ["hover_step", "payload_step", "circle"];
for i = 1:numel(scenarios)
    log = flyarm_trajectory_tracking(P, char(scenarios(i)));
    save(fullfile(outDir, "flyarm_" + scenarios(i) + "_log.mat"), "log", "P");
    plot_scenario(log, fullfile(outDir, "flyarm_" + scenarios(i) + ".png"), opts.show);
end

coupledLog = flyarm_coupled_tracking(P, "arm_reach_payload", controller="ladrc");
save(fullfile(outDir, "flyarm_coupled_arm_payload_log.mat"), "coupledLog", "P");
plot_coupled(coupledLog, fullfile(outDir, "flyarm_coupled_arm_payload.png"), opts.show);

comparison = flyarm_compare_pid_ladrc(P, "payload_step", outputDir=outDir, show=opts.show);
fprintf("PID max attitude after payload: %.4f rad; LADRC: %.4f rad.\n", ...
    comparison.pid.maxAttitudeNorm, comparison.ladrc.maxAttitudeNorm);

fprintf("Generated flyarm MATLAB demo outputs in %s\n", outDir);
if opts.show
    fprintf("Visible figures are open for tuning. Close them manually when finished.\n");
end
end

function plot_scenario(log, outPath, showFigure)
fig = make_figure(showFigure, "w", [100 100 1300 920]);
t = log.time;
C = plot_colors();

subplot(3, 1, 1);
plot(t, log.state(:, 3), "Color", C.blue, "LineWidth", 1.4); hold on;
plot(t, log.reference(:, 3), "--", "Color", C.black, "LineWidth", 1.2);
grid on; ylabel("z (m)"); xlabel("time (s)");
lgd = legend("actual", "ref", "Location", "best");
style_legend(lgd);
title("Flyarm " + string(log.scenario) + " tracking", "Interpreter", "none");
apply_time_axis(gca, t);
apply_padded_ylim(gca, [log.state(:, 3); log.reference(:, 3)], 0.35, 0.18);
style_axes(gca);

subplot(3, 1, 2);
plot(t, log.state(:, 7), "Color", C.blue, "LineWidth", 1.2); hold on;
plot(t, log.state(:, 8), "Color", C.orange, "LineWidth", 1.2);
plot(t, log.state(:, 9), "Color", C.purple, "LineWidth", 1.2);
grid on; ylabel("attitude (rad)"); xlabel("time (s)");
lgd = legend("roll", "pitch", "yaw", "Location", "best");
style_legend(lgd);
if max(abs(log.state(:, 7:9)), [], "all") < 1e-6
    ylim([-1e-3, 1e-3]);
else
    apply_padded_ylim(gca, log.state(:, 7:9), 0.10, 0.18);
end
apply_time_axis(gca, t);
style_axes(gca);

subplot(3, 1, 3);
plot(t, log.wrench(:, 1), "Color", C.blue, "LineWidth", 1.2);
grid on; ylabel("thrust (N)"); xlabel("time (s)");
apply_time_axis(gca, t);
apply_padded_ylim(gca, log.wrench(:, 1), 8.0, 0.16);
style_axes(gca);

drawnow;
exportgraphics(fig, outPath, "Resolution", 220);
plot_scenario_singles(log, outPath, showFigure);
close_if_hidden(fig, showFigure);
end

function plot_coupled(log, outPath, showFigure)
fig = make_figure(showFigure, "w", [100 100 1350 1200]);
t = log.time;
C = plot_colors();
[tauArmPlot, tauPayloadPlot] = coupling_plot_signals(t, log);

subplot(4, 1, 1);
plot(t, log.state(:, 3), "Color", C.blue, "LineWidth", 1.4); hold on;
plot(t, log.reference(:, 3), "--", "Color", C.black, "LineWidth", 1.2);
grid on; ylabel("z (m)"); xlabel("time (s)");
lgd = legend("body", "ref", "Location", "best");
style_legend(lgd);
title("Flyarm coupled arm/gripper/payload simulation", "Interpreter", "none");
apply_time_axis(gca, t);
apply_padded_ylim(gca, [log.state(:, 3); log.reference(:, 3)], 0.45, 0.18);
style_axes(gca);

subplot(4, 1, 2);
plot(t, log.jointPos(:, 1), "Color", C.blue, "LineWidth", 1.1); hold on;
plot(t, log.jointPos(:, 2), "Color", C.orange, "LineWidth", 1.1);
plot(t, log.jointPos(:, 3), "Color", C.purple, "LineWidth", 1.1);
plot(t, log.jointPos(:, 4), "Color", C.green, "LineWidth", 1.1);
grid on; ylabel("arm q (rad)"); xlabel("time (s)");
lgd = legend("base", "B", "C", "D", "Location", "best");
style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, log.jointPos(:, 1:4), 2.20, 0.12);
style_axes(gca);

subplot(4, 1, 3);
plot(t, tauArmPlot(:, 1), "Color", C.blue, "LineWidth", 1.0); hold on;
plot(t, tauArmPlot(:, 2), "Color", C.orange, "LineWidth", 1.0);
plot(t, tauArmPlot(:, 3), "Color", C.purple, "LineWidth", 1.0);
plot(t, tauPayloadPlot(:, 1), "--", "Color", C.magenta, "LineWidth", 1.1);
plot(t, tauPayloadPlot(:, 2), "--", "Color", C.green, "LineWidth", 1.1);
plot(t, tauPayloadPlot(:, 3), "--", "Color", C.cyan, "LineWidth", 1.1);
grid on; ylabel("coupling torque (Nm)"); xlabel("time (s)");
lgd = legend("arm x", "arm y", "arm z", "load x", "load y", "load z", "Location", "best");
style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, [tauArmPlot; tauPayloadPlot], 0.08, 0.18);
style_axes(gca);

subplot(4, 1, 4);
plot(t, vecnorm(log.state(:, 7:9), 2, 2), "Color", C.blue, "LineWidth", 1.2);
grid on; ylabel("|att| (rad)"); xlabel("time (s)");
apply_time_axis(gca, t);
apply_positive_ylim(gca, vecnorm(log.state(:, 7:9), 2, 2), 0.05, 0.18);
style_axes(gca);

drawnow;
exportgraphics(fig, outPath, "Resolution", 220);
plot_coupled_singles(log, outPath, showFigure);
close_if_hidden(fig, showFigure);
end

function plot_scenario_singles(log, outPath, showFigure)
t = log.time;
C = plot_colors();
[folder, name] = fileparts(string(outPath));
basePath = fullfile(folder, name);

fig = make_figure(showFigure, "w", [100 100 1250 560]);
plot(t, log.state(:, 3), "Color", C.blue, "LineWidth", 1.8); hold on;
plot(t, log.reference(:, 3), "--", "Color", C.black, "LineWidth", 1.5);
grid on; xlabel("time (s)"); ylabel("z (m)");
title("Altitude tracking: body height follows the reference", "Interpreter", "none");
lgd = legend("actual body z", "reference z", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, [log.state(:, 3); log.reference(:, 3)], 0.35, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_altitude_tracking.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = make_figure(showFigure, "w", [100 100 1250 560]);
plot(t, log.state(:, 7), "Color", C.blue, "LineWidth", 1.6); hold on;
plot(t, log.state(:, 8), "Color", C.orange, "LineWidth", 1.6);
plot(t, log.state(:, 9), "Color", C.purple, "LineWidth", 1.6);
grid on; xlabel("time (s)"); ylabel("attitude (rad)");
title("Attitude response: roll, pitch, and yaw", "Interpreter", "none");
lgd = legend("roll", "pitch", "yaw", "Location", "best"); style_legend(lgd);
if max(abs(log.state(:, 7:9)), [], "all") < 1e-6
    ylim([-1e-3, 1e-3]);
else
    apply_padded_ylim(gca, log.state(:, 7:9), 0.10, 0.18);
end
apply_time_axis(gca, t);
style_axes(gca);
exportgraphics(fig, basePath + "_attitude_response.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = make_figure(showFigure, "w", [100 100 1250 560]);
plot(t, log.wrench(:, 1), "Color", C.blue, "LineWidth", 1.8);
grid on; xlabel("time (s)"); ylabel("thrust (N)");
title("Total thrust command: force generated by the rotor group", "Interpreter", "none");
lgd = legend("total thrust", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, log.wrench(:, 1), 8.0, 0.16);
style_axes(gca);
exportgraphics(fig, basePath + "_thrust_command.png", "Resolution", 240);
close_if_hidden(fig, showFigure);
end

function plot_coupled_singles(log, outPath, showFigure)
t = log.time;
C = plot_colors();
[folder, name] = fileparts(string(outPath));
basePath = fullfile(folder, name);
[tauArmPlot, tauPayloadPlot] = coupling_plot_signals(t, log);

fig = make_figure(showFigure, "w", [100 100 1250 560]);
plot(t, log.state(:, 3), "Color", C.blue, "LineWidth", 1.8); hold on;
plot(t, log.reference(:, 3), "--", "Color", C.black, "LineWidth", 1.5);
grid on; xlabel("time (s)"); ylabel("z (m)");
title("Coupled altitude tracking: UAV body height during arm/payload motion", "Interpreter", "none");
lgd = legend("body z", "reference z", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, [log.state(:, 3); log.reference(:, 3)], 0.45, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_altitude_tracking.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = make_figure(showFigure, "w", [100 100 1250 560]);
plot(t, log.jointPos(:, 1), "Color", C.blue, "LineWidth", 1.6); hold on;
plot(t, log.jointPos(:, 2), "Color", C.orange, "LineWidth", 1.6);
plot(t, log.jointPos(:, 3), "Color", C.purple, "LineWidth", 1.6);
plot(t, log.jointPos(:, 4), "Color", C.green, "LineWidth", 1.6);
grid on; xlabel("time (s)"); ylabel("arm q (rad)");
title("Arm joint motion: base, B, C, and D joint positions", "Interpreter", "none");
lgd = legend("base", "B", "C", "D", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, log.jointPos(:, 1:4), 2.20, 0.12);
style_axes(gca);
exportgraphics(fig, basePath + "_arm_joint_motion.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = make_figure(showFigure, "w", [100 100 1250 560]);
plot(t, tauArmPlot(:, 1), "Color", C.blue, "LineWidth", 1.4); hold on;
plot(t, tauArmPlot(:, 2), "Color", C.orange, "LineWidth", 1.6);
plot(t, tauArmPlot(:, 3), "Color", C.purple, "LineWidth", 1.6);
grid on; xlabel("time (s)"); ylabel("coupling torque (Nm)");
title("Arm-x dominant coupling torque: filtered arm reaction torque", "Interpreter", "none");
lgd = legend("arm x", "arm y", "arm z", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, tauArmPlot(:, 1:3), 0.02, 0.22);
style_axes(gca);
exportgraphics(fig, basePath + "_arm_coupling_torque.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = make_figure(showFigure, "w", [100 100 1250 560]);
plot(t, tauPayloadPlot(:, 1), "Color", C.magenta, "LineWidth", 1.8); hold on;
plot(t, tauPayloadPlot(:, 2), "Color", C.green, "LineWidth", 1.8);
plot(t, tauPayloadPlot(:, 3), "Color", C.cyan, "LineWidth", 1.8);
grid on; xlabel("time (s)"); ylabel("coupling torque (Nm)");
title("Payload-induced coupling torque: filtered attached-load offset torque", "Interpreter", "none");
lgd = legend("load x", "load y", "load z", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, tauPayloadPlot(:, 1:3), 0.08, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_payload_coupling_torque.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = make_figure(showFigure, "w", [100 100 1250 560]);
plot(t, tauArmPlot(:, 2), "Color", C.orange, "LineWidth", 1.8); hold on;
plot(t, tauArmPlot(:, 3), "Color", C.purple, "LineWidth", 1.8);
plot(t, tauPayloadPlot(:, 1), "--", "Color", C.magenta, "LineWidth", 1.8);
plot(t, tauPayloadPlot(:, 2), "--", "Color", C.green, "LineWidth", 1.8);
plot(t, tauPayloadPlot(:, 3), "--", "Color", C.cyan, "LineWidth", 1.8);
grid on; xlabel("time (s)"); ylabel("coupling torque (Nm)");
title("Minor coupling torques shown separately: filtered view", "Interpreter", "none");
lgd = legend("arm y", "arm z", "load x", "load y", "load z", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, [tauArmPlot(:, 2:3), tauPayloadPlot(:, 1:3)], 0.08, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_minor_coupling_torques.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = make_figure(showFigure, "w", [100 100 1250 560]);
plot(t, vecnorm(log.state(:, 7:9), 2, 2), "Color", C.blue, "LineWidth", 1.8);
grid on; xlabel("time (s)"); ylabel("|att| (rad)");
title("Attitude disturbance norm: body attitude error caused by coupling", "Interpreter", "none");
lgd = legend("attitude norm", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_positive_ylim(gca, vecnorm(log.state(:, 7:9), 2, 2), 0.05, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_attitude_norm.png", "Resolution", 240);
close_if_hidden(fig, showFigure);
end

function style_axes(ax)
set(ax, "Color", "w", "XColor", "k", "YColor", "k", ...
    "GridColor", [0.75 0.75 0.75], "FontSize", 12, "LineWidth", 1.0);
title(ax, ax.Title.String, "Color", "k");
end

function apply_time_axis(ax, t)
t = t(:);
t = t(isfinite(t));
if isempty(t)
    return;
end
t0 = min(t);
t1 = max(t);
xlim(ax, [t0, t1]);
if t1 - t0 <= 12
    ticks = ceil(t0):1:floor(t1);
    if ~isempty(ticks)
        xticks(ax, ticks);
    end
end
end

function apply_padded_ylim(ax, values, minSpan, padFrac)
values = values(:);
values = values(isfinite(values));
if isempty(values)
    return;
end
yMin = min(values);
yMax = max(values);
span = max(yMax - yMin, eps);
targetSpan = max(span * (1 + 2 * padFrac), minSpan);
center = 0.5 * (yMin + yMax);
ylim(ax, nice_limits(center - 0.5 * targetSpan, center + 0.5 * targetSpan));
end

function apply_positive_ylim(ax, values, minTop, padFrac)
values = values(:);
values = values(isfinite(values));
if isempty(values)
    return;
end
top = max(max(values) * (1 + padFrac), minTop);
lim = nice_limits(0, top);
lim(1) = 0;
ylim(ax, lim);
end

function lim = nice_limits(yMin, yMax)
span = max(yMax - yMin, eps);
rawStep = 10 ^ floor(log10(span / 6));
steps = [1, 2, 5, 10] * rawStep;
step = steps(find(steps >= span / 6, 1, "first"));
if isempty(step)
    step = rawStep;
end
lim = [floor(yMin / step) * step, ceil(yMax / step) * step];
if lim(1) == lim(2)
    lim = lim + [-step, step];
end
end

function [tauArmPlot, tauPayloadPlot] = coupling_plot_signals(t, log)
tauFilterSeconds = 0.20;
tauArmPlot = moving_average_series(t, first_order_lowpass(t, log.coupling.tauArm, tauFilterSeconds), 0.45);
tauPayloadPlot = moving_average_series(t, first_order_lowpass(t, log.coupling.tauPayload, tauFilterSeconds), 0.45);
end

function y = first_order_lowpass(t, x, timeConstant)
y = x;
if numel(t) < 2 || timeConstant <= 0
    return;
end
for k = 2:size(x, 1)
    dt = max(eps, t(k) - t(k - 1));
    alpha = dt / (timeConstant + dt);
    y(k, :) = y(k - 1, :) + alpha * (x(k, :) - y(k - 1, :));
end
end

function y = moving_average_series(t, x, windowSeconds)
y = x;
if numel(t) < 2 || windowSeconds <= 0
    return;
end
dt = median(diff(t));
halfWindow = max(1, round(0.5 * windowSeconds / dt));
for k = 1:size(x, 1)
    i0 = max(1, k - halfWindow);
    i1 = min(size(x, 1), k + halfWindow);
    y(k, :) = mean(x(i0:i1, :), 1);
end
end

function style_legend(lgd)
set(lgd, "Color", "w", "TextColor", "k", "EdgeColor", [0.4 0.4 0.4]);
end

function fig = make_figure(showFigure, bgColor, position)
if nargin < 3
    position = [];
end
visible = "off";
if showFigure
    visible = "on";
end
if isempty(position)
    fig = figure("Visible", visible, "Color", bgColor);
else
    fig = figure("Visible", visible, "Color", bgColor, "Position", position);
end
end

function close_if_hidden(fig, showFigure)
if ~showFigure
    close(fig);
end
end

function C = plot_colors()
C = struct();
C.blue = [0.0000, 0.4470, 0.7410];
C.orange = [0.8500, 0.3250, 0.0980];
C.purple = [0.4940, 0.1840, 0.5560];
C.green = [0.4660, 0.6740, 0.1880];
C.magenta = [0.6350, 0.0780, 0.1840];
C.cyan = [0.3010, 0.7450, 0.9330];
C.black = [0.0000, 0.0000, 0.0000];
end
