function result = run_flyarm_simulink_closed_loop(opts)
%RUN_FLYARM_SIMULINK_CLOSED_LOOP Build and run the Simulink PID/LADRC model.
%
% Usage:
%   run_flyarm_simulink_closed_loop
%   run_flyarm_simulink_closed_loop(show=true)
%   run_flyarm_simulink_closed_loop(show=true, openModel=true)
%
% openModel=true leaves the runnable top-level Simulink model open. The
% current model uses interpreted MATLAB function blocks so the top level is
% compact; the detailed controller/mixer/motor/plant logic is inside MATLAB
% functions called by those blocks.

arguments
    opts.show (1, 1) logical = false
    opts.openModel (1, 1) logical = false
end

root = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(root, "matlab"));
addpath(fullfile(root, "simulink"));
outDir = fullfile(root, "output");
if ~exist(outDir, "dir")
    mkdir(outDir);
end

modelPath = build_flyarm_closed_loop_model();
[~, modelName] = fileparts(modelPath);
load_system(modelName);
if opts.openModel
    open_system(modelName);
end
simOut = sim(modelName, "StopTime", "10.0");
if ~opts.openModel
    close_system(modelName, 0);
end

pidRaw = simOut.get("pid_out");
ladrcRaw = simOut.get("ladrc_out");
pidLog = parse_raw(pidRaw, "pid");
ladrcLog = parse_raw(ladrcRaw, "ladrc");

result = struct();
result.modelPath = modelPath;
result.pid = metrics_from_log(pidLog);
result.ladrc = metrics_from_log(ladrcLog);
result.logs.pid = pidLog;
result.logs.ladrc = ladrcLog;

save(fullfile(outDir, "flyarm_simulink_closed_loop_log.mat"), "result");
plot_simulink_result(result, fullfile(outDir, "flyarm_simulink_closed_loop_compare.png"), opts.show);
fprintf("Simulink closed-loop comparison: PID %.4f rad, LADRC %.4f rad.\n", ...
    result.pid.maxAttitudeNorm, result.ladrc.maxAttitudeNorm);
if opts.show
    fprintf("Visible Simulink result figure is open for tuning. Close it manually when finished.\n");
end
end

function log = parse_raw(raw, controller)
if isa(raw, "timeseries")
    data = raw.Data;
    time = raw.Time;
else
    data = raw;
    time = [];
end
if ndims(data) == 3
    data = squeeze(data);
end
if size(data, 2) ~= 25 && size(data, 1) == 25
    data = data.';
end
log = struct();
log.controller = controller;
log.time = data(:, 1);
if ~isempty(time) && numel(time) == size(data, 1)
    log.time = time(:);
end
log.state = data(:, 2:13);
log.reference = data(:, 14:17);
log.wrench = data(:, 18:21);
log.motorOmega = data(:, 22:25);
end

function m = metrics_from_log(log)
t = log.time;
attNorm = vecnorm(log.state(:, 7:9), 2, 2);
zErr = log.state(:, 3) - log.reference(:, 3);
mask = t >= 2.0;
if ~any(mask)
    mask = true(size(t));
end
m = struct();
m.maxAttitudeNorm = max(attNorm(mask));
m.rmsAttitudeNorm = sqrt(mean(attNorm(mask) .^ 2));
m.maxAbsZError = max(abs(zErr(mask)));
end

function plot_simulink_result(result, outPath, showFigure)
pid = result.logs.pid;
ladrc = result.logs.ladrc;
t = pid.time;
visible = "off";
if showFigure
    visible = "on";
end
fig = figure("Visible", visible, "Color", "w", "Position", [100 100 1350 980]);
C = plot_colors();

subplot(3, 1, 1);
plot(t, pid.state(:, 3), "Color", C.blue, "LineWidth", 1.25); hold on;
plot(t, ladrc.state(:, 3), "-.", "Color", C.red, "LineWidth", 1.25);
plot(t, pid.reference(:, 3), "--", "Color", C.black, "LineWidth", 1.1);
grid on; ylabel("z (m)"); xlabel("time (s)");
lgd = legend("PID", "LADRC", "ref", "Location", "best"); style_legend(lgd);
title("Simulink closed-loop PID vs LADRC", "Interpreter", "none");
apply_time_axis(gca, t);
apply_padded_ylim(gca, [pid.state(:, 3); ladrc.state(:, 3); pid.reference(:, 3)], 0.45, 0.18);
style_axes(gca);

subplot(3, 1, 2);
plot(t, vecnorm(pid.state(:, 7:9), 2, 2), "Color", C.blue, "LineWidth", 1.25); hold on;
plot(t, vecnorm(ladrc.state(:, 7:9), 2, 2), "-.", "Color", C.red, "LineWidth", 1.25);
xline(2.0, "--", "payload", "Color", [0.35 0.35 0.35]);
grid on; ylabel("attitude norm (rad)"); xlabel("time (s)");
lgd = legend("PID", "LADRC", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_positive_ylim(gca, [vecnorm(pid.state(:, 7:9), 2, 2); vecnorm(ladrc.state(:, 7:9), 2, 2)], 0.05, 0.18);
style_axes(gca);

subplot(3, 1, 3);
plot(t, pid.state(:, 8), "Color", C.blue, "LineWidth", 1.25); hold on;
plot(t, ladrc.state(:, 8), "-.", "Color", C.red, "LineWidth", 1.25);
grid on; ylabel("pitch (rad)"); xlabel("time (s)");
lgd = legend("PID", "LADRC", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, [pid.state(:, 8); ladrc.state(:, 8)], 0.12, 0.18);
style_axes(gca);

txt = sprintf("Simulink max attitude after disturbance: PID %.3f rad, LADRC %.3f rad", ...
    result.pid.maxAttitudeNorm, result.ladrc.maxAttitudeNorm);
annotation(fig, "textbox", [0.11 0.01 0.85 0.04], "String", txt, ...
    "EdgeColor", "none", "Color", "k", "Interpreter", "none");

drawnow;
exportgraphics(fig, outPath, "Resolution", 220);
plot_simulink_singles(result, outPath, showFigure);
if ~showFigure
    close(fig);
end
end

function plot_simulink_singles(result, outPath, showFigure)
pid = result.logs.pid;
ladrc = result.logs.ladrc;
t = pid.time;
visible = "off";
if showFigure
    visible = "on";
end
C = plot_colors();
[folder, name] = fileparts(string(outPath));
basePath = fullfile(folder, name);

fig = figure("Visible", visible, "Color", "w", "Position", [100 100 1250 560]);
plot(t, pid.state(:, 3), "Color", C.blue, "LineWidth", 1.8); hold on;
plot(t, ladrc.state(:, 3), "-.", "Color", C.red, "LineWidth", 1.8);
plot(t, pid.reference(:, 3), "--", "Color", C.black, "LineWidth", 1.5);
grid on; xlabel("time (s)"); ylabel("z (m)");
title("Simulink altitude tracking: executable PID/LADRC closed-loop output", "Interpreter", "none");
lgd = legend("PID", "LADRC", "reference", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, [pid.state(:, 3); ladrc.state(:, 3); pid.reference(:, 3)], 0.45, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_altitude_tracking.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = figure("Visible", visible, "Color", "w", "Position", [100 100 1250 560]);
plot(t, vecnorm(pid.state(:, 7:9), 2, 2), "Color", C.blue, "LineWidth", 1.8); hold on;
plot(t, vecnorm(ladrc.state(:, 7:9), 2, 2), "-.", "Color", C.red, "LineWidth", 1.8);
xline(2.0, "--", "payload", "Color", [0.35 0.35 0.35], "LabelVerticalAlignment", "bottom");
grid on; xlabel("time (s)"); ylabel("attitude norm (rad)");
title("Simulink attitude disturbance: PID vs LADRC after payload attachment", "Interpreter", "none");
lgd = legend("PID", "LADRC", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_positive_ylim(gca, [vecnorm(pid.state(:, 7:9), 2, 2); vecnorm(ladrc.state(:, 7:9), 2, 2)], 0.05, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_attitude_norm.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = figure("Visible", visible, "Color", "w", "Position", [100 100 1250 560]);
plot(t, pid.state(:, 8), "Color", C.blue, "LineWidth", 1.8); hold on;
plot(t, ladrc.state(:, 8), "-.", "Color", C.red, "LineWidth", 1.8);
xline(2.0, "--", "payload", "Color", [0.35 0.35 0.35], "LabelVerticalAlignment", "bottom");
grid on; xlabel("time (s)"); ylabel("pitch (rad)");
title("Simulink pitch response: LADRC suppresses payload-induced pitch", "Interpreter", "none");
lgd = legend("PID", "LADRC", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, [pid.state(:, 8); ladrc.state(:, 8)], 0.12, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_pitch_response.png", "Resolution", 240);
close_if_hidden(fig, showFigure);
end

function close_if_hidden(fig, showFigure)
if ~showFigure
    close(fig);
end
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

function style_legend(lgd)
set(lgd, "Color", "w", "TextColor", "k", "EdgeColor", [0.4 0.4 0.4]);
end

function C = plot_colors()
C = struct();
C.blue = [0.0000, 0.4470, 0.7410];
C.red = [0.8500, 0.1000, 0.1000];
C.black = [0.0000, 0.0000, 0.0000];
end
