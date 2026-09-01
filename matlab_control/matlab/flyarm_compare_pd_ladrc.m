function comparison = flyarm_compare_pd_ladrc(P, scenario, opts)
%FLYARM_COMPARE_PD_LADRC Compatibility wrapper name for PID vs LADRC.

arguments
    P struct
    scenario (1, :) char = "payload_step"
    opts.dt (1, 1) double = 0.01
    opts.tFinal (1, 1) double = 10.0
    opts.outputDir (1, :) char = ""
    opts.show (1, 1) logical = false
end

pidLog = flyarm_trajectory_tracking(P, scenario, dt=opts.dt, tFinal=opts.tFinal, controller="pid");
ladrcLog = flyarm_trajectory_tracking(P, scenario, dt=opts.dt, tFinal=opts.tFinal, controller="ladrc");

comparison = struct();
comparison.scenario = scenario;
comparison.pid = metrics_from_log(pidLog);
comparison.ladrc = metrics_from_log(ladrcLog);
comparison.logs.pid = pidLog;
comparison.logs.ladrc = ladrcLog;

if strlength(string(opts.outputDir)) > 0
    if ~exist(opts.outputDir, "dir")
        mkdir(opts.outputDir);
    end
    save(fullfile(opts.outputDir, "flyarm_pid_vs_ladrc_" + string(scenario) + "_log.mat"), "comparison", "P");
    plot_comparison(pidLog, ladrcLog, comparison, fullfile(opts.outputDir, "flyarm_pid_vs_ladrc_" + string(scenario) + ".png"), opts.show);
end
end

function m = metrics_from_log(log)
t = log.time;
state = log.state;
ref = log.reference;
att = state(:, 7:9);
posErr = state(:, 1:3) - ref(:, 1:3);
attNorm = vecnorm(att, 2, 2);
zErr = state(:, 3) - ref(:, 3);
disturbanceMask = t >= 2.0;
if any(disturbanceMask)
    attWindow = attNorm(disturbanceMask);
    zWindow = zErr(disturbanceMask);
else
    attWindow = attNorm;
    zWindow = zErr;
end
m = struct();
m.maxAttitudeNorm = max(attWindow);
m.rmsAttitudeNorm = sqrt(mean(attWindow .^ 2));
m.maxAbsZError = max(abs(zWindow));
m.rmsPositionError = sqrt(mean(sum(posErr .^ 2, 2)));
m.finalZ = state(end, 3);
m.finalYaw = state(end, 9);
end

function plot_comparison(pidLog, ladrcLog, comparison, outPath, showFigure)
visible = "off";
if showFigure
    visible = "on";
end
fig = figure("Visible", visible, "Color", "w", "Position", [100 100 1350 1150]);
t = pidLog.time;
C = plot_colors();

subplot(4, 1, 1);
plot(t, pidLog.state(:, 3), "Color", C.blue, "LineWidth", 1.25); hold on;
plot(t, ladrcLog.state(:, 3), "-.", "Color", C.red, "LineWidth", 1.25);
plot(t, pidLog.reference(:, 3), "--", "Color", C.black, "LineWidth", 1.1);
grid on; ylabel("z (m)"); xlabel("time (s)");
lgd = legend("PID", "LADRC", "ref", "Location", "best"); style_legend(lgd);
title("PID vs LADRC - " + string(pidLog.scenario), "Interpreter", "none");
apply_time_axis(gca, t);
apply_padded_ylim(gca, [pidLog.state(:, 3); ladrcLog.state(:, 3); pidLog.reference(:, 3)], 0.45, 0.18);
style_axes(gca);

subplot(4, 1, 2);
plot(t, vecnorm(pidLog.state(:, 7:9), 2, 2), "Color", C.blue, "LineWidth", 1.25); hold on;
plot(t, vecnorm(ladrcLog.state(:, 7:9), 2, 2), "-.", "Color", C.red, "LineWidth", 1.25);
grid on; ylabel("attitude norm (rad)"); xlabel("time (s)");
lgd = legend("PID", "LADRC", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_positive_ylim(gca, [vecnorm(pidLog.state(:, 7:9), 2, 2); vecnorm(ladrcLog.state(:, 7:9), 2, 2)], 0.05, 0.18);
style_axes(gca);

subplot(4, 1, 3);
plot(t, pidLog.state(:, 8), "Color", C.blue, "LineWidth", 1.25); hold on;
plot(t, ladrcLog.state(:, 8), "-.", "Color", C.red, "LineWidth", 1.25);
xline(2.0, "--", "payload", "Color", [0.35 0.35 0.35], "LabelVerticalAlignment", "bottom");
grid on; ylabel("pitch (rad)"); xlabel("time (s)");
lgd = legend("PID", "LADRC", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, [pidLog.state(:, 8); ladrcLog.state(:, 8)], 0.12, 0.18);
style_axes(gca);

subplot(4, 1, 4);
plot(t, ladrcLog.z2Inner(:, 2), "Color", C.purple, "LineWidth", 1.25);
grid on; ylabel("LADRC z2 pitch"); xlabel("time (s)");
apply_time_axis(gca, t);
apply_padded_ylim(gca, ladrcLog.z2Inner(:, 2), 0.60, 0.18);
style_axes(gca);

txt = sprintf("max attitude norm after disturbance: PID %.3f rad, LADRC %.3f rad", ...
    comparison.pid.maxAttitudeNorm, comparison.ladrc.maxAttitudeNorm);
annotation(fig, "textbox", [0.11 0.01 0.85 0.04], "String", txt, ...
    "EdgeColor", "none", "Color", "k", "Interpreter", "none");

drawnow;
exportgraphics(fig, outPath, "Resolution", 220);
plot_comparison_singles(pidLog, ladrcLog, comparison, outPath, showFigure);
if ~showFigure
    close(fig);
end
end

function plot_comparison_singles(pidLog, ladrcLog, comparison, outPath, showFigure)
visible = "off";
if showFigure
    visible = "on";
end
t = pidLog.time;
C = plot_colors();
[folder, name] = fileparts(string(outPath));
basePath = fullfile(folder, name);

fig = figure("Visible", visible, "Color", "w", "Position", [100 100 1250 560]);
plot(t, pidLog.state(:, 3), "Color", C.blue, "LineWidth", 1.8); hold on;
plot(t, ladrcLog.state(:, 3), "-.", "Color", C.red, "LineWidth", 1.8);
plot(t, pidLog.reference(:, 3), "--", "Color", C.black, "LineWidth", 1.5);
grid on; xlabel("time (s)"); ylabel("z (m)");
title("PID vs LADRC altitude tracking: body height compared with reference", "Interpreter", "none");
lgd = legend("PID", "LADRC", "reference", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, [pidLog.state(:, 3); ladrcLog.state(:, 3); pidLog.reference(:, 3)], 0.45, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_altitude_tracking.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = figure("Visible", visible, "Color", "w", "Position", [100 100 1250 560]);
plot(t, vecnorm(pidLog.state(:, 7:9), 2, 2), "Color", C.blue, "LineWidth", 1.8); hold on;
plot(t, vecnorm(ladrcLog.state(:, 7:9), 2, 2), "-.", "Color", C.red, "LineWidth", 1.8);
xline(2.0, "--", "payload", "Color", [0.35 0.35 0.35], "LabelVerticalAlignment", "bottom");
grid on; xlabel("time (s)"); ylabel("attitude norm (rad)");
title("PID vs LADRC attitude disturbance: smaller norm means better rejection", "Interpreter", "none");
lgd = legend("PID", "LADRC", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_positive_ylim(gca, [vecnorm(pidLog.state(:, 7:9), 2, 2); vecnorm(ladrcLog.state(:, 7:9), 2, 2)], 0.05, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_attitude_norm.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = figure("Visible", visible, "Color", "w", "Position", [100 100 1250 560]);
plot(t, pidLog.state(:, 8), "Color", C.blue, "LineWidth", 1.8); hold on;
plot(t, ladrcLog.state(:, 8), "-.", "Color", C.red, "LineWidth", 1.8);
xline(2.0, "--", "payload", "Color", [0.35 0.35 0.35], "LabelVerticalAlignment", "bottom");
grid on; xlabel("time (s)"); ylabel("pitch (rad)");
title("PID vs LADRC pitch response after payload disturbance", "Interpreter", "none");
lgd = legend("PID", "LADRC", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, [pidLog.state(:, 8); ladrcLog.state(:, 8)], 0.12, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_pitch_response.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fig = figure("Visible", visible, "Color", "w", "Position", [100 100 1250 560]);
plot(t, ladrcLog.z2Inner(:, 2), "Color", C.purple, "LineWidth", 1.8);
grid on; xlabel("time (s)"); ylabel("LADRC z2 pitch");
title("LADRC ESO internal state: estimated pitch-channel disturbance", "Interpreter", "none");
lgd = legend("LADRC z2 pitch", "Location", "best"); style_legend(lgd);
apply_time_axis(gca, t);
apply_padded_ylim(gca, ladrcLog.z2Inner(:, 2), 0.60, 0.18);
style_axes(gca);
exportgraphics(fig, basePath + "_ladrc_z2_pitch.png", "Resolution", 240);
close_if_hidden(fig, showFigure);

fprintf("Single-figure PID/LADRC plots saved. Max attitude after payload: PID %.3f rad, LADRC %.3f rad.\n", ...
    comparison.pid.maxAttitudeNorm, comparison.ladrc.maxAttitudeNorm);
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
C.purple = [0.4940, 0.1840, 0.5560];
C.black = [0.0000, 0.0000, 0.0000];
end
