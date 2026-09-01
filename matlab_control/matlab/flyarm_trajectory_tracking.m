function log = flyarm_trajectory_tracking(P, scenario, opts)
%FLYARM_TRAJECTORY_TRACKING Simulate first-version trajectory tracking.
%
% scenario:
%   "hover_step"    - takeoff/height step with motor lag
%   "payload_step"  - same, plus a payload torque disturbance after 2 s
%   "circle"        - horizontal circular reference at fixed altitude
% controller:
%   "pid"           - baseline position/attitude PID
%   "pd"            - legacy position/attitude PD
%   "ladrc"         - cascade LADRC attitude/rate loops

arguments
    P struct
    scenario (1, :) char = "hover_step"
    opts.dt (1, 1) double = 0.01
    opts.tFinal (1, 1) double = 10.0
    opts.controller (1, :) char = "pid"
end

[B, mix] = flyarm_motor_mixer(P);
dt = opts.dt;
time = (0:dt:opts.tFinal).';
n = numel(time);

x = zeros(12, 1);
x(3) = 0.0;
omegaMotor = zeros(4, 1);

state = zeros(n, 12);
refLog = zeros(n, 4);
wrenchLog = zeros(n, 4);
omegaLog = zeros(n, 4);
z2InnerLog = zeros(n, 3);

controller = lower(string(opts.controller));
if controller == "ladrc"
    controllerState = flyarm_ladrc_init(P, dt);
elseif controller == "pid"
    controllerState = flyarm_pid_init(P, dt);
elseif controller == "pd"
    controllerState = [];
else
    error("flyarm_trajectory_tracking:UnknownController", "Unknown controller '%s'.", opts.controller);
end

for k = 1:n
    t = time(k);
    ref = reference_at(t, scenario);
    if controller == "ladrc"
        [wrenchCmd, aux, controllerState] = flyarm_cascade_ladrc_control(t, x, ref, P, controllerState);
        z2InnerLog(k, :) = aux.z2Inner(:).';
    elseif controller == "pid"
        controllerState.dt = dt;
        [wrenchCmd, ~, controllerState] = flyarm_attitude_altitude_pid_control(t, x, ref, P, controllerState);
    else
        [wrenchCmd, ~] = flyarm_attitude_altitude_control(t, x, ref, P);
    end

    desired = [wrenchCmd.thrust; wrenchCmd.moment(:)];
    omegaSqCmd = B \ desired;
    omegaSqCmd = min(max(omegaSqCmd, 0), mix.maxOmega ^ 2);
    omegaCmd = sqrt(omegaSqCmd);
    omegaMotor = flyarm_motor_model(omegaMotor, omegaCmd, dt, P);

    actual = B * (omegaMotor .^ 2);
    u.thrust = actual(1);
    u.moment = actual(2:4);

    disturbances = disturbance_at(t, scenario);
    xdot = flyarm_quad_dynamics(t, x, u, P, disturbances);
    x = x + dt * xdot;
    x(7:9) = arrayfun(@wrap_pi, x(7:9));

    state(k, :) = x.';
    refLog(k, :) = [ref.pos(:).', ref.yaw];
    wrenchLog(k, :) = actual.';
    omegaLog(k, :) = omegaMotor.';
end

log = struct();
log.time = time;
log.state = state;
log.reference = refLog;
log.wrench = wrenchLog;
log.motorOmega = omegaLog;
log.scenario = scenario;
log.controller = char(controller);
log.z2Inner = z2InnerLog;
end

function ref = reference_at(t, scenario)
ref = struct();
ref.yaw = 0.0;
ref.yawRate = 0.0;
switch string(scenario)
    case {"hover_step", "payload_step"}
        [z, zd, zdd] = smooth_transition(t, 0.2, 1.0, 0.5, 2.0);
        ref.pos = [0; 0; z];
        ref.vel = [0; 0; zd];
        ref.acc = [0; 0; zdd];
    case "circle"
        radius = 0.4;
        w = 0.8;
        ref.pos = [radius * cos(w * t); radius * sin(w * t); 1.0];
        ref.vel = [-radius * w * sin(w * t); radius * w * cos(w * t); 0];
        ref.acc = [-radius * w^2 * cos(w * t); -radius * w^2 * sin(w * t); 0];
        ref.yaw = w * t;
        ref.yawRate = w;
    otherwise
        error("flyarm_trajectory_tracking:UnknownScenario", "Unknown scenario '%s'.", scenario);
end
end

function disturbances = disturbance_at(t, scenario)
disturbances = struct();
disturbances.force_b = zeros(3, 1);
disturbances.tau_arm_b = zeros(3, 1);
disturbances.tau_payload_b = zeros(3, 1);
if string(scenario) == "payload_step" && t >= 2.0
    % Approximate the load/gripper offset torque: a small forward arm creates
    % a pitch disturbance when holding a light object.
    disturbances.tau_payload_b = [0.0; 0.025; 0.0];
end
end

function y = wrap_pi(x)
y = mod(x + pi, 2 * pi) - pi;
end

function [y, yd, ydd] = smooth_transition(t, y0, y1, t0, duration)
if t <= t0
    y = y0;
    yd = 0.0;
    ydd = 0.0;
elseif t >= t0 + duration
    y = y1;
    yd = 0.0;
    ydd = 0.0;
else
    r = (t - t0) / duration;
    s = 10 * r^3 - 15 * r^4 + 6 * r^5;
    sd = (30 * r^2 - 60 * r^3 + 30 * r^4) / duration;
    sdd = (60 * r - 180 * r^2 + 120 * r^3) / (duration^2);
    dy = y1 - y0;
    y = y0 + dy * s;
    yd = dy * sd;
    ydd = dy * sdd;
end
end
