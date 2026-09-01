function [y, simState] = flyarm_simulink_step_impl(t, controller, simState)
%FLYARM_SIMULINK_STEP_IMPL One closed-loop step for Simulink function blocks.
%
% Output y is a fixed 25x1 vector:
%   1      time
%   2:13   state x = [p; v; eul; omega]
%   14:17  reference [x y z yaw]
%   18:21  actual wrench [T Mx My Mz]
%   22:25  motor speeds

controller = lower(string(controller));
if isempty(simState) || t == 0 || (isfield(simState, "lastT") && t < simState.lastT)
    simState = init_state(controller);
end

dt = t - simState.lastT;
if dt > 0
    simState = advance_state(t, dt, simState);
end

y = pack_output(t, simState);
end

function simState = init_state(controller)
root = fileparts(fileparts(mfilename("fullpath")));
urdfPath = fullfile(fileparts(root), "urdf", "flyarm_parallel_gripper.urdf");
P = flyarm_params_from_urdf(urdfPath);
[B, mix] = flyarm_motor_mixer(P);

simState = struct();
simState.P = P;
simState.B = B;
simState.mix = mix;
simState.controller = controller;
simState.x = zeros(12, 1);
simState.omegaMotor = zeros(4, 1);
simState.lastT = 0.0;
simState.ref = [0; 0; 0.2; 0];
simState.wrench = zeros(4, 1);
if controller == "ladrc"
    simState.ladrc = flyarm_ladrc_init(P, 0.01);
else
    simState.ladrc = [];
end
if controller == "pid"
    simState.pid = flyarm_pid_init(P, 0.01);
else
    simState.pid = [];
end
end

function simState = advance_state(t, dt, simState)
P = simState.P;
ref = reference_at(t);

if simState.controller == "ladrc"
    simState.ladrc.dt = dt;
    [wrenchCmd, ~, simState.ladrc] = flyarm_cascade_ladrc_control(t, simState.x, ref, P, simState.ladrc);
elseif simState.controller == "pid"
    simState.pid.dt = dt;
    [wrenchCmd, ~, simState.pid] = flyarm_attitude_altitude_pid_control(t, simState.x, ref, P, simState.pid);
else
    [wrenchCmd, ~] = flyarm_attitude_altitude_control(t, simState.x, ref, P);
end

desired = [wrenchCmd.thrust; wrenchCmd.moment(:)];
omegaSqCmd = simState.B \ desired;
omegaSqCmd = min(max(omegaSqCmd, 0), simState.mix.maxOmega ^ 2);
omegaCmd = sqrt(omegaSqCmd);
simState.omegaMotor = flyarm_motor_model(simState.omegaMotor, omegaCmd, dt, P);

actual = simState.B * (simState.omegaMotor .^ 2);
u.thrust = actual(1);
u.moment = actual(2:4);

disturbances = disturbance_at(t);
xdot = flyarm_quad_dynamics(t, simState.x, u, P, disturbances);
simState.x = simState.x + dt * xdot;
simState.x(7:9) = arrayfun(@wrap_pi, simState.x(7:9));

simState.ref = [ref.pos(:); ref.yaw];
simState.wrench = actual;
simState.lastT = t;
end

function ref = reference_at(t)
ref = struct();
ref.yaw = 0.0;
ref.yawRate = 0.0;
[z, zd, zdd] = smooth_transition(t, 0.2, 1.0, 0.5, 2.0);
ref.pos = [0; 0; z];
ref.vel = [0; 0; zd];
ref.acc = [0; 0; zdd];
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

function disturbances = disturbance_at(t)
disturbances = struct();
disturbances.force_b = zeros(3, 1);
disturbances.tau_arm_b = zeros(3, 1);
disturbances.tau_payload_b = zeros(3, 1);
if t >= 2.0
    disturbances.tau_payload_b = [0.0; 0.025; 0.0];
end
end

function y = pack_output(t, simState)
y = zeros(25, 1);
y(1) = t;
y(2:13) = simState.x;
y(14:17) = simState.ref;
y(18:21) = simState.wrench;
y(22:25) = simState.omegaMotor;
end

function y = wrap_pi(x)
y = mod(x + pi, 2 * pi) - pi;
end
