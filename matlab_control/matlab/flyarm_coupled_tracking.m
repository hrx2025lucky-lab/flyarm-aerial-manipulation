function log = flyarm_coupled_tracking(P, scenario, opts)
%FLYARM_COUPLED_TRACKING Simulate quadrotor flight with arm/gripper PD drives.
%
% scenario:
%   "arm_reach_payload" - stow -> open -> straight reach -> close + payload.
%
% controller:
%   "pid", "pd", or "ladrc" selects the body flight controller.
%   Arm and gripper joints are always controlled by joint-space PD drives.

arguments
    P struct
    scenario (1, :) char = "arm_reach_payload"
    opts.dt (1, 1) double = 0.01
    opts.tFinal (1, 1) double = 10.0
    opts.controller (1, :) char = "pid"
end

if ~isfield(P, "articulation")
    P = flyarm_articulation_config(P);
end

[B, mix] = flyarm_motor_mixer(P);
dt = opts.dt;
time = (0:dt:opts.tFinal).';
n = numel(time);

xc = zeros(24, 1);
xc(13:18) = P.articulation.qStow(:);
omegaMotor = zeros(4, 1);
jointTauActual = zeros(6, 1);

bodyState = zeros(n, 12);
jointPos = zeros(n, 6);
jointVel = zeros(n, 6);
jointTarget = zeros(n, 6);
jointTau = zeros(n, 6);
refLog = zeros(n, 4);
wrenchLog = zeros(n, 4);
omegaLog = zeros(n, 4);
tauArmLog = zeros(n, 3);
tauPayloadLog = zeros(n, 3);
armComLog = zeros(n, 3);
gripCenterLog = zeros(n, 3);

controller = lower(string(opts.controller));
if controller == "ladrc"
    controllerState = flyarm_ladrc_init(P, dt);
elseif controller == "pid"
    controllerState = flyarm_pid_init(P, dt);
elseif controller == "pd"
    controllerState = [];
else
    error("flyarm_coupled_tracking:UnknownController", "Unknown controller '%s'.", opts.controller);
end

for k = 1:n
    t = time(k);
    xBody = xc(1:12);
    ref = reference_at(t, scenario);
    qCmd = arm_reference_at(t, P, scenario);
    dist = disturbance_at(t, scenario, P);
    dist.jointTauActual = jointTauActual;

    if controller == "ladrc"
        [wrenchCmd, ~, controllerState] = flyarm_cascade_ladrc_control(t, xBody, ref, P, controllerState);
    elseif controller == "pid"
        controllerState.dt = dt;
        [wrenchCmd, ~, controllerState] = flyarm_attitude_altitude_pid_control(t, xBody, ref, P, controllerState);
    else
        [wrenchCmd, ~] = flyarm_attitude_altitude_control(t, xBody, ref, P);
    end

    desired = [wrenchCmd.thrust; wrenchCmd.moment(:)];
    omegaSqCmd = B \ desired;
    omegaSqCmd = min(max(omegaSqCmd, 0), mix.maxOmega ^ 2);
    omegaCmd = sqrt(omegaSqCmd);
    omegaMotor = flyarm_motor_model(omegaMotor, omegaCmd, dt, P);

    actual = B * (omegaMotor .^ 2);
    u.thrust = actual(1);
    u.moment = actual(2:4);

    [xdot, aux] = flyarm_coupled_dynamics(t, xc, u, qCmd, P, dist);
    jointTauActual = jointTauActual + dt * aux.jointTauDot;
    jointTauActual = min(max(jointTauActual, -P.articulation.effort(:)), P.articulation.effort(:));
    xc = xc + dt * xdot;
    xc(7:9) = arrayfun(@wrap_pi, xc(7:9));
    xc(13:18) = min(max(xc(13:18), P.articulation.qLower), P.articulation.qUpper);
    xc(19:24) = min(max(xc(19:24), -P.articulation.velocity), P.articulation.velocity);

    bodyState(k, :) = xc(1:12).';
    jointPos(k, :) = xc(13:18).';
    jointVel(k, :) = xc(19:24).';
    jointTarget(k, :) = qCmd(:).';
    jointTau(k, :) = aux.jointTau(:).';
    refLog(k, :) = [ref.pos(:).', ref.yaw];
    wrenchLog(k, :) = actual.';
    omegaLog(k, :) = omegaMotor.';
    tauArmLog(k, :) = aux.tauArm_b(:).';
    tauPayloadLog(k, :) = aux.tauPayload_b(:).';
    armComLog(k, :) = aux.kinematics.armCom_b(:).';
    gripCenterLog(k, :) = aux.kinematics.gripCenter_b(:).';
end

log = struct();
log.time = time;
log.state = bodyState;
log.jointPos = jointPos;
log.jointVel = jointVel;
log.jointTarget = jointTarget;
log.jointTau = jointTau;
log.reference = refLog;
log.wrench = wrenchLog;
log.motorOmega = omegaLog;
log.scenario = scenario;
log.controller = char(controller);
log.coupling = struct();
log.coupling.tauArm = tauArmLog;
log.coupling.tauPayload = tauPayloadLog;
log.coupling.armCom = armComLog;
log.coupling.gripCenter = gripCenterLog;
end

function ref = reference_at(t, scenario)
ref = struct();
ref.yaw = 0.0;
ref.yawRate = 0.0;
switch string(scenario)
    case "arm_reach_payload"
        [z1, zd1, zdd1] = smooth_transition(t, 0.2, 1.0, 0.5, 2.0);
        [z2, zd2, zdd2] = smooth_transition(t, 0.0, 0.15, 2.4, 1.2);
        z = z1 + z2;
        ref.pos = [0; 0; z];
        ref.vel = [0; 0; zd1 + zd2];
        ref.acc = [0; 0; zdd1 + zdd2];
    otherwise
        error("flyarm_coupled_tracking:UnknownScenario", "Unknown scenario '%s'.", scenario);
end
end

function qCmd = arm_reference_at(t, P, scenario)
switch string(scenario)
    case "arm_reach_payload"
        qStow = P.articulation.qStow(:);
        qOpen = P.articulation.qOpen(:);
        qStraight = P.articulation.qStraight(:);
        qReachOpen = qStraight;
        qReachOpen(5:6) = qOpen(5:6);
        if t < 0.8
            qCmd = qStow;
        elseif t < 1.2
            frac = smoothstep01((t - 0.8) / 0.4);
            qCmd = (1 - frac) * qStow + frac * qOpen;
        elseif t < 2.0
            frac = smoothstep01((t - 1.2) / 0.8);
            qCmd = (1 - frac) * qOpen + frac * qReachOpen;
        elseif t < 2.4
            frac = smoothstep01((t - 2.0) / 0.4);
            qCmd = (1 - frac) * qReachOpen + frac * qStraight;
        else
            qCmd = qStraight;
        end
    otherwise
        error("flyarm_coupled_tracking:UnknownScenario", "Unknown scenario '%s'.", scenario);
end
end

function disturbances = disturbance_at(t, scenario, P)
disturbances = struct();
disturbances.force_b = zeros(3, 1);
disturbances.tau_arm_b = zeros(3, 1);
disturbances.tau_payload_b = zeros(3, 1);
disturbances.payloadMass = P.coupling.payloadMassDefault;
disturbances.payloadAttached = string(scenario) == "arm_reach_payload" && t >= 2.2;
end

function y = wrap_pi(x)
y = mod(x + pi, 2 * pi) - pi;
end

function s = smoothstep01(r)
r = min(max(r, 0), 1);
s = r * r * (3 - 2 * r);
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
