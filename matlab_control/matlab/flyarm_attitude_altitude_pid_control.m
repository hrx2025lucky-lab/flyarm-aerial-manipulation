function [wrench, aux, state] = flyarm_attitude_altitude_pid_control(~, x, ref, P, state)
%FLYARM_ATTITUDE_ALTITUDE_PID_CONTROL Flight PID baseline controller.
%
% This controller upgrades the previous transparent flight PD baseline with
% bounded integral terms. Arm and gripper joints remain handled separately by
% flyarm_articulated_pd_control.

if nargin < 5 || isempty(state)
    state = flyarm_pid_init(P, 0.01);
end

x = x(:);
pos = x(1:3);
vel = x(4:6);
eul = x(7:9);
omega = x(10:12);

ref = fill_ref_defaults(ref);
gains = state.gains;
dt = state.dt;

posErr = ref.pos - pos;
velErr = ref.vel - vel;
state.intPos = clamp_vec(state.intPos + dt .* posErr, -gains.maxIntPos, gains.maxIntPos);
accCmd = ref.acc + gains.KpPos .* posErr + gains.KiPos .* state.intPos + gains.KdPos .* velErr;
accCmd = clamp_vec(accCmd, -gains.maxAcc, gains.maxAcc);

yaw = ref.yaw;
rollDes = (accCmd(1) * sin(yaw) - accCmd(2) * cos(yaw)) / P.gravity;
pitchDes = (accCmd(1) * cos(yaw) + accCmd(2) * sin(yaw)) / P.gravity;
rollDes = min(max(rollDes, -P.control.maxTilt), P.control.maxTilt);
pitchDes = min(max(pitchDes, -P.control.maxTilt), P.control.maxTilt);

tiltComp = max(0.2, cos(eul(1)) * cos(eul(2)));
thrust = P.mass.total * (P.gravity + accCmd(3)) / tiltComp;
thrust = min(max(thrust, 0), P.control.maxThrust);

eulDes = [rollDes; pitchDes; yaw];
attErr = [eulDes(1:2) - eul(1:2); wrap_pi(eulDes(3) - eul(3))];
omegaDes = [0; 0; ref.yawRate];
omegaErr = omegaDes - omega;
state.intAtt = clamp_vec(state.intAtt + dt .* attErr, -gains.maxIntAtt, gains.maxIntAtt);

momentPid = gains.KpAtt .* attErr + gains.KiAtt .* state.intAtt + gains.KdAtt .* omegaErr;
moment = P.inertia.J_control * momentPid;
moment = clamp_vec(moment, -P.control.maxMoment, P.control.maxMoment);

wrench = struct();
wrench.thrust = thrust;
wrench.moment = moment;

aux = struct();
aux.accCmd = accCmd;
aux.rollDes = rollDes;
aux.pitchDes = pitchDes;
aux.attErr = attErr;
aux.omegaErr = omegaErr;
aux.intPos = state.intPos;
aux.intAtt = state.intAtt;
end

function y = clamp_vec(x, lo, hi)
y = min(max(x, lo), hi);
end

function ref = fill_ref_defaults(ref)
if ~isfield(ref, "pos"), ref.pos = zeros(3, 1); end
if ~isfield(ref, "vel"), ref.vel = zeros(3, 1); end
if ~isfield(ref, "acc"), ref.acc = zeros(3, 1); end
if ~isfield(ref, "yaw"), ref.yaw = 0.0; end
if ~isfield(ref, "yawRate"), ref.yawRate = 0.0; end
ref.pos = ref.pos(:);
ref.vel = ref.vel(:);
ref.acc = ref.acc(:);
end

function y = wrap_pi(x)
y = mod(x + pi, 2 * pi) - pi;
end
