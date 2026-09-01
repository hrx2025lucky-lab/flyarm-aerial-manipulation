function [wrench, aux] = flyarm_attitude_altitude_control(~, x, ref, P, gains)
%FLYARM_ATTITUDE_ALTITUDE_CONTROL Position/altitude and attitude baseline.
%
% The controller is intentionally classical and transparent for thesis
% validation. RL can later replace the reference generator, while this block
% remains the low-level tracking controller.

if nargin < 5 || isempty(gains)
    gains = default_gains();
end

x = x(:);
pos = x(1:3);
vel = x(4:6);
eul = x(7:9);
omega = x(10:12);

ref = fill_ref_defaults(ref);

posErr = ref.pos - pos;
velErr = ref.vel - vel;
accCmd = ref.acc + gains.KpPos .* posErr + gains.KdPos .* velErr;
accCmd = min(max(accCmd, -gains.maxAcc), gains.maxAcc);

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

moment = P.inertia.J_control * (gains.KpAtt .* attErr + gains.KdAtt .* omegaErr);
moment = min(max(moment, -P.control.maxMoment), P.control.maxMoment);

wrench = struct();
wrench.thrust = thrust;
wrench.moment = moment;

aux = struct();
aux.accCmd = accCmd;
aux.rollDes = rollDes;
aux.pitchDes = pitchDes;
aux.attErr = attErr;
aux.omegaErr = omegaErr;
end

function gains = default_gains()
gains = struct();
gains.KpPos = [1.0; 1.0; 2.0];
gains.KdPos = [1.2; 1.2; 1.8];
gains.KpAtt = [5.0; 5.0; 2.5];
gains.KdAtt = [1.6; 1.6; 0.8];
gains.maxAcc = [3.0; 3.0; 4.0];
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
