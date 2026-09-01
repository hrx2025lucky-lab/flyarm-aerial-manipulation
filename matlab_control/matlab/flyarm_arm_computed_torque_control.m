function cmd = flyarm_arm_computed_torque_control(P, q, qd, qRef, qdRef, qddRef, gains)
%FLYARM_ARM_COMPUTED_TORQUE_CONTROL Model-based arm/gripper joint control.
%
% Control law:
%   tau = M_a(q) v + C_a(q,qdot)qdot + G_a(q)
%   v   = qddot_ref + Kd(qdot_ref-qdot) + Kp(q_ref-q)
%
% This is provided for MATLAB/Simulink dynamics validation. The Isaac Lab
% policy can still produce joint targets; this controller explains how a
% model-based low-level arm controller would turn those targets into effort.

if nargin < 7 || isempty(gains)
    gains = default_gains();
end
if ~isfield(P, "articulation")
    error("flyarm_arm_computed_torque_control:MissingConfig", ...
        "Call flyarm_articulation_config(P) before computed-torque control.");
end

q = q(:);
qd = qd(:);
qRef = qRef(:);
qdRef = qdRef(:);
qddRef = qddRef(:);
if any([numel(q), numel(qd), numel(qRef), numel(qdRef), numel(qddRef)] ~= 6)
    error("flyarm_arm_computed_torque_control:BadState", ...
        "Expected 6x1 q, qd, qRef, qdRef and qddRef.");
end

qRef = min(max(qRef, P.articulation.qLower), P.articulation.qUpper);
posErr = qRef - q;
velErr = qdRef - qd;
v = qddRef + gains.Kd(:) .* velErr + gains.Kp(:) .* posErr;

dyn = flyarm_manipulator_dynamics(P, q, qd);
rawTau = dyn.M * v + dyn.Cqd + dyn.G;
tau = min(max(rawTau, -P.articulation.effort(:)), P.articulation.effort(:));

cmd = struct();
cmd.tau = tau;
cmd.rawTau = rawTau;
cmd.v = v;
cmd.posErr = posErr;
cmd.velErr = velErr;
cmd.dynamics = dyn;
cmd.saturated = abs(rawTau) > P.articulation.effort(:);
end

function gains = default_gains()
gains = struct();
gains.Kp = [30; 30; 30; 20; 80; 80];
gains.Kd = [8; 8; 8; 4; 12; 12];
end
