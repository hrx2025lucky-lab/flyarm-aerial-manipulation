function [cmd, aux] = flyarm_articulated_pd_control(q, qd, qCmd, P, qdCmd)
%FLYARM_ARTICULATED_PD_CONTROL Isaac-like joint position PD drive.
%
% Isaac uses implicit joint drives. This reduced MATLAB version exposes the
% same idea explicitly: target position -> bounded joint torque/force.

arguments
    q (:, 1) double
    qd (:, 1) double
    qCmd (:, 1) double
    P struct
    qdCmd (:, 1) double = zeros(size(q))
end

if ~isfield(P, "articulation")
    error("flyarm_articulated_pd_control:MissingConfig", ...
        "Call flyarm_articulation_config(P) before joint PD control.");
end

q = q(:);
qd = qd(:);
qCmd = qCmd(:);
qdCmd = qdCmd(:);
if numel(q) ~= 6 || numel(qd) ~= 6 || numel(qCmd) ~= 6 || numel(qdCmd) ~= 6
    error("flyarm_articulated_pd_control:BadSize", "Expected 6x1 q, qd, qCmd, and qdCmd.");
end

qCmd = min(max(qCmd, P.articulation.qLower), P.articulation.qUpper);
qError = qCmd - q;
qdError = qdCmd - qd;
rawTau = P.articulation.kp(:) .* qError + P.articulation.kd(:) .* qdError;
tau = min(max(rawTau, -P.articulation.effort(:)), P.articulation.effort(:));

cmd = struct();
cmd.qCmd = qCmd;
cmd.qdCmd = qdCmd;
cmd.tau = tau;

aux = struct();
aux.qError = qError;
aux.qdError = qdError;
aux.rawTau = rawTau;
aux.saturated = abs(rawTau) > P.articulation.effort(:);
end
