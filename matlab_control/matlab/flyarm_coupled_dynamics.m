function [xdot, aux] = flyarm_coupled_dynamics(t, x, u, qCmd, P, disturbances)
%FLYARM_COUPLED_DYNAMICS Quadrotor + arm/gripper reduced coupled dynamics.
%
% State:
%   x = [quad_12; q_arm_gripper(6); qd_arm_gripper(6)].
%
% The model follows the Isaac abstraction:
%   rotor wrench acts on the base body;
%   arm/gripper joints are position-PD drives;
%   arm motion and payload contact appear as reaction force/moment on base.

if nargin < 6
    disturbances = struct();
end

if ~isfield(P, "articulation")
    error("flyarm_coupled_dynamics:MissingConfig", ...
        "Call flyarm_articulation_config(P) before coupled dynamics.");
end

x = x(:);
if numel(x) ~= 24
    error("flyarm_coupled_dynamics:BadState", "Expected 24x1 state [quad; q; qd].");
end

xBody = x(1:12);
q = x(13:18);
qd = x(19:24);
qCmd = qCmd(:);

[jointCmd, jointAux] = flyarm_articulated_pd_control(q, qd, qCmd, P);
jointTauDrive = get_field_or(disturbances, "jointTauActual", jointCmd.tau);
jointTauDrive = min(max(jointTauDrive(:), -P.articulation.effort(:)), P.articulation.effort(:));
actuatorTau = max(P.articulation.actuatorTimeConstant(:), eps);
jointTauDot = (jointCmd.tau - jointTauDrive) ./ actuatorTau;

qdd = jointTauDrive ./ P.articulation.jointInertia(:);
qdd = min(max(qdd, -P.articulation.qddLimit(:)), P.articulation.qddLimit(:));

kin = flyarm_articulated_kinematics(P, q);
Jcom = arm_com_jacobian(P, q);
armComAcc_b = Jcom * qdd;
forceArm_b = -P.coupling.armInertiaForceGain * P.articulation.armMass * armComAcc_b;
tauArm_b = cross(kin.armCom_b, forceArm_b) + ...
    P.coupling.jointReactionGain * joint_reaction_moment(kin, jointTauDrive);

R = rotm_zyx(xBody(7), xBody(8), xBody(9));
gravityTotal_b = R.' * [0; 0; -P.mass.total * P.gravity];
rComDelta_b = kin.totalCom_b - P.coupling.trimTotalCom_b;
tauComGravity_b = P.coupling.comGravityGain * cross(rComDelta_b, gravityTotal_b);

[forcePayload_b, tauPayload_b] = payload_wrench_b(P, kin, R, disturbances);

baseDist = struct();
baseDist.force_b = get_field_or(disturbances, "force_b", zeros(3, 1)) + forceArm_b + forcePayload_b;
baseDist.tau_arm_b = get_field_or(disturbances, "tau_arm_b", zeros(3, 1)) + tauArm_b + tauComGravity_b;
baseDist.tau_payload_b = get_field_or(disturbances, "tau_payload_b", zeros(3, 1)) + tauPayload_b;

xBodyDot = flyarm_quad_dynamics(t, xBody, u, P, baseDist);

xdot = zeros(24, 1);
xdot(1:12) = xBodyDot;
xdot(13:18) = qd;
xdot(19:24) = qdd;

aux = struct();
aux.kinematics = kin;
aux.joint = jointAux;
aux.jointTau = jointTauDrive;
aux.jointTauCommand = jointCmd.tau;
aux.jointTauDot = jointTauDot;
aux.qdd = qdd;
aux.forceArm_b = forceArm_b;
aux.tauArm_b = tauArm_b + tauComGravity_b;
aux.forcePayload_b = forcePayload_b;
aux.tauPayload_b = tauPayload_b;
aux.tauComGravity_b = tauComGravity_b;
end

function J = arm_com_jacobian(P, q)
h = 1e-5;
J = zeros(3, 6);
for i = 1:6
    dq = zeros(6, 1);
    dq(i) = h;
    kp = flyarm_articulated_kinematics(P, q + dq);
    km = flyarm_articulated_kinematics(P, q - dq);
    J(:, i) = (kp.armCom_b - km.armCom_b) / (2 * h);
end
end

function tau_b = joint_reaction_moment(kin, jointTau)
tau_b = zeros(3, 1);
for i = 1:6
    axis = kin.jointAxis_b(:, i);
    if norm(axis) < 1e-12
        continue;
    end
    axis = axis / norm(axis);
    type = kin.jointType(i);
    if type == "revolute" || type == "continuous"
        tau_b = tau_b - jointTau(i) * axis;
    elseif type == "prismatic"
        f_b = -jointTau(i) * axis;
        tau_b = tau_b + cross(kin.jointOrigin_b(:, i), f_b);
    end
end
end

function [force_b, tau_b] = payload_wrench_b(P, kin, R, disturbances)
attached = get_field_or(disturbances, "payloadAttached", false);
payloadMass = get_field_or(disturbances, "payloadMass", P.coupling.payloadMassDefault);
force_b = zeros(3, 1);
tau_b = zeros(3, 1);
if attached && payloadMass > 0
    force_b = R.' * [0; 0; -payloadMass * P.gravity];
    tau_b = cross(kin.gripCenter_b, force_b);
end
end

function R = rotm_zyx(roll, pitch, yaw)
cr = cos(roll); sr = sin(roll);
cp = cos(pitch); sp = sin(pitch);
cy = cos(yaw); sy = sin(yaw);
Rz = [cy, -sy, 0; sy, cy, 0; 0, 0, 1];
Ry = [cp, 0, sp; 0, 1, 0; -sp, 0, cp];
Rx = [1, 0, 0; 0, cr, -sr; 0, sr, cr];
R = Rz * Ry * Rx;
end

function value = get_field_or(s, name, fallback)
if isfield(s, name)
    value = s.(name);
else
    value = fallback;
end
end
