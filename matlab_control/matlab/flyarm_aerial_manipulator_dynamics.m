function dyn = flyarm_aerial_manipulator_dynamics(P, eul, q, nu, opts)
%FLYARM_AERIAL_MANIPULATOR_DYNAMICS Full flyarm coupled dynamics terms.
%
% Layer 2 model in body generalized velocity coordinates:
%   M(chi) nudot + C(chi,nu)nu + G(chi) = tau_rotor + tau_joint + Jc' Fc + d
%
% chi = [p_w; roll pitch yaw; q_arm], nu = [v_b; omega_b; qdot].
% The mass matrix is assembled from all URDF link COM Jacobians. Its block
% structure exposes the body-arm coupling terms used in aerial manipulator
% papers, while the existing Isaac/RL stack remains responsible for complex
% contact-rich grasping.

if nargin < 5
    opts = struct();
end
if ~isfield(P, "articulation")
    error("flyarm_aerial_manipulator_dynamics:MissingConfig", ...
        "Call flyarm_articulation_config(P) before coupled dynamics terms.");
end

eul = eul(:);
q = q(:);
nu = nu(:);
if numel(eul) ~= 3 || numel(q) ~= 6 || numel(nu) ~= 12
    error("flyarm_aerial_manipulator_dynamics:BadState", ...
        "Expected eul 3x1, q 6x1 and nu=[v_b;omega_b;qdot] 12x1.");
end

R = rotm_zyx(eul(1), eul(2), eul(3));
gravity_b = get_field_or(opts, "gravity_b", R.' * [0; 0; -P.gravity]);
fdStep = get_field_or(opts, "finiteDifferenceStep", 1e-5);

terms = flyarm_link_kinematic_terms(P, q);
M = aerial_mass_matrix(P, terms);
G = aerial_gravity_vector(terms, gravity_b);
Cnu = aerial_velocity_product(P, q, nu, M, fdStep);

kin = flyarm_articulated_kinematics(P, q);
JvGrip = point_joint_jacobian(terms, kin.gripCenter_b);
JcGrip_b = [eye(3), -skew(kin.gripCenter_b), JvGrip];

dyn = struct();
dyn.M = M;
dyn.Cnu = Cnu;
dyn.G = G;
dyn.terms = terms;
dyn.blocks = struct();
dyn.blocks.M_bb = M(1:6, 1:6);
dyn.blocks.M_ba = M(1:6, 7:12);
dyn.blocks.M_ab = M(7:12, 1:6);
dyn.blocks.M_aa = M(7:12, 7:12);
dyn.inputMap = struct();
dyn.inputMap.rotorWrench = [eye(6); zeros(6, 6)];
dyn.inputMap.jointTorque = [zeros(6, 6); eye(6)];
dyn.inputMap.Jc_grip_b = JcGrip_b;
dyn.equation = "M(chi) nudot + C(chi,nu)nu + G(chi) = tau_rotor + tau_joint + J_c^T F_c + d";
end

function M = aerial_mass_matrix(P, terms)
M = zeros(12, 12);
for i = 1:numel(terms.allBodies)
    body = terms.allBodies(i);
    [Jv, Jw] = generalized_jacobians_for_body(terms, body);
    M = M + body.mass * (Jv.' * Jv) + Jw.' * body.inertia_b * Jw;
end

if isfield(P.articulation, "jointInertia")
    M(7:12, 7:12) = M(7:12, 7:12) + diag(1e-3 * P.articulation.jointInertia(:));
end
M = 0.5 * (M + M.') + 1e-10 * eye(12);
end

function G = aerial_gravity_vector(terms, gravity_b)
G = zeros(12, 1);
for i = 1:numel(terms.allBodies)
    body = terms.allBodies(i);
    [Jv, ~] = generalized_jacobians_for_body(terms, body);
    gravityForce_b = body.mass * gravity_b(:);
    G = G - Jv.' * gravityForce_b;
end
end

function Cnu = aerial_velocity_product(P, q, nu, M, h)
v_b = nu(1:3);
omega_b = nu(4:6);
qdot = nu(7:12);

Cnu = zeros(12, 1);

% Body-frame Newton-Euler transport terms. These are the dominant velocity
% products for the floating base representation used here.
Cnu(1:3) = cross(omega_b, M(1:3, 1:3) * v_b);
angularMomentum = M(4:6, 4:6) * omega_b + M(4:6, 7:12) * qdot;
Cnu(4:6) = cross(omega_b, angularMomentum);

Cnu = Cnu + configuration_velocity_product(P, q, nu, h);
end

function Cnu = configuration_velocity_product(P, q, nu, h)
dM = zeros(12, 12, 6);
for k = 1:6
    dq = zeros(6, 1);
    dq(k) = h;
    Mp = aerial_mass_matrix(P, flyarm_link_kinematic_terms(P, q + dq));
    Mm = aerial_mass_matrix(P, flyarm_link_kinematic_terms(P, q - dq));
    dM(:, :, k) = (Mp - Mm) / (2 * h);
end

Cnu = zeros(12, 1);
for i = 1:12
    acc = 0.0;
    for j = 1:12
        for k = 1:12
            christoffel = 0.5 * (mass_derivative(dM, i, j, k) ...
                + mass_derivative(dM, i, k, j) ...
                - mass_derivative(dM, j, k, i));
            acc = acc + christoffel * nu(j) * nu(k);
        end
    end
    Cnu(i) = acc;
end
end

function value = mass_derivative(dM, row, col, coordIndex)
if coordIndex <= 6
    value = 0.0;
else
    value = dM(row, col, coordIndex - 6);
end
end

function [Jv, Jw] = generalized_jacobians_for_body(terms, body)
Jv = zeros(3, 12);
Jw = zeros(3, 12);
Jv(:, 1:3) = eye(3);
Jv(:, 4:6) = -skew(body.com_b);
Jw(:, 4:6) = eye(3);

[Jv_q, Jw_q] = joint_jacobians_for_point(terms, body.com_b, body.active);
Jv(:, 7:12) = Jv_q;
Jw(:, 7:12) = Jw_q;
end

function Jv = point_joint_jacobian(terms, point_b)
Jv = zeros(3, 6);
for j = 1:6
    axis = terms.jointAxis_b(:, j);
    if norm(axis) < 1e-12
        continue;
    end
    axis = axis / norm(axis);
    jointType = terms.jointType(j);
    if jointType == "revolute" || jointType == "continuous"
        Jv(:, j) = cross(axis, point_b - terms.jointOrigin_b(:, j));
    elseif jointType == "prismatic"
        Jv(:, j) = axis;
    end
end
end

function [Jv, Jw] = joint_jacobians_for_point(terms, point_b, activeMask)
Jv = zeros(3, 6);
Jw = zeros(3, 6);
for j = 1:6
    if ~activeMask(j)
        continue;
    end
    axis = terms.jointAxis_b(:, j);
    if norm(axis) < 1e-12
        continue;
    end
    axis = axis / norm(axis);
    jointType = terms.jointType(j);
    if jointType == "revolute" || jointType == "continuous"
        Jv(:, j) = cross(axis, point_b - terms.jointOrigin_b(:, j));
        Jw(:, j) = axis;
    elseif jointType == "prismatic"
        Jv(:, j) = axis;
    end
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

function S = skew(v)
S = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
end

function value = get_field_or(s, name, fallback)
if isfield(s, name)
    value = s.(name);
else
    value = fallback;
end
end
