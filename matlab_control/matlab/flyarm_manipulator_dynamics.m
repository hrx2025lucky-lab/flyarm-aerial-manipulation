function dyn = flyarm_manipulator_dynamics(P, q, qd, opts)
%FLYARM_MANIPULATOR_DYNAMICS URDF-based arm/gripper joint dynamics.
%
% Layer 1 model:
%   M_a(q) qddot + C_a(q,qdot) qdot + G_a(q) = tau_a
%
% The mass and gravity terms are assembled from URDF link masses, inertias,
% joint axes and COM Jacobians. C_a(q,qdot)qdot is obtained from numerical
% Christoffel derivatives of M_a(q), which keeps the implementation close to
% the textbook manipulator dynamics form without hand-expanding symbols.

if nargin < 4
    opts = struct();
end
if ~isfield(P, "articulation")
    error("flyarm_manipulator_dynamics:MissingConfig", ...
        "Call flyarm_articulation_config(P) before manipulator dynamics.");
end

q = q(:);
qd = qd(:);
if numel(q) ~= 6 || numel(qd) ~= 6
    error("flyarm_manipulator_dynamics:BadState", "Expected 6x1 q and qd.");
end

gravity_b = get_field_or(opts, "gravity_b", [0; 0; -P.gravity]);
fdStep = get_field_or(opts, "finiteDifferenceStep", 1e-5);

terms = flyarm_link_kinematic_terms(P, q);
M = manipulator_mass_matrix(P, q);
G = manipulator_gravity_vector(P, terms, gravity_b);
Cqd = manipulator_velocity_product(P, q, qd, fdStep);

dyn = struct();
dyn.M = M;
dyn.Cqd = Cqd;
dyn.G = G;
dyn.terms = terms;
dyn.equation = "M_a(q) qddot + C_a(q,qdot) qdot + G_a(q) = tau_a";
end

function M = manipulator_mass_matrix(P, q)
terms = flyarm_link_kinematic_terms(P, q);
M = zeros(6, 6);
for i = 1:numel(terms.armBodies)
    body = terms.armBodies(i);
    [Jv, Jw] = joint_jacobians_for_body(terms, body);
    M = M + body.mass * (Jv.' * Jv) + Jw.' * body.inertia_b * Jw;
end

% Small reflected drive inertia avoids singular numerical matrices for very
% light wrist/finger links while leaving the URDF link terms dominant.
if isfield(P.articulation, "jointInertia")
    M = M + diag(1e-3 * P.articulation.jointInertia(:));
else
    M = M + 1e-8 * eye(6);
end
M = 0.5 * (M + M.');
end

function G = manipulator_gravity_vector(~, terms, gravity_b)
G = zeros(6, 1);
for i = 1:numel(terms.armBodies)
    body = terms.armBodies(i);
    [Jv, ~] = joint_jacobians_for_body(terms, body);
    gravityForce_b = body.mass * gravity_b(:);
    G = G - Jv.' * gravityForce_b;
end
end

function Cqd = manipulator_velocity_product(P, q, qd, h)
dM = zeros(6, 6, 6);
for k = 1:6
    dq = zeros(6, 1);
    dq(k) = h;
    Mp = manipulator_mass_matrix(P, q + dq);
    Mm = manipulator_mass_matrix(P, q - dq);
    dM(:, :, k) = (Mp - Mm) / (2 * h);
end

Cqd = zeros(6, 1);
for i = 1:6
    acc = 0.0;
    for j = 1:6
        for k = 1:6
            christoffel = 0.5 * (dM(i, j, k) + dM(i, k, j) - dM(j, k, i));
            acc = acc + christoffel * qd(j) * qd(k);
        end
    end
    Cqd(i) = acc;
end
end

function [Jv, Jw] = joint_jacobians_for_body(terms, body)
Jv = zeros(3, 6);
Jw = zeros(3, 6);
for j = 1:6
    if ~body.active(j)
        continue;
    end
    axis = terms.jointAxis_b(:, j);
    if norm(axis) < 1e-12
        continue;
    end
    axis = axis / norm(axis);
    jointType = terms.jointType(j);
    if jointType == "revolute" || jointType == "continuous"
        r = body.com_b - terms.jointOrigin_b(:, j);
        Jv(:, j) = cross(axis, r);
        Jw(:, j) = axis;
    elseif jointType == "prismatic"
        Jv(:, j) = axis;
    end
end
end

function value = get_field_or(s, name, fallback)
if isfield(s, name)
    value = s.(name);
else
    value = fallback;
end
end
