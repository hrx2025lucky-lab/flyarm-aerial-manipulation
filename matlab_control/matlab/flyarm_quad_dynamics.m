function xdot = flyarm_quad_dynamics(~, x, u, P, disturbances)
%FLYARM_QUAD_DYNAMICS 12-state rigid-body quadrotor dynamics.
%
% State x = [p_w(3); v_w(3); eul_zyx(roll,pitch,yaw)(3); omega_b(3)].
% Input u can contain either:
%   - u.thrust scalar and u.moment 3x1, or
%   - u.motorOmega 4x1, converted through flyarm_motor_mixer.

if nargin < 5
    disturbances = struct();
end

x = x(:);
posDot = x(4:6);
eul = x(7:9);
omega = x(10:12);

if isfield(u, "motorOmega")
    [B, ~] = flyarm_motor_mixer(P);
    wrench = B * (u.motorOmega(:) .^ 2);
    thrust = wrench(1);
    moment = wrench(2:4);
else
    thrust = u.thrust;
    moment = u.moment(:);
end

Fdist = get_field_or(disturbances, "force_b", zeros(3, 1));
tauArm = get_field_or(disturbances, "tau_arm_b", zeros(3, 1));
tauPayload = get_field_or(disturbances, "tau_payload_b", zeros(3, 1));

R = rotm_zyx(eul(1), eul(2), eul(3));
forceWorld = R * ([0; 0; thrust] + Fdist);
accWorld = forceWorld / P.mass.total + [0; 0; -P.gravity];

J = P.inertia.J_control;
omegaDot = J \ (moment + tauArm + tauPayload - cross(omega, J * omega));
eulDot = euler_rates_zyx(eul(1), eul(2)) * omega;

xdot = zeros(12, 1);
xdot(1:3) = posDot;
xdot(4:6) = accWorld;
xdot(7:9) = eulDot;
xdot(10:12) = omegaDot;
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

function E = euler_rates_zyx(roll, pitch)
cr = cos(roll); sr = sin(roll);
cp = cos(pitch);
tp = tan(pitch);
if abs(cp) < 1e-6
    cp = sign(cp) * 1e-6;
end
E = [1, sr * tp, cr * tp; 0, cr, -sr; 0, sr / cp, cr / cp];
end

function value = get_field_or(s, name, fallback)
if isfield(s, name)
    value = s.(name);
else
    value = fallback;
end
end
