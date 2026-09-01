function [B, mix] = flyarm_motor_mixer(P)
%FLYARM_MOTOR_MIXER Build rotor-speed-squared to wrench allocation matrix.
%
% Rotor order is P.rotors.names, currently [fl fr rr rl]. The body-frame
% convention follows the flyarm URDF notes: x forward, y left, z up. Each
% rotor force is along +body z, so r x F contributes [y*f; -x*f; 0].

kf = P.motor.kf;
km = P.motor.km;
r = P.rotors.position_b;
spin = P.rotors.spin_dir(:);

B = zeros(4, 4);
for i = 1:4
    x = r(i, 1);
    y = r(i, 2);
    B(:, i) = [kf; y * kf; -x * kf; spin(i) * km];
end

mix = struct();
mix.B = B;
mix.kf = kf;
mix.km = km;
mix.hoverOmega = P.motor.hoverOmega;
mix.maxOmega = P.motor.maxOmega;
mix.rotorNames = P.rotors.names;
end
