function [wrench, aux, state] = flyarm_cascade_ladrc_control(~, x, ref, P, state)
%FLYARM_CASCADE_LADRC_CONTROL Cascade LADRC controller for flyarm.
%
% Structure:
%   position PD -> desired acceleration/thrust/attitude
%   outer LADRC: roll/pitch angle -> p/q rate setpoint
%   inner LADRC: p/q/r angular rate -> body moment

x = x(:);
pos = x(1:3);
vel = x(4:6);
eul = x(7:9);
omega = x(10:12);
ref = fill_ref_defaults(ref);
cfg = state.cfg;
dt = state.dt;

[thrust, rollDes, pitchDes, accCmd, state] = position_to_thrust_attitude(pos, vel, eul, ref, P, state);

angleRef = [rollDes; pitchDes];
angleMeas = eul(1:2);
[rateRp, state.outer] = ladrc_step(state.outer, angleRef, angleMeas, ...
    cfg.out_wc * ones(2, 1), cfg.out_wo * ones(2, 1), cfg.out_b0 * ones(2, 1), ...
    dt, cfg.z2_clamp_out * ones(2, 1));
rateRp = min(max(rateRp, -cfg.max_rate), cfg.max_rate);

rateRef = [rateRp(1); rateRp(2); ref.yawRate];
wcInner = [cfg.in_wc_rp; cfg.in_wc_rp; cfg.in_wc_yaw];
woInner = [cfg.in_wo_rp; cfg.in_wo_rp; cfg.in_wo_yaw];
[moment, state.inner] = ladrc_step(state.inner, rateRef, omega, wcInner, woInner, ...
    cfg.in_b0(:), dt, cfg.z2_clamp_inner(:));
moment = min(max(moment, -P.control.maxMoment), P.control.maxMoment);

wrench = struct();
wrench.thrust = thrust;
wrench.moment = moment;

aux = struct();
aux.accCmd = accCmd;
aux.rollDes = rollDes;
aux.pitchDes = pitchDes;
aux.posInt = state.posInt;
aux.rateRef = rateRef;
aux.z2Outer = state.outer.z2;
aux.z2Inner = state.inner.z2;
end

function [u, loop] = ladrc_step(loop, ref, y, wc, wo, b0, dt, z2Clamp)
beta1 = 2 .* wo;
beta2 = wo .* wo;
e = loop.z1 - y;
loop.z1 = loop.z1 + dt .* (loop.z2 - beta1 .* e + b0 .* loop.uPrev);
loop.z2 = loop.z2 + dt .* (-beta2 .* e);
loop.z2 = min(max(loop.z2, -z2Clamp), z2Clamp);
u = (wc .* (ref - loop.z1) - loop.z2) ./ b0;
loop.uPrev = u;
end

function [thrust, rollDes, pitchDes, accCmd, state] = position_to_thrust_attitude(pos, vel, eul, ref, P, state)
cfg = state.cfg;
posErr = ref.pos - pos;
velErr = ref.vel - vel;
state.posInt = min(max(state.posInt + state.dt .* posErr, -cfg.pos_max_int(:)), cfg.pos_max_int(:));
accCmd = ref.acc + cfg.pos_kp(:) .* posErr + cfg.pos_ki(:) .* state.posInt + cfg.pos_kd(:) .* velErr;
accCmd = min(max(accCmd, -cfg.pos_max_acc(:)), cfg.pos_max_acc(:));

yaw = ref.yaw;
rollDes = (accCmd(1) * sin(yaw) - accCmd(2) * cos(yaw)) / P.gravity;
pitchDes = (accCmd(1) * cos(yaw) + accCmd(2) * sin(yaw)) / P.gravity;
rollDes = min(max(rollDes, -P.control.maxTilt), P.control.maxTilt);
pitchDes = min(max(pitchDes, -P.control.maxTilt), P.control.maxTilt);

tiltComp = max(0.2, cos(eul(1)) * cos(eul(2)));
thrust = P.mass.total * (P.gravity + accCmd(3)) / tiltComp;
thrust = min(max(thrust, 0), P.control.maxThrust);
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
