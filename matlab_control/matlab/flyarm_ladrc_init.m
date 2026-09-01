function state = flyarm_ladrc_init(P, dt, cfg)
%FLYARM_LADRC_INIT Initialize MATLAB cascade LADRC observer states.

if nargin < 3 || isempty(cfg)
    cfg = struct();
end

state = struct();
state.dt = dt;
state.cfg = fill_cfg(P, cfg);
state.posInt = zeros(3, 1);
state.outer.z1 = zeros(2, 1);
state.outer.z2 = zeros(2, 1);
state.outer.uPrev = zeros(2, 1);
state.inner.z1 = zeros(3, 1);
state.inner.z2 = zeros(3, 1);
state.inner.uPrev = zeros(3, 1);
end

function cfg = fill_cfg(P, cfg)
cfg = set_default(cfg, "pos_kp", [1.0; 1.0; 2.0]);
cfg = set_default(cfg, "pos_ki", [0.02; 0.02; 0.80]);
cfg = set_default(cfg, "pos_kd", [1.2; 1.2; 1.9]);
cfg = set_default(cfg, "pos_max_acc", [3.0; 3.0; 4.0]);
cfg = set_default(cfg, "pos_max_int", [0.8; 0.8; 2.0]);
cfg = set_default(cfg, "out_wc", 2.0);
cfg = set_default(cfg, "out_wo", 8.1);
cfg = set_default(cfg, "out_b0", 1.0);
cfg = set_default(cfg, "in_wc_rp", 10.5);
cfg = set_default(cfg, "in_wo_rp", 15.0);
cfg = set_default(cfg, "in_wc_yaw", 8.0);
cfg = set_default(cfg, "in_wo_yaw", 13.0);
cfg = set_default(cfg, "max_rate", 3.0);
cfg = set_default(cfg, "z2_clamp_out", 1.0);

J = diag(P.inertia.J_control);
cfg.in_b0 = 1 ./ J(:);
cfg.z2_clamp_inner = [0.3 * cfg.in_b0(1); 0.3 * cfg.in_b0(2); 0.2 * cfg.in_b0(3)];
end

function cfg = set_default(cfg, name, value)
if ~isfield(cfg, name)
    cfg.(name) = value;
end
end
