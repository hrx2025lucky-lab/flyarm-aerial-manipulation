function run_first_version_checks()
%RUN_FIRST_VERSION_CHECKS Basic regression checks for the flyarm MATLAB model.
%
% This script is intentionally small: it verifies the first-version control
% package can load the paper-facing URDF, build rotor/motor parameters, and
% simulate the hover equilibrium without depending on Simscape.

repoRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(repoRoot, "matlab"));

urdfPath = fullfile(fileparts(repoRoot), "urdf", "flyarm_parallel_gripper.urdf");
assert(isfile(urdfPath), "Missing paper-facing parallel-gripper URDF: %s", urdfPath);

P = flyarm_params_from_urdf(urdfPath);

assert(P.mass.total > 2.5 && P.mass.total < 3.0, "Unexpected flyarm mass %.4f kg", P.mass.total);
assert(all(eig(P.inertia.J_control) > 0), "Control inertia must be positive definite");
assert(numel(P.rotors.names) == 4, "Expected four rotors");
assert(size(P.rotors.position_b, 1) == 4 && size(P.rotors.position_b, 2) == 3, "Rotor position shape mismatch");
assert(all(vecnorm(P.rotors.position_b(:, 1:2), 2, 2) > 0.15), "Rotor arms look too short");
assert(all(strcmp({P.gripper.joints.type}, "prismatic")), "Paper-facing model must use prismatic gripper joints");

[B, mix] = flyarm_motor_mixer(P);
assert(all(size(B) == [4, 4]), "Mixer matrix must be 4x4");
assert(rcond(B) > 1e-6, "Mixer matrix is close to singular");
assert(isfield(mix, "hoverOmega") && mix.hoverOmega > 0, "Missing hover motor speed");

omega0 = zeros(4, 1);
omegaCmd = mix.hoverOmega * ones(4, 1);
omega1 = flyarm_motor_model(omega0, omegaCmd, 0.02, P);
omega2 = flyarm_motor_model(omega1, omegaCmd, 0.02, P);
assert(all(omega2 > omega1), "Motor first-order response should move toward command");
assert(all(omega2 <= omegaCmd + 1e-9), "Motor response should not overshoot a constant command");

xHover = zeros(12, 1);
uHover.thrust = P.mass.total * P.gravity;
uHover.moment = zeros(3, 1);
xdot = flyarm_quad_dynamics(0.0, xHover, uHover, P);
assert(abs(xdot(6)) < 1e-9, "Hover vertical acceleration should be near zero, got %.3g", xdot(6));
assert(norm(xdot(10:12)) < 1e-9, "Hover angular acceleration should be near zero");

ref = struct();
ref.pos = [0; 0; 1.0];
ref.vel = zeros(3, 1);
ref.acc = zeros(3, 1);
ref.yaw = 0.0;
ref.yawRate = 0.0;
[wrench, aux] = flyarm_attitude_altitude_control(0.0, xHover, ref, P);
assert(wrench.thrust > 0, "Controller thrust must be positive");
assert(isfield(aux, "rollDes") && isfield(aux, "pitchDes"), "Controller diagnostics missing desired attitude");

pidState = flyarm_pid_init(P, 0.01);
[pidWrench, pidAux, pidState] = flyarm_attitude_altitude_pid_control(0.0, xHover, ref, P, pidState);
assert(pidWrench.thrust > 0, "PID thrust must be positive");
assert(all(isfinite(pidWrench.moment)), "PID moment must be finite");
assert(isfield(pidAux, "intPos") && isfield(pidState, "intAtt"), "PID diagnostics/state must expose integral terms");

log = flyarm_trajectory_tracking(P, "hover_step");
assert(isfield(log, "time") && numel(log.time) > 10, "Trajectory tracking log missing time vector");
assert(isfield(log, "state") && size(log.state, 1) == numel(log.time), "Trajectory state log shape mismatch");
assert(all(isfinite(log.state), "all"), "Trajectory tracking produced non-finite states");

ladrcState = flyarm_ladrc_init(P, 0.01);
[ladrcWrench, ladrcAux, ladrcState] = flyarm_cascade_ladrc_control(0.0, xHover, ref, P, ladrcState);
assert(ladrcWrench.thrust > 0, "LADRC thrust must be positive");
assert(all(isfinite(ladrcWrench.moment)), "LADRC moment must be finite");
assert(isfield(ladrcAux, "z2Inner"), "LADRC diagnostics must expose inner ESO disturbance estimate");
assert(isfield(ladrcState, "inner"), "LADRC state must preserve inner-loop observer state");

ladrcLog = flyarm_trajectory_tracking(P, "payload_step", controller="ladrc");
assert(isfield(ladrcLog, "controller") && strcmp(ladrcLog.controller, "ladrc"), "LADRC tracking log must identify controller");
assert(all(isfinite(ladrcLog.state), "all"), "LADRC trajectory tracking produced non-finite states");

comparison = flyarm_compare_pid_ladrc(P, "payload_step");
assert(isfield(comparison, "pid") && isfield(comparison, "ladrc"), "Comparison must include PID and LADRC metrics");
assert(comparison.ladrc.maxAttitudeNorm <= comparison.pid.maxAttitudeNorm + 1e-9, ...
    "LADRC should not have larger max attitude error than PID in payload_step comparison");

P = flyarm_articulation_config(P);
assert(numel(P.articulation.qStow) == 6, "Coupled model must expose 4 arm joints + 2 gripper joints");
assert(all(P.articulation.kp(:) > 0), "Isaac-like joint PD stiffness must be positive");
assert(all(P.articulation.kd(:) > 0), "Isaac-like joint PD damping must be positive");

qStow = P.articulation.qStow(:);
qStraight = P.articulation.qStraight(:);
kinStow = flyarm_articulated_kinematics(P, qStow);
kinStraight = flyarm_articulated_kinematics(P, qStraight);
assert(isfield(kinStow, "armCom_b") && numel(kinStow.armCom_b) == 3, "Kinematics must return arm COM");
assert(norm(kinStraight.armCom_b - kinStow.armCom_b) > 1e-3, ...
    "Arm COM should change between stowed and straight poses");

armDyn = flyarm_manipulator_dynamics(P, qStraight, zeros(6, 1));
assert(all(size(armDyn.M) == [6, 6]), "Manipulator dynamics must expose a 6x6 M_a(q)");
assert(norm(armDyn.M - armDyn.M.', "fro") < 1e-8, "Manipulator mass matrix must be symmetric");
assert(all(eig((armDyn.M + armDyn.M.') / 2) > 1e-8), "Manipulator mass matrix must be positive definite");
assert(numel(armDyn.Cqd) == 6 && all(isfinite(armDyn.Cqd)), "Manipulator C_a(q,qdot)qdot must be finite");
assert(numel(armDyn.G) == 6 && all(isfinite(armDyn.G)), "Manipulator G_a(q) must be finite");

fullDyn = flyarm_aerial_manipulator_dynamics(P, zeros(3, 1), qStraight, zeros(12, 1));
assert(all(size(fullDyn.M) == [12, 12]), "Aerial manipulator dynamics must expose a 12x12 M(chi)");
assert(norm(fullDyn.M - fullDyn.M.', "fro") < 1e-8, "Aerial manipulator mass matrix must be symmetric");
assert(all(eig((fullDyn.M + fullDyn.M.') / 2) > 1e-8), "Aerial manipulator mass matrix must be positive definite");
assert(isfield(fullDyn.blocks, "M_ba") && all(size(fullDyn.blocks.M_ba) == [6, 6]), ...
    "Aerial manipulator dynamics must expose the body-arm coupling block M_ba");
assert(norm(fullDyn.blocks.M_ba, "fro") > 1e-6, "Body-arm coupling block should be non-zero away from the trim pose");
assert(numel(fullDyn.Cnu) == 12 && all(isfinite(fullDyn.Cnu)), "C(chi,chidot)chidot must be finite");
assert(numel(fullDyn.G) == 12 && all(isfinite(fullDyn.G)), "G(chi) must be finite");

ctCmd = flyarm_arm_computed_torque_control(P, qStow, zeros(6, 1), qStraight, zeros(6, 1), zeros(6, 1));
assert(all(isfinite(ctCmd.tau)), "Computed-torque arm command must be finite");
assert(all(abs(ctCmd.tau) <= P.articulation.effort + 1e-9), "Computed-torque command must respect effort limits");

qd0 = zeros(6, 1);
[jointCmd, jointAux] = flyarm_articulated_pd_control(qStow, qd0, qStraight, P);
assert(all(isfinite(jointCmd.tau)), "Joint PD command must be finite");
assert(all(abs(jointCmd.tau) <= P.articulation.effort + 1e-9), "Joint PD command must respect effort limits");
assert(isfield(jointAux, "qError"), "Joint PD diagnostics must include qError");

xc = [zeros(12, 1); qStow; qd0];
uCoupled.thrust = P.mass.total * P.gravity;
uCoupled.moment = zeros(3, 1);
distCoupled = struct("payloadMass", 0.15, "payloadAttached", true);
xdotCoupled = flyarm_coupled_dynamics(0.0, xc, uCoupled, qStraight, P, distCoupled);
assert(numel(xdotCoupled) == 24, "Coupled dynamics state derivative must be 24x1");
assert(all(isfinite(xdotCoupled)), "Coupled dynamics produced non-finite values");
assert(norm(xdotCoupled(10:12)) > 1e-6, "Arm/payload coupling should create a body angular acceleration");

coupledLog = flyarm_coupled_tracking(P, "arm_reach_payload", controller="ladrc");
assert(isfield(coupledLog, "jointPos") && size(coupledLog.jointPos, 2) == 6, ...
    "Coupled tracking must log 6 joint positions");
assert(isfield(coupledLog, "coupling") && isfield(coupledLog.coupling, "tauArm"), ...
    "Coupled tracking must log arm coupling torque");
assert(all(isfinite(coupledLog.state), "all"), "Coupled tracking produced non-finite body states");
assert(all(isfinite(coupledLog.jointPos), "all"), "Coupled tracking produced non-finite joint states");

simResult = run_flyarm_simulink_closed_loop();
assert(isfield(simResult, "modelPath") && isfile(simResult.modelPath), "Simulink closed-loop model was not saved");
assert(isfield(simResult, "pid") && isfield(simResult, "ladrc"), "Simulink result must include PID and LADRC logs");
assert(simResult.ladrc.maxAttitudeNorm <= simResult.pid.maxAttitudeNorm + 1e-9, ...
    "Simulink LADRC should not have larger max attitude error than PID");

fprintf("All first-version flyarm MATLAB checks passed. Total mass %.4f kg, hover omega %.2f rad/s.\n", ...
    P.mass.total, mix.hoverOmega);
end
