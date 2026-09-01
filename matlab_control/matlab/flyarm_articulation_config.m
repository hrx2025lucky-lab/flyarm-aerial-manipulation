function P = flyarm_articulation_config(P)
%FLYARM_ARTICULATION_CONFIG Add Isaac-like arm/gripper control parameters.
%
% The Isaac flyarm task drives the arm and gripper through joint position
% targets. PhysX then applies implicit PD drives at the joints while rotor
% thrust/moment are applied to the base body. This function mirrors that
% logic for the MATLAB coupled model.

armNames = ["base_rotate_joint", "axis_B_joint", "axis_C_joint", "axis_D_joint"];
gripperNames = ["gripper_right_joint", "gripper_left_joint"];
jointNames = [armNames, gripperNames];

jointInfo = repmat(P.joints(1), 1, numel(jointNames));
for i = 1:numel(jointNames)
    jointInfo(i) = find_joint(P, jointNames(i));
end

lower = [jointInfo.lower].';
upper = [jointInfo.upper].';
effort = [jointInfo.effort].';
velocity = [jointInfo.velocity].';

% Isaac task settings are position-PD joint drives. In this reduced explicit
% ODE, use a critically damped equivalent arm drive to avoid reaction spikes
% that an implicit PhysX drive would normally absorb.
% gripper branch differs by URDF. The paper URDF uses prismatic fingers;
% current Isaac fallback can also use revolute fingers, so infer units.
armEquivalentInertia = [0.040; 0.025; 0.018; 0.006];
armKp = 25.0;
kp = [armKp * ones(4, 1); 0; 0];
kd = [2 * sqrt(armKp * armEquivalentInertia); 0; 0];
velocity(1:4) = min(fill_nan(velocity(1:4), 5.0), 5.0);
effort(1:4) = fill_nan(effort(1:4), 10.0);

gripperTypes = string({jointInfo(5:6).type});
if all(gripperTypes == "prismatic")
    kp(5:6) = 2000.0;        % N/m, Franka-like parallel gripper scale
    kd(5:6) = 100.0;         % Ns/m
    effort(5:6) = fill_nan(effort(5:6), 20.0);
    velocity(5:6) = fill_nan(velocity(5:6), 0.1);
    qGripClose = lower(5:6); % current paper URDF: 0 = closed
    qGripOpen = upper(5:6);
else
    kp(5:6) = 50.0;          % Nm/rad, Isaac revolute fallback
    kd(5:6) = 2.0;           % Nms/rad
    effort(5:6) = min(fill_nan(effort(5:6), 1.0), 1.0);
    velocity(5:6) = fill_nan(velocity(5:6), 5.0);
    qGripOpen = [0.8; 0.8];
    qGripClose = [0.20; 0.20];
end

qStowArm = [0.45; -0.41; 0.35; -0.10];
qStraightArm = [0.45; 0.0; -1.48; -0.10];
qStowArm = min(max(qStowArm, lower(1:4)), upper(1:4));
qStraightArm = min(max(qStraightArm, lower(1:4)), upper(1:4));

P.articulation = struct();
P.articulation.jointNames = jointNames(:);
P.articulation.armJointNames = armNames(:);
P.articulation.gripperJointNames = gripperNames(:);
P.articulation.joints = jointInfo;
P.articulation.qLower = lower;
P.articulation.qUpper = upper;
P.articulation.qOpen = [qStowArm; qGripOpen];
P.articulation.qClose = [qStowArm; qGripClose];
P.articulation.qStow = [qStowArm; qGripClose];
P.articulation.qStraight = [qStraightArm; qGripClose];
P.articulation.kp = kp;
P.articulation.kd = kd;
P.articulation.effort = effort;
P.articulation.velocity = velocity;
P.articulation.actuatorTimeConstant = [0.01; 0.01; 0.01; 0.01; 0.02; 0.02];

% Equivalent joint inertias/masses for the reduced ODE. These are not used
% as CAD truth; they make the Isaac PD drive dynamics explicit and bounded.
P.articulation.jointInertia = [armEquivalentInertia; 0.010; 0.010];
qddLimit = [40; 50; 60; 80; 250; 250];
if all(gripperTypes == "prismatic")
    qddLimit(5:6) = 5.0;    % m/s^2; avoids unrealistically sharp finger reaction spikes
end
P.articulation.qddLimit = qddLimit;
P.articulation.armLinkNames = ["arm_base", "arm_base_rotate", "link_bc", ...
    "link_cd", "gripper_mount", "gripper_base", "gripper_right", "gripper_left"];
P.articulation.armMass = sum_link_mass(P, P.articulation.armLinkNames);

P.coupling = struct();
% The reduced model is for controller validation, not a full constrained
% multibody solve. Use conservative gains so the equivalent reaction captures
% the direction and timing of arm disturbances without swamping the rotor model.
P.coupling.armInertiaForceGain = 0.20;
P.coupling.jointReactionGain = 0.02;
P.coupling.comGravityGain = 0.10;
P.coupling.payloadMassDefault = 0.15;

trimKin = flyarm_articulated_kinematics(P, P.articulation.qStow);
P.coupling.trimArmCom_b = trimKin.armCom_b;
P.coupling.trimTotalCom_b = trimKin.totalCom_b;
end

function joint = find_joint(P, name)
idx = find(strcmp({P.joints.name}, char(name)), 1);
if isempty(idx)
    error("flyarm_articulation_config:MissingJoint", "Missing joint %s.", name);
end
joint = P.joints(idx);
end

function values = fill_nan(values, fallback)
values(isnan(values)) = fallback;
end

function m = sum_link_mass(P, names)
m = 0.0;
for i = 1:numel(names)
    idx = find(strcmp({P.links.name}, char(names(i))), 1);
    if ~isempty(idx)
        m = m + P.links(idx).mass;
    end
end
end
