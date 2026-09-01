function kin = flyarm_articulated_kinematics(P, q)
%FLYARM_ARTICULATED_KINEMATICS Reduced URDF kinematics for arm/gripper.
%
% q order:
%   [base_rotate, axis_B, axis_C, axis_D, gripper_right, gripper_left].

if ~isfield(P, "articulation")
    error("flyarm_articulated_kinematics:MissingConfig", ...
        "Call flyarm_articulation_config(P) before articulated kinematics.");
end

q = q(:);
if numel(q) ~= 6
    error("flyarm_articulated_kinematics:BadState", "Expected 6 joint values.");
end

activeNames = string(P.articulation.jointNames(:));
jointAxis = zeros(3, 6);
jointOrigin = zeros(3, 6);
jointType = strings(6, 1);

armMoment = zeros(3, 1);
armMass = 0.0;
totalMoment = zeros(3, 1);
totalMass = 0.0;

    function add_total_link(linkName, Tlink)
        link = find_link(P, linkName);
        if link.mass <= 0
            return;
        end
        c = transform_point(Tlink, link.inertial_xyz(:));
        totalMoment = totalMoment + link.mass * c;
        totalMass = totalMass + link.mass;
    end

    function add_arm_link(linkName, Tlink)
        link = find_link(P, linkName);
        if link.mass <= 0
            return;
        end
        c = transform_point(Tlink, link.inertial_xyz(:));
        armMoment = armMoment + link.mass * c;
        armMass = armMass + link.mass;
        totalMoment = totalMoment + link.mass * c;
        totalMass = totalMass + link.mass;
    end

Tbase = eye(4);
add_total_link("base_link", Tbase);

for i = 1:numel(P.rotors.names)
    j = find_joint(P, P.rotors.names(i));
    Trotor = child_transform(Tbase, j, 0.0);
    add_total_link(string(j.child), Trotor);
end

TarmBase = child_transform(Tbase, find_joint(P, "arm_mount_joint"), 0.0);
add_arm_link("arm_base", TarmBase);

[T, jointOrigin(:, 1), jointAxis(:, 1)] = active_child_transform(TarmBase, find_joint(P, "base_rotate_joint"), q(1));
jointType(1) = "revolute";
add_arm_link("arm_base_rotate", T);

[T, jointOrigin(:, 2), jointAxis(:, 2)] = active_child_transform(T, find_joint(P, "axis_B_joint"), q(2));
jointType(2) = "revolute";
add_arm_link("link_bc", T);

[T, jointOrigin(:, 3), jointAxis(:, 3)] = active_child_transform(T, find_joint(P, "axis_C_joint"), q(3));
jointType(3) = "revolute";
add_arm_link("link_cd", T);

[T, jointOrigin(:, 4), jointAxis(:, 4)] = active_child_transform(T, find_joint(P, "axis_D_joint"), q(4));
jointType(4) = "revolute";
add_arm_link("gripper_mount", T);

TgripperBase = child_transform(T, find_joint(P, "parallel_gripper_mount_joint"), 0.0);
add_arm_link("gripper_base", TgripperBase);

jRight = find_joint(P, activeNames(5));
[Tright, jointOrigin(:, 5), jointAxis(:, 5)] = active_child_transform(TgripperBase, jRight, q(5));
jointType(5) = string(jRight.type);
add_arm_link("gripper_right", Tright);

jLeft = find_joint(P, activeNames(6));
[Tleft, jointOrigin(:, 6), jointAxis(:, 6)] = active_child_transform(TgripperBase, jLeft, q(6));
jointType(6) = string(jLeft.type);
add_arm_link("gripper_left", Tleft);

if armMass > 0
    armCom = armMoment / armMass;
else
    armCom = zeros(3, 1);
end
if totalMass > 0
    totalCom = totalMoment / totalMass;
else
    totalCom = zeros(3, 1);
end

rightCom = transform_point(Tright, find_link(P, "gripper_right").inertial_xyz(:));
leftCom = transform_point(Tleft, find_link(P, "gripper_left").inertial_xyz(:));
gripCenter = 0.5 * (rightCom + leftCom);

kin = struct();
kin.q = q;
kin.armCom_b = armCom;
kin.totalCom_b = totalCom;
kin.armMass = armMass;
kin.totalMassFromLinks = totalMass;
kin.gripCenter_b = gripCenter;
kin.ee_b = transform_point(TgripperBase, [0.08; 0; 0.02]);
kin.jointAxis_b = jointAxis;
kin.jointOrigin_b = jointOrigin;
kin.jointType = jointType;
kin.transforms = struct("gripperBase", TgripperBase, "rightFinger", Tright, "leftFinger", Tleft);
end

function [Tchild, origin_b, axis_b] = active_child_transform(Tparent, joint, q)
Torigin = Tparent * origin_transform(joint);
axis = joint.axis(:);
if norm(axis) < 1e-12
    axis = [1; 0; 0];
else
    axis = axis / norm(axis);
end
origin_b = Torigin(1:3, 4);
axis_b = Torigin(1:3, 1:3) * axis;
Tchild = Torigin * motion_transform(joint, q, axis);
end

function Tchild = child_transform(Tparent, joint, q)
axis = joint.axis(:);
if norm(axis) < 1e-12
    axis = [1; 0; 0];
else
    axis = axis / norm(axis);
end
Tchild = Tparent * origin_transform(joint) * motion_transform(joint, q, axis);
end

function T = origin_transform(joint)
R = rpy_to_rotm(joint.origin_rpy(:));
T = eye(4);
T(1:3, 1:3) = R;
T(1:3, 4) = joint.origin_xyz(:);
end

function T = motion_transform(joint, q, axis)
T = eye(4);
type = string(joint.type);
if type == "revolute" || type == "continuous"
    T(1:3, 1:3) = axis_angle_to_rotm(axis, q);
elseif type == "prismatic"
    T(1:3, 4) = axis * q;
end
end

function p = transform_point(T, pLocal)
pH = T * [pLocal(:); 1];
p = pH(1:3);
end

function R = rpy_to_rotm(rpy)
r = rpy(1); p = rpy(2); y = rpy(3);
cr = cos(r); sr = sin(r);
cp = cos(p); sp = sin(p);
cy = cos(y); sy = sin(y);
Rx = [1, 0, 0; 0, cr, -sr; 0, sr, cr];
Ry = [cp, 0, sp; 0, 1, 0; -sp, 0, cp];
Rz = [cy, -sy, 0; sy, cy, 0; 0, 0, 1];
R = Rz * Ry * Rx;
end

function R = axis_angle_to_rotm(axis, angle)
axis = axis(:) / norm(axis);
x = axis(1); y = axis(2); z = axis(3);
c = cos(angle); s = sin(angle); C = 1 - c;
R = [x*x*C + c, x*y*C - z*s, x*z*C + y*s; ...
     y*x*C + z*s, y*y*C + c, y*z*C - x*s; ...
     z*x*C - y*s, z*y*C + x*s, z*z*C + c];
end

function joint = find_joint(P, name)
idx = find(strcmp({P.joints.name}, char(name)), 1);
if isempty(idx)
    error("flyarm_articulated_kinematics:MissingJoint", "Missing joint %s.", name);
end
joint = P.joints(idx);
end

function link = find_link(P, name)
idx = find(strcmp({P.links.name}, char(name)), 1);
if isempty(idx)
    error("flyarm_articulated_kinematics:MissingLink", "Missing link %s.", name);
end
link = P.links(idx);
end
