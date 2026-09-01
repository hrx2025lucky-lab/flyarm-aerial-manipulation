function terms = flyarm_link_kinematic_terms(P, q)
%FLYARM_LINK_KINEMATIC_TERMS URDF-based link COM and joint Jacobian data.
%
% q order follows P.articulation.jointNames:
%   [base_rotate, axis_B, axis_C, axis_D, gripper_right, gripper_left].
%
% This helper is intentionally kinematic only. Dynamics functions use these
% transforms to build M(q), C(q,qdot)qdot and G(q) from URDF mass/inertia.

if ~isfield(P, "articulation")
    error("flyarm_link_kinematic_terms:MissingConfig", ...
        "Call flyarm_articulation_config(P) before requesting link terms.");
end

q = q(:);
if numel(q) ~= 6
    error("flyarm_link_kinematic_terms:BadState", "Expected 6 joint values.");
end

activeNames = string(P.articulation.jointNames(:));
jointAxis = zeros(3, 6);
jointOrigin = zeros(3, 6);
jointType = strings(6, 1);

allBodies = empty_body();
armBodies = empty_body();

    function add_body(linkName, Tlink, activeMask, isArm)
        link = find_link(P, linkName);
        if link.mass <= 0
            return;
        end
        entry = struct();
        entry.name = string(linkName);
        entry.mass = link.mass;
        entry.com_b = transform_point(Tlink, link.inertial_xyz(:));
        entry.R_b = Tlink(1:3, 1:3);
        entry.inertia_b = entry.R_b * link.inertia * entry.R_b.';
        entry.active = logical(activeMask(:));
        if isempty(allBodies)
            allBodies = entry;
        else
            allBodies(end + 1) = entry; %#ok<AGROW>
        end
        if isArm
            if isempty(armBodies)
                armBodies = entry;
            else
                armBodies(end + 1) = entry; %#ok<AGROW>
            end
        end
    end

Tbase = eye(4);
add_body("base_link", Tbase, false(6, 1), false);

for i = 1:numel(P.rotors.names)
    rotorJoint = find_joint(P, P.rotors.names(i));
    Trotor = child_transform(Tbase, rotorJoint, 0.0);
    add_body(string(rotorJoint.child), Trotor, false(6, 1), false);
end

TarmBase = child_transform(Tbase, find_joint(P, "arm_mount_joint"), 0.0);
add_body("arm_base", TarmBase, false(6, 1), true);

activeMask = false(6, 1);
[T, jointOrigin(:, 1), jointAxis(:, 1)] = active_child_transform(TarmBase, find_joint(P, "base_rotate_joint"), q(1));
jointType(1) = "revolute";
activeMask(1) = true;
add_body("arm_base_rotate", T, activeMask, true);

[T, jointOrigin(:, 2), jointAxis(:, 2)] = active_child_transform(T, find_joint(P, "axis_B_joint"), q(2));
jointType(2) = "revolute";
activeMask(2) = true;
add_body("link_bc", T, activeMask, true);

[T, jointOrigin(:, 3), jointAxis(:, 3)] = active_child_transform(T, find_joint(P, "axis_C_joint"), q(3));
jointType(3) = "revolute";
activeMask(3) = true;
add_body("link_cd", T, activeMask, true);

[T, jointOrigin(:, 4), jointAxis(:, 4)] = active_child_transform(T, find_joint(P, "axis_D_joint"), q(4));
jointType(4) = "revolute";
activeMask(4) = true;
add_body("gripper_mount", T, activeMask, true);

TgripperBase = child_transform(T, find_joint(P, "parallel_gripper_mount_joint"), 0.0);
add_body("gripper_base", TgripperBase, activeMask, true);

jRight = find_joint(P, activeNames(5));
[Tright, jointOrigin(:, 5), jointAxis(:, 5)] = active_child_transform(TgripperBase, jRight, q(5));
jointType(5) = string(jRight.type);
rightMask = activeMask;
rightMask(5) = true;
add_body("gripper_right", Tright, rightMask, true);

jLeft = find_joint(P, activeNames(6));
[Tleft, jointOrigin(:, 6), jointAxis(:, 6)] = active_child_transform(TgripperBase, jLeft, q(6));
jointType(6) = string(jLeft.type);
leftMask = activeMask;
leftMask(6) = true;
add_body("gripper_left", Tleft, leftMask, true);

terms = struct();
terms.q = q;
terms.allBodies = allBodies;
terms.armBodies = armBodies;
terms.jointAxis_b = jointAxis;
terms.jointOrigin_b = jointOrigin;
terms.jointType = jointType;
terms.transforms = struct("gripperBase", TgripperBase, "rightFinger", Tright, "leftFinger", Tleft);
end

function bodies = empty_body()
bodies = struct("name", {}, "mass", {}, "com_b", {}, "R_b", {}, "inertia_b", {}, "active", {});
end

function [Tchild, origin_b, axis_b] = active_child_transform(Tparent, joint, q)
Torigin = Tparent * origin_transform(joint);
axis = normalized_axis(joint.axis(:));
origin_b = Torigin(1:3, 4);
axis_b = Torigin(1:3, 1:3) * axis;
Tchild = Torigin * motion_transform(joint, q, axis);
end

function Tchild = child_transform(Tparent, joint, q)
axis = normalized_axis(joint.axis(:));
Tchild = Tparent * origin_transform(joint) * motion_transform(joint, q, axis);
end

function axis = normalized_axis(axis)
if norm(axis) < 1e-12
    axis = [1; 0; 0];
else
    axis = axis / norm(axis);
end
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
    error("flyarm_link_kinematic_terms:MissingJoint", "Missing joint %s.", name);
end
joint = P.joints(idx);
end

function link = find_link(P, name)
idx = find(strcmp({P.links.name}, char(name)), 1);
if isempty(idx)
    error("flyarm_link_kinematic_terms:MissingLink", "Missing link %s.", name);
end
link = P.links(idx);
end
